"""FastAPI 服务：REST + 看板静态托管。"""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .app import App, build_app, startup
from .core.detector import LocalRiskDetector
from .persona.library import MissionRegistry
from .persona.strategy import Intent, Mission


class ChatRequest(BaseModel):
    peer: str = Field(default="anonymous")
    text: str
    session_id: str | None = None
    channel: str = "web"
    mission: dict[str, Any] | None = None


class MissionRequest(BaseModel):
    peer: str
    intent: str = "none"
    target: str = ""
    notes: str = ""


class DetectRequest(BaseModel):
    text: str
    caller_number: str = ""


def create_server(config_path: str | None = None,
                  profile_path: str | None = None) -> FastAPI:
    app_ctx: App = build_app(config_path, profile_path=profile_path)
    detector = LocalRiskDetector()
    registry = MissionRegistry()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await startup(app_ctx)
        yield
        await app_ctx.close()

    api = FastAPI(title="LiuEr", version="0.1.0", lifespan=lifespan)

    # ---------- 核心对话 ----------

    @api.post("/api/chat")
    async def chat(req: ChatRequest):
        channel = app_ctx.web_channel if req.channel == "web" else app_ctx.voice_channel
        from .channel.base import InboundMessage
        reply = await channel.handle_inbound(InboundMessage(
            peer=req.peer, text=req.text, session_id=req.session_id, meta={"mission": req.mission}))
        session = app_ctx.engine.get_session(reply.session_id)
        return {
            "session_id": reply.session_id,
            "reply": reply.text,
            "persona": session.persona.to_dict() if session else None,
            "stage": session.strategy.stage.value if session else None,
            "harvested": session.strategy.harvested if session else {},
            "intel_summary": session.intel_summary if session else {},
            # 短期思考：仅供审计与看板展示，不参与发送
            "thinking": session.last_thinking if session else "",
            "think_mode": session.last_think_mode if session else "",
            "think_degraded": session.last_think_degraded if session else False,
        }

    @api.post("/api/voice")
    async def voice(req: ChatRequest):
        from .channel.base import InboundMessage
        reply = await app_ctx.voice_channel.handle_inbound(
            InboundMessage(peer=req.peer, text=req.text, session_id=req.session_id))
        return {"session_id": reply.session_id, "reply": reply.text,
                "has_audio": reply.audio is not None,
                "voice_metrics": app_ctx.voice_channel.metrics_dict()}

    @api.post("/api/voice/tts")
    async def tts_test(req: ChatRequest):
        """TTS 试听：文本进、WAV（base64）出。用于装好依赖后快速验证。"""
        import base64
        from .persona.model import Accent
        text = (req.text or "").strip()
        if not text:
            raise HTTPException(400, "text 不能为空")
        accent_raw = (req.mission or {}).get("accent", "putonghua") if req.mission else "putonghua"
        try:
            accent = Accent(accent_raw)
        except ValueError:
            accent = Accent.PUTONGHUA
        tts = app_ctx.voice_channel.tts
        try:
            audio = await tts.synthesize(text, voice=tts.voice_for_accent(accent))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(503, f"合成失败（{type(tts).__name__}）: {exc}")
        is_mock = type(tts).__name__ == "MockTTS"
        return JSONResponse({
            "engine": type(tts).__name__,
            "accent": accent.value,
            "bytes": len(audio),
            "is_mock": is_mock,
            "note": "MockTTS 返回的是文本字节，非音频" if is_mock else "WAV 音频",
            "audio_base64": base64.b64encode(audio).decode(),
        })

    # ---------- 风险检测 ----------

    @api.post("/api/detect")
    async def detect(req: DetectRequest):
        if req.caller_number:
            score = detector.score_call(req.caller_number, req.text)
        else:
            score = detector.score_text(req.text)
        return {"score": score.score, "should_divert": score.should_divert,
                "signals": score.signals}

    # ---------- 情报 ----------

    @api.get("/api/intel")
    async def intel(client_id: str = "default", kind: str | None = None,
                    min_confidence: float = 0.0, limit: int = 100):
        records = app_ctx.store.query(client_id=client_id, kind=kind,
                                      min_confidence=min_confidence, limit=limit)
        return {"count": len(records), "records": [r.to_dict() for r in records]}

    @api.get("/api/intel/stats")
    async def intel_stats(client_id: str | None = None):
        return app_ctx.store.stats(client_id)

    # ---------- 会话 ----------

    @api.get("/api/sessions")
    async def sessions(limit: int = 50, status: str | None = None):
        return {"sessions": app_ctx.store.list_sessions(limit, status)}

    @api.get("/api/sessions/{session_id}")
    async def session_detail(session_id: str):
        data = app_ctx.store.get_session(session_id)
        if not data:
            raise HTTPException(404, "会话不存在")
        return data

    # ---------- 任务下发 ----------

    @api.post("/api/mission")
    async def mission(req: MissionRequest):
        try:
            intent = Intent(req.intent)
        except ValueError:
            raise HTTPException(400, f"无效 intent: {req.intent}")
        m = Mission(mission_id=uuid.uuid4().hex[:8], intent=intent,
                    target=req.target, notes=req.notes, created_at=time.time())
        registry.register(req.peer, m)
        return m.to_dict()

    @api.get("/api/mission")
    async def list_missions():
        return {"missions": registry.all()}

    # ---------- 系统 ----------

    @api.get("/api/health")
    async def health():
        llm_health = await app_ctx.llm.health()
        from .channel.voice import health_check
        voice_health = await health_check(app_ctx.cfg)
        tts = getattr(app_ctx.voice_channel, "tts", None)
        return {
            "status": "ok",
            "llm": llm_health,
            "voice": voice_health,
            # TTS 引擎状态：看板据此提示「能不能出声」
            "tts": {
                "engine": type(tts).__name__ if tts else "none",
                "ready": bool(tts) and type(tts).__name__ != "MockTTS",
            },
            "persona": app_ctx.library.stats(),
            "metrics": app_ctx.store.metrics(),
            "deployment": {
                "mode": app_ctx.cfg.get("deployment.mode"),
                "offline": app_ctx.cfg.get("deployment.offline"),
            },
        }

    @api.get("/api/persona/preview")
    async def persona_preview(count: int = 5):
        people = [p.to_dict() for p in app_ctx.library.stream(min(count, 20))]
        return {"count": len(people), "personas": people}

    @api.get("/api/profile")
    async def profile():
        """当前生效的人格档案与模态轮询状态。"""
        if not app_ctx.profile:
            return {"loaded": False,
                    "note": "未加载档案，使用人格库随机分配"}
        p = app_ctx.profile
        return {
            "loaded": True,
            "codename": p.codename,
            "path": str(p.path),
            "based_on_real_person": p.real_person,
            "opener": p.opener,
            "tactics_note": p.tactics_note,
        }

    @api.get("/api/modes")
    async def modes():
        """12 模态轮询序列与当前会话的推进情况。"""
        from .persona import MODE_CYCLE, MODE_LABELS, mode_cycle_labels
        sessions = []
        for s in app_ctx.engine._sessions.values():
            sessions.append(s.strategy.modes.to_dict())
        return {
            "cycle": mode_cycle_labels(),
            "cycle_length": len(MODE_CYCLE),
            "active_sessions": sessions[:10],
        }

    @api.get("/api/honeypot")
    async def honeypot(limit: int = 10):
        return {"enabled": app_ctx.honeypot.enabled, "size": app_ctx.honeypot.size(),
                "numbers": app_ctx.honeypot.numbers(limit)}

    # ---------- 看板 ----------

    @api.get("/")
    async def index():
        from .config import PROJECT_ROOT
        return FileResponse(PROJECT_ROOT / "web" / "index.html")

    @api.get("/api/version")
    async def version():
        from . import __version__
        return {"version": __version__}

    return api


app = None


def get_app() -> FastAPI:
    global app
    if app is None:
        app = create_server()
    return app
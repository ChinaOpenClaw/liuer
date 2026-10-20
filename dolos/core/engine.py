"""核心对话引擎。

这是整个系统的心脏：接管一次会话，驱动「人格 + 策略 + 情报」三件套。

设计要点：
1. 人格与策略分离，对应论文双层结构。人格负责「我是谁」，策略负责「现在干什么」。
2. 每轮对话后立即跑情报管道，不等会话结束——时效性优先。
3. 终止判据独立于模型，避免模型自行编造结束理由。
4. 系统提示分层注入：身份层常驻，策略层每轮刷新。
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from ..intel.extractor import IntelPipeline
from ..llm.client import OllamaClient
from ..persona.library import PersonaLibrary
from ..persona.model import Person
from ..persona.strategy import Mission, Stage, StrategyEngine, StrategyState

log = logging.getLogger(__name__)


@dataclass
class Turn:
    role: str      # scammer | agent
    text: str
    ts: float
    intel_count: int = 0
    stage: str = ""


@dataclass
class Session:
    session_id: str
    channel: str
    peer: str
    persona: Person
    strategy: StrategyState
    turns: list[Turn] = field(default_factory=list)
    intel_summary: dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    status: str = "active"

    def add_turn(self, role: str, text: str, *, intel_count: int = 0) -> None:
        self.turns.append(Turn(role=role, text=text, ts=time.time(),
                                intel_count=intel_count,
                                stage=self.strategy.stage.value))


class DolosEngine:
    def __init__(self, config, llm: OllamaClient, store, pipeline: IntelPipeline,
                 library: PersonaLibrary) -> None:
        self.cfg = config
        self.llm = llm
        self.store = store
        self.pipeline = pipeline
        self.library = library
        self.strategy_engine = StrategyEngine()

        self.max_context = int(config.get("llm.max_context_messages", 40))
        self.max_turns = int(config.get("llm.max_turns", 80))
        self.max_seconds = int(config.get("llm.max_session_seconds", 3600))
        self.client_id = "default"

        self._sessions: dict[str, Session] = {}

    # ---------- 会话生命周期 ----------

    def start_session(self, *, channel: str, peer: str,
                      mission: Mission | None = None) -> Session:
        persona = self.library.acquire(fresh=True)
        mission = mission or Mission(mission_id="default")
        strategy = StrategyState(stage=Stage.OPENER, mission=mission,
                                 started_at=time.time())
        sid = uuid.uuid4().hex[:12]
        session = Session(session_id=sid, channel=channel, peer=peer,
                          persona=persona, strategy=strategy)
        self._sessions[sid] = session
        self.store.create_session(sid, channel, peer, persona.persona_id,
                                  mission.mission_id)
        self.store.incr("sessions_total")
        log.info("会话启动 %s | 通道=%s | 人格=%s(%d岁)",
                 sid, channel, persona.name, persona.age)
        return session

    def get_session(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def end_session(self, session: Session, *, reason: str = "") -> None:
        if session.status == "ended":
            return
        session.status = "ended"
        session.ended_at = time.time()
        session.strategy.should_terminate = True
        session.strategy.terminate_reason = reason or "会话结束"
        transcript = [{"role": t.role, "text": t.text, "ts": t.ts,
                       "stage": t.stage, "intel": t.intel_count} for t in session.turns]
        self.store.update_session(
            session.session_id, status="ended", ended_at=session.ended_at,
            transcript=transcript, stage=session.strategy.stage.value,
            turn_count=len(session.turns), summary=reason,
        )
        self.store.incr("sessions_ended")
        log.info("会话结束 %s | 轮数=%d | 时长=%.0fs | 原因=%s",
                 session.session_id, len(session.turns),
                 (session.ended_at or 0) - session.started_at, reason)

    # ---------- 消息构建 ----------

    def _system_prompt(self, session: Session) -> str:
        persona = session.persona
        identity = persona.identity_prompt()
        strategy = self.strategy_engine.build_prompt(persona, session.strategy)
        return f"{identity}\n\n{strategy}"

    def _message_history(self, session: Session, limit: int | None = None) -> list[dict[str, str]]:
        limit = limit or self.max_context
        msgs: list[dict[str, str]] = [{"role": "system", "content": self._system_prompt(session)}]
        # 只保留最近若干轮，避免上下文膨胀
        recent = session.turns[-limit:]
        for t in recent:
            role = "user" if t.role == "scammer" else "assistant"
            msgs.append({"role": role, "content": t.text})
        return msgs

    # ---------- 核心回合 ----------

    async def on_scammer_message(self, session: Session, text: str) -> str:
        """收到诈骗方一句话，返回 agent 的回应。"""
        if session.status == "ended":
            return ""

        text = (text or "").strip()
        if not text:
            return ""

        session.add_turn("scammer", text)
        session.strategy.turn_count += 1

        # 情报管道：每轮都跑，时效性优先
        result = await self.pipeline.process_turn(
            text, client_id=self.client_id, session_id=session.session_id)
        session.intel_summary = result.get("tags", session.intel_summary)
        # 记录本轮套到的关键信息，供策略提示使用
        for f in result.get("findings", []):
            key = f["kind"]
            session.strategy.mark_harvested(key, f["value"])

        # 阶段推进 + 怀疑度评估（规则判定，不依赖模型）
        self._update_stage(session, text, result)

        # 终止判据
        should_stop, reason = self.strategy_engine.should_terminate(
            session.strategy, now=time.time(),
            max_seconds=self.max_seconds, max_turns=self.max_turns)
        if should_stop:
            session.add_turn("agent", "")
            self.end_session(session, reason=reason)
            return ""

        # 生成回应
        messages = self._message_history(session)
        try:
            reply = await self.llm.chat(
                messages, temperature=self.cfg.get("llm.temperature", 0.85),
                max_tokens=160)
        except Exception as exc:  # noqa: BLE001
            log.error("LLM 生成失败: %s", exc)
            reply = self._fallback_reply(session)

        reply = self._sanitize(reply)
        if reply:
            session.add_turn("agent", reply)
            self.store.incr("agent_replies")
        self._persist(session)
        return reply

    def _persist(self, session: Session) -> None:
        """增量落库。让运行中的会话也能查到完整转录。"""
        transcript = [{"role": t.role, "text": t.text, "ts": t.ts,
                       "stage": t.stage, "intel": t.intel_count} for t in session.turns]
        self.store.update_session(
            session.session_id,
            stage=session.strategy.stage.value,
            turn_count=len(session.turns),
            transcript=transcript,
        )

    def _fallback_reply(self, session: Session) -> str:
        """模型不可用时的兜底。用人格破绽生成一句拖延话术。"""
        p = session.persona
        fallbacks = [
            "啊？你等一下，我这信号不太好，你能再说一遍吗？",
            "我、我找一下老花镜，你稍等啊……",
            "我记一下，你等我拿个本子。",
            "这个……你说的是啥意思？我没太听明白。",
            "我这手机有点卡，你再说一次行吗？",
        ]
        # 用 persona_id 做稳定选择，保证同一人格表现一致
        idx = sum(ord(c) for c in p.persona_id) % len(fallbacks)
        return fallbacks[idx]

    @staticmethod
    def _sanitize(reply: str) -> str:
        """清洗模型输出：去 markdown、去 AI 自称、限长。"""
        if not reply:
            return ""
        r = reply.strip()
        # 去掉 markdown 符号
        r = re.sub(r"[*_`#>~\[\]()]+\s*", "", r)
        # 去掉常见 AI 自称（模型偶尔会破功）
        r = re.sub(r"(作为(一个)?(AI|人工智能)|我是(一个)?(AI|人工智能|机器人|助手)|"
                   r"我(只)?是(一个)?(语言)?模型|作为AI)", "", r, flags=re.I)
        r = r.strip(" :：-—")
        # 多句截断，最多 3 句
        parts = [p for p in re.split(r"(?<=[。！？!?])", r) if p.strip()]
        if len(parts) > 3:
            r = "".join(parts[:3])
        return r.strip()

    def _update_stage(self, session: Session, scammer_text: str,
                      result: dict[str, Any]) -> None:
        """基于规则推进阶段。不依赖模型，避免判断漂移。"""
        st = session.strategy
        text = scammer_text
        tags = result.get("tags", {})

        # 怀疑度：对方质疑身份 / 提及 AI
        if re.search(r"(机器人|AI|人工智能|是不是人|录音|自动|程序|机器)", text, re.I):
            st.suspicion_level = min(1.0, st.suspicion_level + 0.35)
            if st.stage != Stage.DEFLECT:
                st.stage = Stage.DEFLECT
            return

        # 进入收割阶段：对方索取账户/钱包/链接
        if tags.get("findings") or result.get("findings"):
            if st.stage in (Stage.ENGAGE, Stage.DEEP_TRUST, Stage.CURIOUS):
                st.stage = Stage.HARVEST
        # 出现权威话术
        elif "authority_impersonation" in tags.get("tactics", []) and st.stage == Stage.OPENER:
            st.stage = Stage.CURIOUS
        # 普通推进
        elif st.turn_count >= 2 and st.stage == Stage.OPENER:
            st.stage = Stage.CURIOUS
        elif st.turn_count >= 5 and st.stage in (Stage.CURIOUS,):
            st.stage = Stage.ENGAGE
        elif st.turn_count >= 8 and st.stage == Stage.ENGAGE:
            st.stage = Stage.DEEP_TRUST

        # 轮次推进（缓慢向后）
        if st.turn_count and st.turn_count % 15 == 0 and st.stage == Stage.DEEP_TRUST:
            st.stage = Stage.CLOSING

    # ---------- 流式生成（语音通道用） ----------

    async def stream_reply(self, session: Session, text: str) -> AsyncIterator[str]:
        """流式生成回应。语音通道用它实现句级 TTS 提前触发。"""
        if session.status == "ended":
            return

        text = (text or "").strip()
        if not text:
            return

        session.add_turn("scammer", text)
        session.strategy.turn_count += 1

        result = await self.pipeline.process_turn(
            text, client_id=self.client_id, session_id=session.session_id)
        session.intel_summary = result.get("tags", session.intel_summary)
        for f in result.get("findings", []):
            session.strategy.mark_harvested(f["kind"], f["value"])
        self._update_stage(session, text, result)

        should_stop, reason = self.strategy_engine.should_terminate(
            session.strategy, now=time.time(),
            max_seconds=self.max_seconds, max_turns=self.max_turns)
        if should_stop:
            self.end_session(session, reason=reason)
            return

        messages = self._message_history(session)
        buf: list[str] = []
        async for chunk in self.llm.chat_stream(messages, max_tokens=120):
            buf.append(chunk)
            yield chunk

        full = self._sanitize("".join(buf))
        if full:
            session.add_turn("agent", full)
            self.store.incr("agent_replies")
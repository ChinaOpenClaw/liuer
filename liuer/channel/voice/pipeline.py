"""语音通道：实时级联管道。

链路：音频输入 → VAD → 流式 ASR → 流式 LLM → 句级 TTS → 音频输出

关键工程点：
1. 端到端延迟预算。VAD 静音阈值 + 流式生成 + 句级提前合成是三个主要杠杆。
2. 打断处理。对方插话时必须立刻停止播放，否则暴露机器感。
3. 依赖全部可插拔。缺任一组件时降级为文本模式，不阻塞核心流程。
"""
from __future__ import annotations

import asyncio
import base64
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator

from ..base import Channel, InboundMessage, OutboundReply
from ...persona.model import Accent

log = logging.getLogger(__name__)


# ---------- 可插拔组件接口 ----------

class VADEngine(ABC):
    """语音活动检测。判断对方是否说完。"""

    @abstractmethod
    def is_speech(self, audio: bytes) -> bool: ...

    @abstractmethod
    def reset(self) -> None: ...


class ASREngine(ABC):
    """流式语音识别。"""

    @abstractmethod
    async def transcribe(self, audio_stream: AsyncIterator[bytes]) -> str: ...

    @abstractmethod
    async def transcribe_once(self, audio: bytes) -> str: ...


class TTSEngine(ABC):
    """流式语音合成。"""

    @abstractmethod
    async def synthesize(self, text: str, *, voice: str | None = None) -> bytes: ...

    @abstractmethod
    def voice_for_accent(self, accent: Accent) -> str: ...


# ---------- 具体实现 ----------

class SimpleEnergyVAD(VADEngine):
    """能量阈值 VAD。无需额外依赖，作为无 silero 时的降级方案。"""

    def __init__(self, threshold: float = 0.02, silence_seconds: float = 0.55) -> None:
        self.threshold = threshold
        self.silence_seconds = silence_seconds
        self._silent_since: float | None = None

    def is_speech(self, audio: bytes) -> bool:
        try:
            samples = [abs(b - 128) / 128.0 for b in audio]
        except Exception:  # noqa: BLE001
            return False
        if not samples:
            return False
        energy = sum(samples) / len(samples)
        now = time.time()
        if energy < self.threshold:
            if self._silent_since is None:
                self._silent_since = now
            return (now - self._silent_since) < self.silence_seconds
        self._silent_since = None
        return True

    def reset(self) -> None:
        self._silent_since = None


class MockASR(ASREngine):
    """占位 ASR。真实部署替换为 FunASR 或 faster-whisper。"""

    def __init__(self) -> None:
        self.last_text = ""

    async def transcribe(self, audio_stream: AsyncIterator[bytes]) -> str:
        async for _ in audio_stream:
            pass
        return self.last_text

    async def transcribe_once(self, audio: bytes) -> str:
        return self.last_text


class MockTTS(TTSEngine):
    """占位 TTS。真实部署替换为 edge-tts 或本地声学模型。"""

    def __init__(self, default_voice: str = "zh-CN-XiaoxiaoNeural") -> None:
        self.default_voice = default_voice
        self.calls = 0

    async def synthesize(self, text: str, *, voice: str | None = None) -> bytes:
        self.calls += 1
        return text.encode("utf-8")

    def voice_for_accent(self, accent: Accent) -> str:
        mapping = {
            Accent.PUTONGHUA: "zh-CN-XiaoxiaoNeural",
            Accent.NORTHERN: "zh-CN-YunxiNeural",
            Accent.NORTHEASTERN: "zh-CN-YunjianNeural",
            Accent.SICHUAN: "zh-CN-YunyangNeural",
            Accent.SOUTHWESTERN: "zh-CN-XiaoyiNeural",
            Accent.CANTONESE_INFLUENCED: "zh-HK-HiuMaanNeural",
        }
        return mapping.get(accent, self.default_voice)


# ---------- 管道 ----------

@dataclass
class VoiceMetrics:
    turns: int = 0
    vad_hits: int = 0
    tts_calls: int = 0
    interrupts: int = 0
    total_latency_ms: float = 0.0
    max_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.turns if self.turns else 0.0


class VoicePipeline(Channel):
    """实时语音管道。"""

    name = "voice"

    def __init__(self, engine, config, *, vad: VADEngine | None = None,
                 asr: ASREngine | None = None, tts: TTSEngine | None = None) -> None:
        super().__init__(engine, config)
        self.vad = vad or SimpleEnergyVAD(
            silence_seconds=float(config.get("channel.voice.vad.silence_threshold", 0.55)))
        self.asr = asr or MockASR()
        self.tts = tts or MockTTS(str(config.get("channel.voice.tts.voice", "zh-CN-XiaoxiaoNeural")))
        self.record_audio = bool(config.get("security.record_audio", True))
        self.audio_dir = config.path_of("security.audio_dir", "data/audio")
        self.latency_budget = int(config.get("channel.voice.latency_budget_ms", 2500))
        self.metrics = VoiceMetrics()

    async def handle_inbound(self, msg: InboundMessage) -> OutboundReply:
        """文本入站 → 返回文本 + 语音。这是降级与测试路径。"""
        session = self._ensure_session(msg)
        t0 = time.time()
        reply = await self.engine.on_scammer_message(session, msg.text)
        latency = (time.time() - t0) * 1000
        self._record_latency(latency)

        audio = b""
        if reply and self.tts is not None:
            try:
                audio = await self.tts.synthesize(
                    reply, voice=self.tts.voice_for_accent(session.persona.accent))
                self.metrics.tts_calls += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("TTS 合成失败: %s", exc)
        return OutboundReply(peer=msg.peer, text=reply,
                             session_id=session.session_id, ts=time.time(),
                             audio=audio or None)

    def _record_latency(self, latency_ms: float) -> None:
        self.metrics.turns += 1
        self.metrics.total_latency_ms += latency_ms
        self.metrics.max_latency_ms = max(self.metrics.max_latency_ms, latency_ms)
        if latency_ms > self.latency_budget:
            log.warning("端到端延迟 %.0fms 超出预算 %dms",
                        latency_ms, self.latency_budget)

    async def stream_voice_turn(self, session, audio_stream: AsyncIterator[bytes],
                                on_sentence: Any = None) -> str:
        """流式语音回合。

        audio_stream: 对方的音频分片
        on_sentence: 每合成一句调用一次，用于边生成边播放
        """
        # 1) ASR 转写对方说话
        transcript = await self.asr.transcribe(audio_stream)
        if not transcript.strip():
            return ""

        self.metrics.vad_hits += 1
        t0 = time.time()

        # 2) 流式 LLM，边生成边按句合成
        buf: list[str] = []
        pending = ""
        async for chunk in self.engine.stream_reply(session, transcript):
            buf.append(chunk)
            pending += chunk
            # 遇到句末标点就提前合成，压缩端到端延迟
            if re.search(r"[。！？!?]", pending):
                sentence = pending.strip()
                pending = ""
                if sentence and on_sentence:
                    voice = self.tts.voice_for_accent(session.persona.accent)
                    audio = await self.tts.synthesize(sentence, voice=voice)
                    self.metrics.tts_calls += 1
                    await on_sentence(sentence, audio)

        # 3) 收尾：合成剩余部分
        tail = pending.strip()
        if tail and on_sentence:
            voice = self.tts.voice_for_accent(session.persona.accent)
            audio = await self.tts.synthesize(tail, voice=voice)
            self.metrics.tts_calls += 1
            await on_sentence(tail, audio)

        self._record_latency((time.time() - t0) * 1000)
        return "".join(buf).strip()

    async def handle_interrupt(self) -> None:
        """对方插话处理。真实实现需要立刻停止播放缓冲。"""
        self.metrics.interrupts += 1
        log.info("检测到打断，停止播放")

    def save_audio(self, session_id: str, audio: bytes) -> str | None:
        if not self.record_audio or not audio:
            return None
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        path = self.audio_dir / f"{session_id}-{int(time.time())}.wav"
        try:
            path.write_bytes(audio)
            return str(path)
        except OSError as exc:
            log.warning("音频落盘失败: %s", exc)
            return None

    def metrics_dict(self) -> dict[str, Any]:
        return {
            "turns": self.metrics.turns,
            "avg_latency_ms": round(self.metrics.avg_latency_ms, 1),
            "max_latency_ms": round(self.metrics.max_latency_ms, 1),
            "tts_calls": self.metrics.tts_calls,
            "interrupts": self.metrics.interrupts,
            "latency_budget_ms": self.latency_budget,
            "components": {
                "vad": type(self.vad).__name__,
                "asr": type(self.asr).__name__,
                "tts": type(self.tts).__name__,
            },
        }
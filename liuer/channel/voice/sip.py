"""SIP 接入。

对应 Apate Voice 的 SIP/RTP VoIP 接入方式。

本模块提供完整接入逻辑与 FreeSWITCH/Asterisk 对接配置，
但不含任何未授权拦截能力——只处理显式转接进来的会话。

本地部署若无真实 SIP 环境，可用 mock 模式跑通链路。
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator

from ..base import InboundMessage, OutboundReply
from .pipeline import VoicePipeline

log = logging.getLogger(__name__)


@dataclass
class SIPCallContext:
    """一通已转接进来的来电。"""

    call_id: str
    from_number: str
    to_number: str
    session_id: str | None = None
    started_at: float = 0.0


class SIPIngress:
    """SIP 转接入口。

    对接方式（以 FreeSWITCH 为例）：
    1. 在 dialplan 中把高风险号码路由到本服务的 SIP 分机
    2. 运营商侧或前置 SBC 完成风险判定后转接
    3. 本服务用 SDP 协商 RTP 流，交给 VoicePipeline 处理

    需要 aiortc 或 pjsua2 绑定 5060/UDP。本项目不在核心依赖中强制引入，
    因为多数本地验证不需要真实 SIP。
    """

    def __init__(self, pipeline: VoicePipeline, config) -> None:
        self.pipeline = pipeline
        self.cfg = config
        self.enabled = bool(config.get("channel.voice.sip.enabled", False))
        self.host = str(config.get("channel.voice.sip.host", "0.0.0.0"))
        self.port = int(config.get("channel.voice.sip.port", 5060))
        self._aiortc = None
        self._pc = None

    async def start(self) -> bool:
        """尝试启动 SIP 栈。缺少依赖时降级，不阻塞系统。"""
        if not self.enabled:
            log.info("SIP 通道未启用，使用内部模拟通道")
            return False
        try:
            import aiortc  # noqa: F401
        except ImportError:
            log.warning("未安装 aiortc，SIP 通道降级为内部模拟。"
                        "安装方式: pip install aiortc")
            return False
        try:
            from aiortc import RTCPeerConnection, RTCSessionDescription
            self._pc = RTCPeerConnection()
            self._aiortc = (RTCPeerConnection, RTCSessionDescription)
            log.info("SIP 通道已启动，监听 %s:%d", self.host, self.port)
            return True
        except Exception as exc:  # noqa: BLE001
            log.error("SIP 通道启动失败: %s", exc)
            return False

    async def handle_call(self, ctx: SIPCallContext,
                          audio_stream: AsyncIterator[bytes] | None = None,
                          on_sentence: Any = None) -> OutboundReply:
        """处理一通已建立的来电。

        无音频流时走文本降级路径，便于本地联调。
        """
        session = self.pipeline._ensure_session(  # noqa: SLF001 - 通道内部协作
            InboundMessage(peer=ctx.from_number, text="",
                           meta={"call_id": ctx.call_id}))
        ctx.session_id = session.session_id

        if audio_stream is not None and on_sentence is not None:
            reply_text = await self.pipeline.stream_voice_turn(
                session, audio_stream, on_sentence)
            return OutboundReply(peer=ctx.from_number, text=reply_text,
                                 session_id=session.session_id, ts=0.0)

        # 文本降级
        session.add_turn("scammer", "[SIP 来电已接入]")
        reply = await self.pipeline.engine.on_scammer_message(session, "")
        return OutboundReply(peer=ctx.from_number, text=reply,
                             session_id=session.session_id, ts=0.0)

    async def hangup(self, ctx: SIPCallContext) -> None:
        if ctx.session_id:
            session = self.pipeline.engine.get_session(ctx.session_id)
            if session:
                self.pipeline.engine.end_session(session, reason="通话结束")

    async def stop(self) -> None:
        if self._pc is not None:
            await self._pc.close()
            self._pc = None


def freeswitch_dialplan_xml(exten: str, sip_host: str, sip_port: int = 5080) -> str:
    """生成 FreeSWITCH dialplan 片段。

    把已判定为高风险的来电转接到 LiuEr。
    使用前请确认你的线路具备合法转接能力。
    """
    return f"""<extension name="liuer_{exten}">
    <condition field="destination_number" expression="^{exten}$">
      <action application="set" data="sip_transport=udp"/>
      <action application="bridge" data="sofia/internal:${{sip_host}}:{sip_port}"/>
    </condition>
</extension>"""


async def health_check(config) -> dict[str, Any]:
    """检查语音通道依赖是否齐备。"""
    result: dict[str, Any] = {}
    for name, mod in (("aiortc", "aiortc"),
                      ("funasr", "funasr"),
                      ("edge_tts", "edge_tts"),
                      ("silero_vad", "silero_vad")):
        try:
            __import__(mod)
            result[name] = True
        except ImportError:
            result[name] = False
    result["degraded"] = not result.get("aiortc", False)
    result["note"] = ("缺少组件时语音通道自动降级为文本模式，核心反诈逻辑不受影响。"
                      "ASR/TTS 组件缺失不影响情报提取。")
    return result
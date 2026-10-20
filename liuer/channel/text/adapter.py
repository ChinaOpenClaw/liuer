"""文本通道。

三种模式：
- simulated：完全本地，供开发测试。随机生成诈骗话术，验证整条链路。
- web：浏览器控制台对聊，供人工注入真实样本。
- whatsapp / telegram：对外适配器，需自行接入并自行确保平台合规。

重要：外部平台适配器只提供接入骨架，不含任何绕过平台 ToS 的手段。
接入前需自行评估当地法律与平台政策。
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from typing import Any

from ..base import Channel, InboundMessage, OutboundReply
from ...persona.strategy import Intent, Mission

log = logging.getLogger(__name__)

# 模拟器用的诈骗话术脚本，用于本地验证链路
SIM_SCRIPTS = {
    "authority_impersonation": [
        "您好，这里是市公安局反诈中心的，您的身份证涉嫌一起洗钱案件。",
        "我是XX市监督管理局，现在有人用您的名义开了公司，需要您配合调查。",
        "您的社保卡被冻结了，请配合我走一下资金清查流程。",
    ],
    "tech_support": [
        "您好，您的账户出现异常交易，需要您现在打开手机配合验证。",
        "您的订单存在问题退款，需要您打开这个链接填写银行卡信息。",
        "您的银行账户被冻结，需要下载一个软件远程配合解冻。",
    ],
    "investment": [
        "我们这边有个内部渠道，年化收益能到30%，要不要了解一下？",
        "跟着老师操作，保证收益，投多少都有分成。",
        "USDT 投资群，稳赚不赔，先加个微信详聊。",
    ],
    "job_scam": [
        "我们招兼职，日结佣金很高，只需要做几个任务就行。",
        "有个居家赚钱的项目，佣金55%，您有兴趣吗？",
    ],
    "family_emergency": [
        "妈，我手机摔了，这是同学号码，我出点事急需用钱。",
        "爸，我出事了，现在不能说太多，先给我转点钱应急。",
    ],
}


class SimulatedTextChannel(Channel):
    """本地模拟通道。随机扮演诈骗方，验证完整链路。"""

    name = "text_simulated"

    async def handle_inbound(self, msg: InboundMessage) -> OutboundReply:
        session = self._ensure_session(msg)
        reply = await self.engine.on_scammer_message(session, msg.text)
        return OutboundReply(peer=msg.peer, text=reply,
                             session_id=session.session_id, ts=time.time())

    def generate_next(self, session: Session) -> str:
        """生成下一句诈骗话术，用于模拟器推进。"""
        tactics = session.intel_summary.get("tactics") or ["tech_support"]
        tactic = random.choice(tactics) if tactics else "tech_support"
        pool = SIM_SCRIPTS.get(tactic, SIM_SCRIPTS["tech_support"])
        return random.choice(pool)


class WebTextChannel(Channel):
    """浏览器控制台通道。人工注入样本，观察 agent 应对。"""

    name = "text_web"

    async def handle_inbound(self, msg: InboundMessage) -> OutboundReply:
        mission = None
        if (msg.meta or {}).get("mission"):
            m = msg.meta["mission"]
            mission = Mission(mission_id=str(uuid.uuid4().hex[:8]),
                              intent=Intent(m.get("intent", "none")),
                              target=m.get("target", ""),
                              notes=m.get("notes", ""),
                              created_at=time.time())
        session = self._ensure_session(msg, mission)
        reply = await self.engine.on_scammer_message(session, msg.text)
        return OutboundReply(peer=msg.peer, text=reply,
                             session_id=session.session_id, ts=time.time())


class ExternalTextChannel(Channel):
    """WhatsApp / Telegram 适配器骨架。

    仅提供接入契约。要真正连通需要自行实现 platform_client，
    并确保符合平台条款与当地法律。本项目不提供任何规避检测的手段。
    """

    name = "text_external"

    def __init__(self, engine, config, adapter: str = "whatsapp") -> None:
        super().__init__(engine, config)
        self.adapter = adapter
        self.platform_client: Any = None   # 由使用者注入

    async def connect(self, platform_client: Any) -> None:
        self.platform_client = platform_client

    async def handle_inbound(self, msg: InboundMessage) -> OutboundReply:
        session = self._ensure_session(msg)
        reply = await self.engine.on_scammer_message(session, msg.text)
        return OutboundReply(peer=msg.peer, text=reply,
                             session_id=session.session_id, ts=time.time())

    async def send(self, peer: str, text: str) -> None:
        """向外部平台发送消息。需先注入 platform_client。"""
        if self.platform_client is None:
            raise RuntimeError(
                f"{self.adapter} 通道未连接。请先实现并注入 platform_client，"
                "并自行确保接入方式符合平台条款与当地法律。")
        await self.platform_client.send_text(peer, text)


class HoneypotPool:
    """蜜罐号码池。

    作用是被动吸引群发短信与自动拨号系统。
    本地部署下主要用于测试与研究，生产环境使用需遵守当地电信法规。
    """

    def __init__(self, config) -> None:
        size = int(config.get("channel.text.honeypot_pool_size", 500))
        self._numbers = [f"1{prefix}{i:08d}"
                         for prefix in ("3", "5", "7", "8", "9")
                         for i in random.sample(range(10**8), size // 5 + 1)][:size]
        self._enabled = bool(config.get("channel.text.honeypot_enabled", False))

    @property
    def enabled(self) -> bool:
        return self._enabled

    def numbers(self, limit: int = 10) -> list[str]:
        if not self._enabled:
            return []
        return random.sample(self._numbers, min(limit, len(self._numbers)))

    def size(self) -> int:
        return len(self._numbers)
"""通道抽象。

不同通道的差异只在「怎么收一句话」和「怎么把回应送出去」，
中间的人格、策略、情报逻辑完全共用。
"""
from __future__ import annotations

import abc
import uuid
from dataclasses import dataclass
from typing import Any

from ..core.engine import LiuErEngine, Session
from ..persona.strategy import Mission


@dataclass
class InboundMessage:
    peer: str                 # 对端标识：号码、账号 ID
    text: str
    session_id: str | None = None
    meta: dict[str, Any] | None = None


@dataclass
class OutboundReply:
    peer: str
    text: str
    session_id: str
    ts: float
    audio: bytes | None = None   # 语音通道回填音频


class Channel(abc.ABC):
    """通道基类。"""

    name: str = "base"

    def __init__(self, engine: LiuErEngine, config) -> None:
        self.engine = engine
        self.cfg = config

    @abc.abstractmethod
    async def handle_inbound(self, msg: InboundMessage) -> OutboundReply:
        """处理一条入站消息，返回出站回复。"""

    def _ensure_session(self, msg: InboundMessage,
                        mission: Mission | None = None) -> Session:
        """复用未结束的会话，避免每条消息都换人格。"""
        if msg.session_id:
            existing = self.engine.get_session(msg.session_id)
            if existing and existing.status == "active":
                return existing
        return self.engine.start_session(channel=self.name, peer=msg.peer,
                                        mission=mission)

    async def close(self) -> None:
        return None
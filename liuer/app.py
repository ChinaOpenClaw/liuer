"""应用装配与生命周期。

所有组件在此初始化并注入，保证单测与生产用同一套装配路径。
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from .channel.text import HoneypotPool, SimulatedTextChannel, WebTextChannel
from .channel.voice import VoicePipeline, health_check
from .config import Config, Keystore, RuntimePaths, load_config
from .core import LiuErEngine
from .intel import IntelPipeline, IntelStore
from .llm import OllamaClient
from .persona import PersonaLibrary, PersonaProfile, load_profile

log = logging.getLogger(__name__)


@dataclass
class App:
    cfg: Config
    llm: OllamaClient
    store: IntelStore
    pipeline: IntelPipeline
    library: PersonaLibrary
    engine: LiuErEngine
    web_channel: WebTextChannel
    sim_channel: SimulatedTextChannel
    voice_channel: VoicePipeline
    honeypot: HoneypotPool
    profile: PersonaProfile | None = None

    async def close(self) -> None:
        await self.llm.__aexit__(None, None, None)


def build_app(config_path: str | None = None,
              profile_path: str | None = None) -> App:
    cfg = load_config(config_path)
    paths = RuntimePaths(cfg.root, raw=cfg.raw)

    # 档案优先于配置项：命令行显式指定时覆盖 config.yaml
    profile_path = profile_path or cfg.get("persona.profile")
    profile = load_profile(profile_path) if profile_path else None

    keystore = Keystore.load_or_create(
        paths.ensure("security.keystore", "data/.keystore"),
        str(cfg.get("security.encryption_key", "")))

    store = IntelStore(paths.ensure("intel.database", "data/liuer.db"), keystore.key)
    llm = OllamaClient(cfg)
    pipeline = IntelPipeline(cfg, store, llm)
    library = PersonaLibrary(cfg)
    engine = LiuErEngine(cfg, llm, store, pipeline, library, profile)

    web_channel = WebTextChannel(engine, cfg)
    sim_channel = SimulatedTextChannel(engine, cfg)
    from .channel.voice.tts_hf import build_tts
    voice_channel = VoicePipeline(engine, cfg, tts=build_tts(cfg))
    honeypot = HoneypotPool(cfg)

    return App(cfg=cfg, llm=llm, store=store, pipeline=pipeline, library=library,
               engine=engine, web_channel=web_channel, sim_channel=sim_channel,
               voice_channel=voice_channel, honeypot=honeypot, profile=profile)


async def startup(app: App) -> None:
    await app.llm.start()
    health = await app.llm.health()
    if not health.get("ok"):
        log.warning("Ollama 未就绪: %s", health.get("error"))
        log.warning("系统仍会启动，但对话生成不可用。情报规则层不受影响。")
    else:
        log.info("Ollama 就绪 | 对话模型=%s 抽取模型=%s",
                 app.llm.chat_model, app.llm.extract_model)
    vh = await health_check(app.cfg)
    if vh.get("degraded"):
        log.info("语音组件降级: %s", vh.get("note"))
    log.info("LiuEr 启动完成 | 部署模式=%s | 离线=%s",
             app.cfg.get("deployment.mode"), app.cfg.get("deployment.offline"))
"""本地 Ollama 客户端。

设计要点：
1. 默认纯离线。所有请求指向 127.0.0.1，不出网。
2. 对话与抽取用不同模型，避免大模型拖累抽取延迟。
3. 抽取走独立的短超时 + 规则兜底，规则失败不阻塞会话。
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import AsyncIterator, Any

import httpx

log = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, config) -> None:
        self.base_url = str(config.get("llm.base_url", "http://127.0.0.1:11434")).rstrip("/")
        self.chat_model = str(config.get("llm.chat_model", "qwen2.5:14b-instruct"))
        self.extract_model = str(config.get("llm.extract_model", "qwen3.5:9b"))
        # num_ctx 直接决定 KV 缓存占用的显存，是 8GB 卡能否全放显存的关键杠杆
        self.num_ctx = int(config.get("llm.num_ctx", 4096))
        # -1 表示常驻不卸载，避免每次请求重新加载
        self.keep_alive = int(config.get("llm.keep_alive", -1))
        self.temperature = float(config.get("llm.temperature", 0.85))
        self.top_p = float(config.get("llm.top_p", 0.9))
        self.timeout = float(config.get("llm.request_timeout", 60))
        # Qwen3.5 默认开启思考模式。实测 think=True 时 80 token 的预算会被思考过程
        # 吃光导致正文为空，且延迟翻倍。人格对话必须关掉。
        self.think = bool(config.get("llm.think", False))
        self.offline = bool(config.get("deployment.offline", True))
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "OllamaClient":
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout, connect=5.0),
        )
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("OllamaClient 未初始化，请使用 async with 或调用 start()")
        return self._client

    async def start(self) -> "OllamaClient":
        await self.__aenter__()
        return self

    @staticmethod
    def _clean(text: str) -> str:
        """剥掉可能混入正文的思考标签与首尾空白。

        即便 think=False，某些模型或模板仍可能吐出 <think> 块，
        人格对话里这等于直接暴露机器身份，必须兜底清除。
        """
        if not text:
            return ""
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        cleaned = re.sub(r"</?think>", "", cleaned)
        return cleaned.strip()

    async def health(self) -> dict[str, Any]:
        """探测服务与模型可用性。"""
        try:
            resp = await self.client.get("/api/tags", timeout=5.0)
            resp.raise_for_status()
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            return {
                "ok": True,
                "base_url": self.base_url,
                "available_models": models,
                "chat_model_ready": any(self.chat_model.split(":")[0] in m for m in models),
                "extract_model_ready": any(self.extract_model.split(":")[0] in m for m in models),
            }
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "base_url": self.base_url, "error": str(exc)}

    async def chat(self, messages: list[dict[str, str]], *,
                   model: str | None = None,
                   temperature: float | None = None,
                   max_tokens: int = 256,
                   stop: list[str] | None = None) -> str:
        """单轮非流式对话。"""
        payload = {
            "model": model or self.chat_model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "think": self.think,
            "options": {
                "temperature": self.temperature if temperature is None else temperature,
                "top_p": self.top_p,
                "num_predict": max_tokens,
                "num_ctx": self.num_ctx,
                **({"stop": stop} if stop else {}),
            },
        }
        resp = await self.client.post("/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return self._clean((data.get("message") or {}).get("content", ""))

    async def chat_stream(self, messages: list[dict[str, str]], *,
                          model: str | None = None,
                          temperature: float | None = None,
                          max_tokens: int = 128) -> AsyncIterator[str]:
        """流式对话。语音通道用它实现句级 TTS 触发。"""
        payload = {
            "model": model or self.chat_model,
            "messages": messages,
            "stream": True,
            "keep_alive": self.keep_alive,
            "think": self.think,
            "options": {
                "temperature": self.temperature if temperature is None else temperature,
                "top_p": self.top_p,
                "num_predict": max_tokens,
                "num_ctx": self.num_ctx,
            },
        }
        async with self.client.stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if data.get("done"):
                    break
                chunk = (data.get("message") or {}).get("content", "")
                if chunk:
                    yield chunk

    async def complete(self, prompt: str, *, model: str | None = None,
                       temperature: float = 0.1, max_tokens: int = 512) -> str:
        """补全接口，用于结构化抽取。"""
        payload = {
            "model": model or self.extract_model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
            "think": self.think,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": self.num_ctx,
            },
        }
        resp = await self.client.post("/api/generate", json=payload)
        resp.raise_for_status()
        return self._clean(resp.json().get("response", ""))

    async def pull_model(self, model: str) -> bool:
        """预拉取模型，便于离线部署前准备。"""
        log.info("拉取模型 %s", model)
        resp = await self.client.post("/api/pull", json={"name": model, "stream": False},
                                      timeout=httpx.Timeout(1800.0))
        return resp.status_code == 200
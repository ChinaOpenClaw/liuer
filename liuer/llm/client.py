"""本地 Ollama 客户端。

设计要点：
1. 默认纯离线。所有请求指向 127.0.0.1，不出网。
2. 对话与抽取用不同模型，避免大模型拖累抽取延迟。
3. 抽取走独立的短超时 + 规则兜底，规则失败不阻塞会话。
4. 短期思考（deliberation）显式可控，且绝不能污染正文。

关于「短期思考」的实现选择（实测结论，勿轻易改回原生思考）：
    qwen3.5:4b 在 Ollama 0.32 下开启 think=true 时，思考链会一路展开不收敛——
    num_predict=416 时 416 个 token 全部用于思考、正文为空；
    加到 1024 时思考仍达 3.9K 字符、正文依旧为空，单次耗时 11 秒。
    实时对话既要低延迟又要正文非空，所以默认走 brief 双阶段：
      阶段一 用极小预算（默认 160 token）+ 低温度，让模型写一小段中文内心活动；
      阶段二 把这段内心活动作为「不可外泄的私有上下文」注入系统提示，再生成正文。
    好处是思考长度、语言、结构都可控；思考失败或超时自动降级为直出，绝不阻塞。
    仍保留 native 模式（原生 thinking 字段）供支持该特性的模型切换，并带空正文重试兜底。
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx

log = logging.getLogger(__name__)

# 思考注入模板。强调「不可外泄」，防止模型把思考内容直接说给对方。
THINK_INJECTION = (
    "\n\n[内心活动｜仅你自己可见，绝对不能对对方说出口，也不要复述其中的任何措辞]\n"
    "{thinking}\n"
    "[内心活动结束。现在按上述判断，用你自己的口气说出该说的话]"
)

# 阶段一的思考指令。要求中文、简短、三条以内，避免思考膨胀。
THINK_INSTRUCTION = (
    "（请先不要回复对方。用中文写下不超过 3 条的简短内心活动，每条一行，总共不超过 80 字：\n"
    "1) 对方这句话想让我做什么；\n"
    "2) 我这轮最该达到什么目的；\n"
    "3) 我该怎么说（语气、口癖、要暴露的小破绽）。\n"
    "只写内心活动本身，不要写开场白，不要写你要说的话，不要写标题。）"
)


@dataclass
class ThinkResult:
    """一次带思考的生成结果。thinking 仅用于审计与看板展示，不参与发送。"""
    text: str = ""
    thinking: str = ""
    mode: str = "off"          # brief | native | off
    degraded: bool = False     # True 表示思考未生效（超时/失败/降级重试）
    think_ms: float = 0.0
    reply_ms: float = 0.0
    note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def used(self) -> bool:
        return bool(self.thinking) and not self.degraded


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

        # ---- 短期思考 ----
        # think_mode: brief（双阶段，默认）| native（原生 thinking 字段）| off（关闭）
        self.think_mode = str(config.get("llm.think_mode", "")).strip().lower()
        if self.think_mode not in ("brief", "native", "off"):
            # 兼容旧配置：llm.think=false 视为关闭，否则默认 brief
            self.think_mode = "brief" if config.get("llm.think", True) else "off"
        # 阶段一思考预算（token）。brief 模式下这是硬上限，思考不会无限展开。
        self.think_budget = int(config.get("llm.think_budget", 160))
        # native 模式下额外追加给思考的 token（思考 + 正文共享 num_predict）
        self.native_think_budget = int(config.get("llm.native_think_budget", 768))
        # 思考用低温，保证稳定与收敛
        self.think_temperature = float(config.get("llm.think_temperature", 0.35))
        # 思考阶段独立超时。超时即降级直出，不拖累正文。
        self.think_timeout = float(config.get("llm.think_timeout", 8))
        # 思考文本最长保留字符数（超出截断，防止看板爆炸）
        self.think_max_chars = int(config.get("llm.think_max_chars", 400))

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
        cleaned = re.sub(r"<thinking>.*?</thinking>", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"</?think(ing)?>", "", cleaned)
        return cleaned.strip()

    @staticmethod
    def _clean_thinking(text: str, max_chars: int) -> str:
        """清洗思考文本：去标签、去开场白套话、限长。"""
        if not text:
            return ""
        t = re.sub(r"<think(ing)?>.*?</think(ing)?>", "", text, flags=re.DOTALL)
        t = re.sub(r"</?think(ing)?>", "", t)
        t = re.sub(r"^\s*(思考(过程)?|内心活动|Thinking Process)\s*[:：]\s*", "", t.strip())
        # 去掉模型常见的应答套话前缀
        t = re.sub(r"^\s*(好的|好|明白|收到)[,，。.！!]?\s*", "", t)
        t = re.sub(r"[*_`#>]+", "", t)
        t = t.strip()
        if len(t) > max_chars:
            t = t[:max_chars].rstrip() + "…"
        return t

    @staticmethod
    def _inject_thinking(messages: list[dict[str, str]], thinking: str) -> list[dict[str, str]]:
        """把思考作为私有上下文注入 system 提示，不进入对话历史。"""
        if not thinking:
            return messages
        out = [dict(m) for m in messages]
        for i, m in enumerate(out):
            if m.get("role") == "system":
                out[i] = {"role": "system",
                          "content": m.get("content", "") + THINK_INJECTION.format(thinking=thinking)}
                return out
        # 没有 system 消息时插到最前面
        out.insert(0, {"role": "system",
                       "content": THINK_INJECTION.format(thinking=thinking).strip()})
        return out

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
                "think_mode": self.think_mode,
            }
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "base_url": self.base_url, "error": str(exc),
                    "think_mode": self.think_mode}

    # ---------- 原始调用 ----------

    def _payload(self, messages: list[dict[str, str]], *, model: str | None,
                 temperature: float, num_predict: int, think: bool,
                 stop: list[str] | None = None) -> dict[str, Any]:
        return {
            "model": model or self.chat_model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "think": think,
            "options": {
                "temperature": temperature,
                "top_p": self.top_p,
                "num_predict": num_predict,
                "num_ctx": self.num_ctx,
                **({"stop": stop} if stop else {}),
            },
        }

    async def _raw_chat(self, messages: list[dict[str, str]], *, model: str | None = None,
                        temperature: float, num_predict: int, think: bool,
                        stop: list[str] | None = None) -> tuple[str, str]:
        """返回 (content, thinking)。原生思考在 thinking 字段，已与正文分离。"""
        payload = self._payload(messages, model=model, temperature=temperature,
                                num_predict=num_predict, think=think, stop=stop)
        resp = await self.client.post("/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        msg = data.get("message") or {}
        return self._clean(msg.get("content", "")), (msg.get("thinking") or "").strip()

    # ---------- 短期思考 ----------

    async def _brief_think(self, messages: list[dict[str, str]],
                           model: str | None) -> str:
        """阶段一：极小预算的中文内心活动。失败即返回空串，绝不抛给上层。"""
        think_msgs = [dict(m) for m in messages]
        think_msgs.append({"role": "user", "content": THINK_INSTRUCTION})
        try:
            content, native = await asyncio.wait_for(
                self._raw_chat(think_msgs, model=model,
                               temperature=self.think_temperature,
                               num_predict=self.think_budget, think=False),
                timeout=self.think_timeout,
            )
        except asyncio.TimeoutError:
            log.warning("短期思考超时（%.1fs），降级为直出", self.think_timeout)
            return ""
        except Exception as exc:  # noqa: BLE001
            log.warning("短期思考失败，降级为直出: %s", exc)
            return ""
        raw = content or native
        return self._clean_thinking(raw, self.think_max_chars)

    # ---------- 对外主入口 ----------

    async def chat_detailed(self, messages: list[dict[str, str]], *,
                            model: str | None = None,
                            temperature: float | None = None,
                            max_tokens: int = 256,
                            stop: list[str] | None = None,
                            think_mode: str | None = None) -> ThinkResult:
        """带短期思考的对话。thinking 与正文分离返回，正文可直接发送。"""
        mode = (think_mode or self.think_mode or "off").lower()
        temp = self.temperature if temperature is None else temperature
        res = ThinkResult(mode=mode)

        if mode == "off":
            t0 = time.perf_counter()
            text, _ = await self._raw_chat(messages, model=model, temperature=temp,
                                           num_predict=max_tokens, think=False, stop=stop)
            res.text, res.reply_ms = self._clean(text), (time.perf_counter() - t0) * 1000
            return res

        if mode == "native":
            return await self._chat_native(messages, model=model, temperature=temp,
                                           max_tokens=max_tokens, stop=stop)

        return await self._chat_brief(messages, model=model, temperature=temp,
                                      max_tokens=max_tokens, stop=stop)

    async def _chat_brief(self, messages: list[dict[str, str]], *, model, temperature,
                          max_tokens: int, stop) -> ThinkResult:
        res = ThinkResult(mode="brief")
        t0 = time.perf_counter()
        res.thinking = await self._brief_think(messages, model)
        res.think_ms = (time.perf_counter() - t0) * 1000
        if not res.thinking:
            res.degraded, res.note = True, "思考为空或超时，已直出"

        t1 = time.perf_counter()
        target = self._inject_thinking(messages, res.thinking) if res.thinking else messages
        text, _ = await self._raw_chat(target, model=model, temperature=temperature,
                                       num_predict=max_tokens, think=False, stop=stop)
        res.reply_ms = (time.perf_counter() - t1) * 1000
        res.text = self._clean(text)

        # 正文为空的兜底：拿掉思考注入，用原始上下文重生成一次
        if not res.text:
            log.warning("brief 模式正文为空，去掉思考注入重试")
            text, _ = await self._raw_chat(messages, model=model, temperature=temperature,
                                           num_predict=max_tokens, think=False, stop=stop)
            res.text = self._clean(text)
            res.degraded, res.note = True, (res.note + "；正文为空已重试").strip("；")
        return res

    async def _chat_native(self, messages: list[dict[str, str]], *, model, temperature,
                           max_tokens: int, stop) -> ThinkResult:
        """原生思考字段模式。思考与正文共享 num_predict，故预算需叠加。"""
        res = ThinkResult(mode="native")
        budget = max_tokens + self.native_think_budget
        t0 = time.perf_counter()
        text, thinking = await self._raw_chat(
            messages, model=model, temperature=temperature,
            num_predict=budget, think=True, stop=stop)
        res.reply_ms = (time.perf_counter() - t0) * 1000
        res.think_ms = res.reply_ms
        res.thinking = self._clean_thinking(thinking, self.think_max_chars)
        res.text = self._clean(text)

        if not res.text:
            # 原生思考吃光预算是 qwen3.5:4b 的已知问题，直接降级直出重试
            log.warning("native 模式正文为空（思考吃光预算），降级为直出重试")
            t1 = time.perf_counter()
            text, _ = await self._raw_chat(messages, model=model, temperature=temperature,
                                           num_predict=max_tokens, think=False, stop=stop)
            res.text = self._clean(text)
            res.reply_ms = (time.perf_counter() - t1) * 1000
            res.degraded, res.note = True, "原生思考耗尽预算，已降级直出"
        return res

    async def chat(self, messages: list[dict[str, str]], *,
                   model: str | None = None,
                   temperature: float | None = None,
                   max_tokens: int = 256,
                   stop: list[str] | None = None) -> str:
        """单轮非流式对话。只要正文，思考过程通过 chat_detailed 获取。"""
        res = await self.chat_detailed(messages, model=model, temperature=temperature,
                                       max_tokens=max_tokens, stop=stop)
        return res.text

    # ---------- 流式 ----------

    async def chat_stream_events(self, messages: list[dict[str, str]], *,
                                 model: str | None = None,
                                 temperature: float | None = None,
                                 max_tokens: int = 128,
                                 think_mode: str | None = None
                                 ) -> AsyncIterator[dict[str, Any]]:
        """事件流：{"type": "thinking"|"content"|"done", ...}。

        语音通道可以只消费 content 做句级 TTS；看板可先展示 thinking。
        """
        mode = (think_mode or self.think_mode or "off").lower()
        temp = self.temperature if temperature is None else temperature
        thinking = ""

        if mode == "brief":
            t0 = time.perf_counter()
            thinking = await self._brief_think(messages, model)
            if thinking:
                yield {"type": "thinking", "text": thinking,
                       "ms": (time.perf_counter() - t0) * 1000}
            target = self._inject_thinking(messages, thinking) if thinking else messages
            use_think = False
            num_predict = max_tokens
        elif mode == "native":
            target, use_think = messages, True
            num_predict = max_tokens + self.native_think_budget
        else:
            target, use_think = messages, False
            num_predict = max_tokens

        payload = self._payload(target, model=model, temperature=temp,
                                num_predict=num_predict, think=use_think)
        buf: list[str] = []
        async with self.client.stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = data.get("message") or {}
                # 原生思考走 thinking 字段，绝不能混入要念出来的正文
                th = msg.get("thinking") or ""
                if th and mode == "native":
                    thinking += th
                    yield {"type": "thinking_delta", "text": th}
                    continue
                chunk = msg.get("content", "")
                if chunk:
                    buf.append(chunk)
                    yield {"type": "content", "text": chunk}
                if data.get("done"):
                    break

        full = self._clean("".join(buf))
        degraded = False
        if not full:
            # 兜底重跑一次非流式直出
            text, _ = await self._raw_chat(messages, model=model, temperature=temp,
                                           num_predict=max_tokens, think=False)
            full = self._clean(text)
            degraded = True
        yield {"type": "done", "text": full, "thinking": thinking, "degraded": degraded}

    async def chat_stream(self, messages: list[dict[str, str]], *,
                          model: str | None = None,
                          temperature: float | None = None,
                          max_tokens: int = 128) -> AsyncIterator[str]:
        """流式对话。只吐正文，语音通道用它实现句级 TTS 触发。"""
        async for ev in self.chat_stream_events(
                messages, model=model, temperature=temperature, max_tokens=max_tokens):
            if ev.get("type") == "content":
                yield ev["text"]
            elif ev.get("type") == "done" and ev.get("degraded") and ev.get("text"):
                yield ev["text"]

    async def complete(self, prompt: str, *, model: str | None = None,
                       temperature: float = 0.1, max_tokens: int = 512) -> str:
        """补全接口，用于结构化抽取。

        抽取要的是稳定 JSON，不需要思考，且思考会拖慢实时管道，故恒定关闭。
        """
        payload = {
            "model": model or self.extract_model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
            "think": False,
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

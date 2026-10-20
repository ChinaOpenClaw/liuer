"""短期思考（deliberation）的离线测试。

不依赖 Ollama：用假的 _raw_chat 替掉真实 HTTP 调用，
验证「思考与正文分离、思考失败降级、正文为空重试」等关键路径。

背景：qwen3.5:4b 原生 think=true 思考链不收敛，实测 1024 预算下正文仍为空，
因此默认改为 brief 双阶段。这里的用例正是防止有人改回后重新踩坑。
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from liuer.config import load_config  # noqa: E402
from liuer.core.engine import LiuErEngine  # noqa: E402
from liuer.intel import IntelPipeline, IntelStore  # noqa: E402
from liuer.llm.client import OllamaClient, ThinkResult  # noqa: E402
from liuer.persona import PersonaLibrary, Mission  # noqa: E402

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


MSGS = [
    {"role": "system", "content": "你是沈静宜，1966 年生，退休中学教师。"},
    {"role": "user", "content": "阿姨，我是公安局的，您名下银行卡涉嫌洗钱。"},
]


class FakeRaw:
    """替换 OllamaClient._raw_chat，按调用类型返回预置结果。

    思考调用的标志：最后一条 user 消息包含「内心活动」指令。
    """
    def __init__(self, *, thinking="1) 对方想吓我\n2) 我该拖住他\n3) 装听不清",
                 reply="啊？公安局？我这信号不太好，你再说一遍。",
                 think_raises=None, empty_reply=False):
        self.thinking = thinking
        self.reply = reply
        self.think_raises = think_raises
        self.empty_reply = empty_reply
        self.calls: list[dict] = []

    async def __call__(self, messages, *, model=None, temperature=0.0,
                       num_predict=0, think=False, stop=None):
        last = messages[-1].get("content", "") if messages else ""
        is_think = "内心活动" in last
        self.calls.append({"is_think": is_think, "think": think,
                           "num_predict": num_predict, "n_msgs": len(messages)})
        if is_think and self.think_raises:
            raise self.think_raises
        if is_think:
            return self.thinking, ""
        if self.empty_reply:
            return "", ""
        return self.reply, ""


def make_client(mode: str) -> OllamaClient:
    cfg = load_config()
    c = OllamaClient(cfg)
    c.think_mode = mode
    c._client = object()  # 绕过未初始化断言
    return c


async def t_brief() -> None:
    print("\n=== 1. brief 模式：思考与正文分离 ===")
    c = make_client("brief")
    fake = FakeRaw()
    c._raw_chat = fake
    res: ThinkResult = await c.chat_detailed(MSGS, max_tokens=160)

    check("返回 ThinkResult", isinstance(res, ThinkResult))
    check("正文非空", bool(res.text), repr(res.text))
    check("思考非空", bool(res.thinking), repr(res.thinking))
    check("思考未降级", not res.degraded)
    check("模式为 brief", res.mode == "brief", res.mode)
    check("正文不含思考内容", "对方想吓我" not in res.text, res.text)
    check("调用了两次（思考 + 正文）", len(fake.calls) == 2, str(fake.calls))
    think_call = fake.calls[0]
    check("思考阶段先执行", think_call["is_think"])
    check("思考阶段不开启原生 think", think_call["think"] is False)
    check("思考预算受 think_budget 限制",
          think_call["num_predict"] == c.think_budget, str(think_call))
    check("正文预算仍为 max_tokens", fake.calls[1]["num_predict"] == 160)
    check("正文阶段注入了内心活动", fake.calls[1]["n_msgs"] == 2)


async def t_injection() -> None:
    print("\n=== 2. 思考注入只改 system，不动对话历史 ===")
    c = make_client("brief")
    injected = c._inject_thinking(MSGS, "测试思考")
    check("消息条数不变", len(injected) == len(MSGS))
    check("原始消息未被修改", MSGS[0]["content"].endswith("退休中学教师。"),
          MSGS[0]["content"])
    check("system 被追加内心活动", "内心活动" in injected[0]["content"])
    check("user 消息保持原样", injected[1]["content"] == MSGS[1]["content"])
    check("强调不可外泄", "不能对对方说出口" in injected[0]["content"])

    no_sys = [{"role": "user", "content": "hi"}]
    out = c._inject_thinking(no_sys, "想了一下")
    check("无 system 时插到最前", out[0]["role"] == "system")


async def t_brief_degrade() -> None:
    print("\n=== 3. 思考超时/失败：降级但不阻断回复 ===")
    c = make_client("brief")
    fake = FakeRaw(think_raises=asyncio.TimeoutError())
    c._raw_chat = fake
    res = await c.chat_detailed(MSGS, max_tokens=160)
    check("降级标记为真", res.degraded)
    check("正文仍然产出", bool(res.text), repr(res.text))
    check("思考为空", res.thinking == "")
    check("降级有说明", bool(res.note), res.note)


async def t_empty_reply_retry() -> None:
    print("\n=== 4. 正文为空：去掉思考注入重试 ===")
    c = make_client("brief")
    calls = []

    async def raw(messages, *, model=None, temperature=0.0,
                  num_predict=0, think=False, stop=None):
        last = messages[-1].get("content", "") if messages else ""
        is_think = "内心活动" in last
        injected = "内心活动｜" in messages[0].get("content", "")
        calls.append({"is_think": is_think, "injected": injected})
        if is_think:
            return "想了想", ""
        # 带注入的正文调用返回空（模拟思考反噬），无注入的重试返回结果
        if injected:
            return "", ""
        return "喂？你说啥？", ""

    c._raw_chat = raw
    res = await c.chat_detailed(MSGS, max_tokens=160)
    check("重试后拿到正文", res.text == "喂？你说啥？", repr(res.text))
    check("标记为降级", res.degraded)
    check("重试不再注入思考",
          any((not c["is_think"]) and (not c["injected"]) for c in calls), str(calls))


async def t_native() -> None:
    print("\n=== 5. native 模式：thinking 字段与正文分离 ===")
    c = make_client("native")
    calls = []

    async def raw(messages, *, model=None, temperature=0.0,
                  num_predict=0, think=False, stop=None):
        calls.append({"think": think, "num_predict": num_predict})
        if think:
            return "", "Thinking Process: 分析对方意图……"
        return "你说什么？我听不清。", ""

    c._raw_chat = raw
    res = await c.chat_detailed(MSGS, max_tokens=160)
    check("原生思考被捕获", "分析对方意图" in res.thinking, repr(res.thinking))
    check("剥离了 Thinking Process 前缀",
          not res.thinking.startswith("Thinking Process"), repr(res.thinking))
    check("正文非空", bool(res.text))
    check("预算叠加（正文 + 思考）",
          calls[0]["num_predict"] == 160 + c.native_think_budget, str(calls[0]))

    print("\n=== 6. native 思考吃光预算：自动降级直出 ===")
    c2 = make_client("native")
    calls2 = []

    async def raw2(messages, *, model=None, temperature=0.0,
                   num_predict=0, think=False, stop=None):
        calls2.append({"think": think})
        if think:
            return "", "思考……" * 500      # 思考吃光预算，正文为空
        return "我找下老花镜。", ""

    c2._raw_chat = raw2
    res2 = await c2.chat_detailed(MSGS, max_tokens=160)
    check("降级重试拿到正文", res2.text == "我找下老花镜。", repr(res2.text))
    check("标记为降级", res2.degraded)
    check("重试关闭原生思考", calls2[-1]["think"] is False, str(calls2))


async def t_off() -> None:
    print("\n=== 7. off 模式：单次调用，无思考 ===")
    c = make_client("off")
    fake = FakeRaw()
    c._raw_chat = fake
    res = await c.chat_detailed(MSGS, max_tokens=160)
    check("只调用一次", len(fake.calls) == 1, str(fake.calls))
    check("无思考内容", res.thinking == "")
    check("正文非空", bool(res.text))


async def t_stream() -> None:
    print("\n=== 8. 流式事件：thinking 先于 content，正文不含思考 ===")
    c = make_client("brief")
    c._raw_chat = FakeRaw(thinking="先判断对方在吓唬我", reply="啊？你说啥？")

    events: list[dict] = []
    c.client  # noqa: B018  # 确保属性可访问（已置桩）

    async def fake_stream(messages, *, model=None, temperature=0.0,
                          num_predict=0, think=False):
        # 模拟 Ollama 流式分片
        for ch in ["啊？", "你说啥？"]:
            yield {"message": {"content": ch, "thinking": ""}}
        yield {"message": {"content": "", "thinking": ""}, "done": True}

    # 直接用桩替换底层流式请求，验证事件组装逻辑
    import json

    class _Resp:
        async def __aenter__(self_inner):
            return self_inner

        async def __aexit__(self_inner, *exc):
            return None

        def raise_for_status(self_inner):
            return None

        async def aiter_lines(self_inner):
            for part in ['{"message":{"content":"啊？","thinking":""},"done":false}',
                         '{"message":{"content":"你说啥？","thinking":""},"done":false}',
                         '{"message":{"content":"","thinking":""},"done":true}']:
                yield part

    class _Client:
        def stream(self_inner, *args, **kwargs):
            return _Resp()

    c._client = _Client()
    async for ev in c.chat_stream_events(MSGS, max_tokens=120):
        events.append(ev)

    types = [e["type"] for e in events]
    check("首事件是 thinking", types[0] == "thinking", str(types))
    check("存在 content 事件", "content" in types, str(types))
    check("以 done 收尾", types[-1] == "done", str(types))
    done = events[-1]
    check("done 携带完整正文", done["text"] == "啊？你说啥？", repr(done["text"]))
    check("done 携带思考", "吓唬" in done["thinking"], repr(done["thinking"]))


async def t_engine() -> None:
    print("\n=== 9. 引擎集成：思考落 Turn，不进正文 ===")
    cfg = load_config()
    llm = make_client("brief")
    llm._raw_chat = FakeRaw(thinking="1) 他在冒充警察\n2) 我该追问\n3) 装糊涂")
    import tempfile
    from liuer.config import Keystore
    tmp = Path(tempfile.mkdtemp(prefix="liuer-think-"))
    key = Keystore.load_or_create(tmp / ".keystore").key
    store = IntelStore(tmp / "test.db", key)
    pipeline = IntelPipeline(cfg, store, llm)
    library = PersonaLibrary(cfg)
    engine = LiuErEngine(cfg, llm, store, pipeline, library)

    session = engine.start_session(channel="web", peer="tester")
    reply = await engine.on_scammer_message(session, "阿姨，我是公安局的，您涉嫌洗钱。")
    check("回复非空", bool(reply), repr(reply))
    check("思考不出现在回复里", "冒充警察" not in reply, reply)
    check("session 记录了思考", bool(session.last_thinking), repr(session.last_thinking))
    check("记录思考模式", session.last_think_mode == "brief", session.last_think_mode)
    agent_turns = [t for t in session.turns if t.role == "agent"]
    check("Turn 带思考字段", bool(agent_turns) and bool(agent_turns[-1].thinking))


async def t_sanitize() -> None:
    print("\n=== 10. 清洗：思考泄漏与思考文本规整 ===")
    c = make_client("brief")
    leak = LiuErEngine._sanitize("内心活动：他可能真是警察。喂，你说什么？")
    check("剥离思考泄漏", "他可能真是警察" not in leak, leak)
    check("保留正常正文", "喂，你说什么" in leak, leak)

    raw = "<thinking>想了一下</thinking>好的，我的内心活动是：先拖住他"
    cleaned = c._clean_thinking(raw, 400)
    check("去除 think 标签", "<thinking>" not in cleaned, cleaned)
    check("去除应答套话", not cleaned.startswith("好的"), cleaned)
    long = c._clean_thinking("想" * 900, 400)
    check("超长截断", len(long) <= 401, str(len(long)))


async def main() -> int:
    await t_brief()
    await t_injection()
    await t_brief_degrade()
    await t_empty_reply_retry()
    await t_native()
    await t_off()
    await t_stream()
    await t_engine()
    await t_sanitize()
    print(f"\n{'=' * 46}")
    print(f"  通过 {PASS}  失败 {FAIL}")
    print(f"{'=' * 46}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

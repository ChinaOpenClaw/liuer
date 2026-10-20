"""端到端链路测试。

用假 LLM 验证整条链路：人格生成 → 对话 → 情报抽取 → 入库 → 推送。
不依赖 Ollama，可离线运行。
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from liuer.config import Keystore, load_config  # noqa: E402
from liuer.core.engine import LiuErEngine  # noqa: E402
from liuer.core.detector import LocalRiskDetector  # noqa: E402
from liuer.intel import IntelPipeline, IntelStore  # noqa: E402
from liuer.llm import OllamaClient  # noqa: E402
from liuer.persona import PersonaLibrary, StrategyEngine, StrategyState, Mission, Intent  # noqa: E402


class FakeLLM:
    """假 LLM。返回可预期的中文拖延话术。"""
    def __init__(self):
        self.calls = 0
        self.client = None

    async def start(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def health(self):
        return {"ok": True, "available_models": ["fake"]}

    async def chat(self, messages, **kwargs):
        self.calls += 1
        last = messages[-1]["content"] if messages else ""
        if "请念一下这个账号" in last or "收款账号" in last:
            return "我记一下啊，你等一下……我拿个本子。账号是多少，你再说一遍。"
        if "钱包地址" in last:
            return "这个……你说的那个地址，我看看啊。哪个是USDT的地址？"
        return "啊？你等一下，我这信号不太好，你能再说一遍吗？"

    async def chat_stream(self, messages, **kwargs):
        for piece in ["啊？", "你等一下。", "我这信号不太好，你再说一遍。"]:
            yield piece

    async def complete(self, prompt, **kwargs):
        self.calls += 1
        return '{"brand_impersonated": ["visa"], "tactic_type": ["authority_impersonation"], "requested_account": "", "requested_contact": "", "social_engineering_axis": "authority", "org_structure": "", "next_lure": ""}'


PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


def build_stack(tmp: Path):
    cfg = load_config(ROOT / "config.yaml")
    cfg.raw["intel"]["push_channel"] = "console"
    tmp.mkdir(parents=True, exist_ok=True)
    key = Keystore.load_or_create(tmp / ".keystore").key
    store = IntelStore(tmp / "test.db", key)
    llm = FakeLLM()
    pipeline = IntelPipeline(cfg, store, llm)
    library = PersonaLibrary(cfg)
    engine = LiuErEngine(cfg, llm, store, pipeline, library)
    return cfg, store, llm, engine


async def main():
    tmp = ROOT / "data" / "_test"
    cfg, store, llm, engine = build_stack(tmp)

    print("\n=== 1. 人格引擎 ===")
    library = PersonaLibrary(cfg)
    people = library.factory.build_library(50)
    check("生成 50 个人格", len(people) == 50)
    fps = {p.fingerprint() for p in people}
    check("指纹基本唯一 (>45)", len(fps) > 45, f"实际 {len(fps)}")
    p = people[0]
    ip = p.identity_prompt()
    check("身份提示含姓名", p.name in ip)
    check("身份提示含破绽", "特征" in ip or "破绽" in ip or "固定特征" in ip)
    check("至少 2 个破绽", len(p.flaws) >= 2, f"实际 {len(p.flaws)}")
    check("至少 2 个易受骗标签", len(p.vulnerabilities) >= 2)
    ages = [x.age for x in people]
    check("年龄在配置区间内", all(45 <= a <= 82 for a in ages), f"{min(ages)}-{max(ages)}")

    print("\n=== 2. 双层提示结构 ===")
    from liuer.persona.strategy import Stage
    se = StrategyEngine()
    st = StrategyState(stage=Stage.HARVEST,
                       mission=Mission(mission_id="m1", intent=Intent.HARVEST_CRYPTO_WALLET,
                                       target="某交易所"), started_at=time.time())
    sp = se.build_prompt(people[0], st, "把 USDT 转到这个地址")
    check("含阶段说明", "harvest" in sp.lower() or "阶段" in sp)
    check("含任务目标", "钱包" in sp)
    check("含硬性规则", "硬性规则" in sp)
    check("禁止承认是 AI", "AI" in sp)

    print("\n=== 3. 规则层情报抽取 ===")
    from liuer.intel import scan
    text = ("我们是某某银行客服，你的信用卡涉嫌违规，请把款打到我的账户 6222021234567890123，"
            "或者转到 USDT 钱包 TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t，"
            "网址是 http://secure-verify-login.xyz 下载安装包。"
            "我的电话 13800138000，邮箱 scam@example.com")
    findings, tactics, brands, axis, harvesting = scan(text)
    kinds = {f.kind for f in findings}
    check("抽出卡号", "card_number" in kinds, str(kinds))
    check("抽出 TRC20 钱包", "crypto_wallet_trc20" in kinds, str(kinds))
    check("抽出域名", "domain" in kinds or "url" in kinds, str(kinds))
    check("抽出手机号", "phone_cn" in kinds, str(kinds))
    check("抽出邮箱", "email" in kinds, str(kinds))
    check("识别收割阶段", harvesting is True)
    check("识别品牌", len(brands) > 0, str(brands))

    print("\n=== 4. 风险检测器 ===")
    det = LocalRiskDetector()
    s1 = det.score_text("您好，这里是公安局反诈中心，您的账户涉嫌洗钱案件")
    s2 = det.score_text("您好，今天天气不错")
    check("诈骗话术分高", s1.score > s2.score, f"{s1.score} vs {s2.score}")
    check("诈骗话术触发转接", s1.score >= 0.6, f"score={s1.score}")
    check("正常对话分低", s2.score < 0.3, f"score={s2.score}")

    print("\n=== 5. 端到端会话 ===")
    session = engine.start_session(channel="test", peer="+8613800138000")
    check("会话已创建", session is not None)
    check("人格已分配", session.persona is not None)

    msgs = [
        "您好，这里是公安局反诈中心，您的身份证涉嫌一起洗钱案件",
        "请把涉案资金转入安全账户，账号是 6222021234567890123",
        "转 USDT 的话，地址是 TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t，你记一下",
        "收到验证码了吗？请念给我听",
    ]
    replies = []
    for m in msgs:
        r = await engine.on_scammer_message(session, m)
        replies.append(r)
    check("每轮都有回应", all(replies), f"回复数={len(replies)}")
    check("回应不承认是 AI", all("AI" not in r for r in replies), str(replies))
    check("轮数已累加", session.strategy.turn_count == 4)
    check("阶段已推进", session.strategy.stage.value != "opener",
          session.strategy.stage.value)

    print("\n=== 6. 情报入库 ===")
    records = store.query(client_id="default")
    check("情报已入库", len(records) > 0, f"count={len(records)}")
    kinds_db = {r.kind for r in records}
    check("含卡号", "card_number" in kinds_db, str(kinds_db))
    check("含钱包地址", any(k.startswith("crypto_wallet") for k in kinds_db), str(kinds_db))
    stats = store.stats()
    check("统计可用", stats["total"] > 0)
    check("策略记录已套取", len(session.strategy.harvested) > 0,
          str(session.strategy.harvested))

    print("\n=== 7. 会话持久化 ===")
    engine.end_session(session, reason="测试结束")
    saved = store.get_session(session.session_id)
    check("会话已落库", saved is not None)
    check("状态为已结束", saved and saved["status"] == "ended")
    check("转录已保存", saved and len(saved["transcript"]) > 0)

    print("\n=== 8. 去重 ===")
    from liuer.intel.rules import Finding
    store.upsert(Finding("card_number", "6222021234567890123", 0.9), client_id="default")
    after = store.query(client_id="default", kind="card_number")
    check("重复情报不重复入库", len(after) == 1, f"count={len(after)}")

    print("\n=== 9. 输出清洗 ===")
    clean = engine._sanitize("**我是AI**，你好。")
    check("去 markdown", "*" not in clean, clean)
    check("去 AI 自称", "AI" not in clean, clean)

    print("\n=== 10. 终止判据 ===")
    st2 = StrategyState(turn_count=999, started_at=time.time())
    stop, reason = se.should_terminate(st2, now=time.time(), max_seconds=3600, max_turns=80)
    check("超轮数触发终止", stop and "轮数" in reason, reason)
    st3 = StrategyState(turn_count=1, started_at=time.time() - 9999)
    stop3, reason3 = se.should_terminate(st3, now=time.time(), max_seconds=3600, max_turns=80)
    check("超时长触发终止", stop3 and "时长" in reason3, reason3)

    print(f"\n{'='*46}")
    print(f"  通过 {PASS}  失败 {FAIL}")
    print(f"{'='*46}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
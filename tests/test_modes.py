"""12 模态轮询 + 人格档案的测试。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from liuer.persona import (  # noqa: E402
    MODE_CYCLE,
    MODE_LABELS,
    ResponseMode,
    StrategyEngine,
    StrategyState,
    load_profile,
    discover_profiles,
)
from liuer.persona.strategy import Stage  # noqa: E402

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


def main() -> int:
    print("\n=== 1. 轮询序列 ===")
    labels = [MODE_LABELS[m] for m in MODE_CYCLE]
    check("序列长度为 12", len(MODE_CYCLE) == 12, str(len(MODE_CYCLE)))
    check("顺序符合需求", labels == ["语境", "个人情况", "反身性格", "同理心",
                                 "爱心安抚", "耐心", "诡辩", "套话",
                                 "耐心", "反身性格", "同理心", "爱心安抚"],
          str(labels))
    # 柔性模态权重加倍
    from collections import Counter
    cnt = Counter(labels)
    check("耐心出现 2 次", cnt["耐心"] == 2, str(cnt))
    check("同理心出现 2 次", cnt["同理心"] == 2, str(cnt))
    check("爱心安抚出现 2 次", cnt["爱心安抚"] == 2, str(cnt))
    check("反身性格出现 2 次", cnt["反身性格"] == 2, str(cnt))
    check("套话出现 1 次", cnt["套话"] == 1, str(cnt))
    check("诡辩出现 1 次", cnt["诡辩"] == 1, str(cnt))

    print("\n=== 2. 轮询推进（全放开阶段）===")
    # 纯轮询验证要在 HARVEST 阶段做：OPENER 会过滤套话，序列本就会跳过
    st = StrategyState(stage=Stage.HARVEST)
    seq = [st.next_mode() for _ in range(12)]
    check("推进 12 次得到完整一轮", seq == list(MODE_CYCLE), str([m.value for m in seq]))
    st2 = StrategyState(stage=Stage.HARVEST)
    first24 = [st2.next_mode() for _ in range(24)]
    check("第二轮循环一致", first24[:12] == first24[12:24],
          str([m.value for m in first24[12:]]))

    print("\n=== 2b. 早期阶段过滤套话后的序列 ===")
    st2b = StrategyState(stage=Stage.OPENER)
    early_seq = [st2b.next_mode() for _ in range(12)]
    check("套话被跳过", ResponseMode.ELICIT not in early_seq,
          str([m.value for m in early_seq]))
    check("其余模态仍推进", len(set(early_seq)) >= 7, str(len(set(early_seq))))

    print("\n=== 3. 阶段约束：早期不主动套话 ===")
    st3 = StrategyState(stage=Stage.OPENER)
    early = [st3.next_mode() for _ in range(12)]
    check("OPENER 阶段无套话", ResponseMode.ELICIT not in early,
          str([m.value for m in early]))
    st4 = StrategyState(stage=Stage.HARVEST)
    late = [st4.next_mode() for _ in range(12)]
    check("HARVEST 阶段有套话", ResponseMode.ELICIT in late,
          str([m.value for m in late]))

    print("\n=== 4. 起疑时收敛到柔性 ===")
    st5 = StrategyState(stage=Stage.HARVEST)
    st5.suspicion_level = 0.6
    seq5 = [st5.next_mode() for _ in range(12)]
    check("起疑后无诡辩", ResponseMode.SOPHISTRY not in seq5,
          str([m.value for m in seq5]))
    check("起疑后无套话", ResponseMode.ELICIT not in seq5,
          str([m.value for m in seq5]))
    check("起疑后仍有耐心", ResponseMode.PATIENCE in seq5)

    print("\n=== 5. 诡辩不连续 ===")
    st6 = StrategyState(stage=Stage.HARVEST)
    seq6 = [st6.next_mode() for _ in range(24)]
    bad = [i for i in range(1, len(seq6))
           if seq6[i] == seq6[i - 1] and seq6[i] in (ResponseMode.SOPHISTRY,
                                                     ResponseMode.ELICIT)]
    check("刚性模态不连续", not bad, str(bad))

    print("\n=== 6. 模态进入提示词 ===")
    eng = StrategyEngine()
    st7 = StrategyState(stage=Stage.HARVEST)
    m = st7.next_mode()
    from liuer.persona import load_profile as lp
    prof = lp(ROOT / "profiles" / "mother-1966.yaml")
    person = prof.to_person()
    prompt = eng.build_prompt(person, st7, "把钱转到这个账户")
    check("提示词含模态指令", "本轮模态" in prompt, prompt[:80])
    check("当前模态是首个(语境)", m == ResponseMode.CONTEXT, m.value)

    print("\n=== 7. 启动案例档案 ===")
    check("档案可载入", prof.codename == "启动案例·1966", prof.codename)
    check("标记为真实人物", prof.real_person is True)
    p = person
    check("年龄 60", p.age == 60, str(p.age))
    check("姓名已脱敏(非真名)", p.name == "沈静宜", p.name)
    check("语境非空", len(p.context_profile) > 50, str(len(p.context_profile)))
    check("个人情况非空", len(p.personal_situation) > 50)
    check("反身性格非空", len(p.reflexive_personality) > 50)
    check("同理心非空", len(p.empathy_style) > 5)
    check("安抚非空", len(p.comfort_style) > 5)
    check("耐心非空", len(p.patience_style) > 5)
    check("诡辩/讲理非空", len(p.reasoning_style) > 5)
    check("套话非空", len(p.elicit_style) > 5)
    check("易感点 >=5 条", len(p.susceptibility) >= 5, str(len(p.susceptibility)))
    check("破绽 >=3 条", len(p.flaws) >= 3, str(len(p.flaws)))

    print("\n=== 8. 扩展维度进入身份提示 ===")
    ip = p.identity_prompt()
    check("含语境", p.context_profile[:20] in ip)
    check("含个人情况", p.personal_situation[:20] in ip)
    check("含反身性格", p.reflexive_personality[:20] in ip)
    check("含易感点", "容易被说动" in ip)
    check("含 1966 语境关键词", "1966" in ip)

    print("\n=== 9. 隐私：真实档案不进可公开列表 ===")
    public = discover_profiles(ROOT / "profiles")
    names = [x.name for x in public]
    check("真实档案被排除", "mother-1966.yaml" not in names, str(names))
    check("模板不在可公开列表", "template.yaml" not in names, str(names))

    print("\n=== 10. 模板档案可用 ===")
    tpl = load_profile(ROOT / "profiles" / "template.yaml")
    tp = tpl.to_person()
    check("模板可转人格", tp.name == "李慧兰", tp.name)
    check("模板非真实人物", tpl.real_person is False)

    print(f"\n{'=' * 46}\n  通过 {PASS}  失败 {FAIL}\n{'=' * 46}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
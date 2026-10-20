"""响应模式轮询调度器。

为什么要用轮询而不是固定策略：

固定策略下模型会形成话术惯性，几轮之后回复模式高度重复，
稍有经验的诈骗方就能察觉对面在"按套路走"。轮询强制每一轮切换
行为模态，让对话呈现真人般的起伏——时而共情、时而较真、时而走神。

轮询序列按需求定义，其中「耐心 / 反身性 / 同理心 / 爱心安抚」
各出现两次，权重高于其他模式。原因是这四类属于"柔性模态"，
能把对话拉长而不引起警觉，是拖延战术的主要载体；
而「诡辩 / 套话」属于"刚性模态"，出现过多会让对方起疑。

序列：
语境 → 个人情况 → 反身性格 → 同理心 → 爱心安抚 → 耐心
→ 诡辩 → 套话 → 耐心 → 反身性 → 同理心 → 爱心安抚
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator


class ResponseMode(str, Enum):
    """响应模态。每轮切换一种。"""

    CONTEXT = "context"            # 语境：时代背景与文化语境
    SITUATION = "situation"        # 个人情况：家庭、健康、经济
    REFLEXIVE = "reflexive"        # 反身性格：自我反思、自我质疑
    EMPATHY = "empathy"            # 同理心：理解对方处境
    COMFORT = "comfort"            # 爱心安抚：温暖关怀
    PATIENCE = "patience"          # 耐心：不急不躁
    SOPHISTRY = "sophistry"        # 诡辩：用自己的逻辑追问反驳
    ELICIT = "elicit"              # 套话：主动索取具体信息


# 轮询序列。耐心/反身/同理心/安抚各占两席，柔性模态权重加倍。
MODE_CYCLE: tuple[ResponseMode, ...] = (
    ResponseMode.CONTEXT,      # 1  语境
    ResponseMode.SITUATION,    # 2  个人情况
    ResponseMode.REFLEXIVE,    # 3  反身性格
    ResponseMode.EMPATHY,      # 4  同理心
    ResponseMode.COMFORT,      # 5  爱心安抚
    ResponseMode.PATIENCE,     # 6  耐心
    ResponseMode.SOPHISTRY,    # 7  诡辩
    ResponseMode.ELICIT,       # 8  套话
    ResponseMode.PATIENCE,     # 9  耐心
    ResponseMode.REFLEXIVE,    # 10 反身性
    ResponseMode.EMPATHY,      # 11 同理心
    ResponseMode.COMFORT,      # 12 爱心安抚
)

MODE_LABELS: dict[ResponseMode, str] = {
    ResponseMode.CONTEXT: "语境",
    ResponseMode.SITUATION: "个人情况",
    ResponseMode.REFLEXIVE: "反身性格",
    ResponseMode.EMPATHY: "同理心",
    ResponseMode.COMFORT: "爱心安抚",
    ResponseMode.PATIENCE: "耐心",
    ResponseMode.SOPHISTRY: "诡辩",
    ResponseMode.ELICIT: "套话",
}

# 每种模态的指令。写法上刻意不做模板化，避免模型套出固定句式。
MODE_INSTRUCTIONS: dict[ResponseMode, str] = {
    ResponseMode.CONTEXT: """【本轮模态：语境】
用你这一代人的生活经验和时代背景去理解对方说的话。可以自然带出你这个年纪的人才会有的参照物：单位、粮票、报纸、广播、早年的办事流程。不要刻意怀旧，是顺着对方的话题自然带出来的。""",

    ResponseMode.SITUATION: """【本轮模态：个人情况】
自然地提到一点自己的实际情况——家里有几口人、身体有什么老毛病、退休金什么时候到账、子女在哪边。用自身经历去比对对方说的事，这是你理解世界的方式。一次只说一件，别说成汇报。""",

    ResponseMode.REFLEXIVE: """【本轮模态：反身性格】
表现出自我反省。可以怀疑是自己没听明白、是自己年纪大了记性差、是自己太谨慎多心。
注意：这种自我质疑能让对方觉得你真实可信，但不要把怀疑指向对方——你只是在怀疑自己。""",

    ResponseMode.EMPATHY: """【本轮模态：同理心】
对对方表达理解。比如觉得对方这么晚还在上班不容易、听出对方声音有点累、说对方态度倒是挺好。
这是让对方卸下防备最有效的方式——被理解的人会愿意多说。""",

    ResponseMode.COMFORT: """【本轮模态：爱心安抚】
用长辈关怀晚辈的语气说话。可以叮嘱对方注意身体、别太辛苦、外面冷多穿点。
语气要温，但不要煽情，就是这个年纪的人随口会说的那种关心。""",

    ResponseMode.PATIENCE: """【本轮模态：耐心】
表现出完全不急。可以慢慢来、让对方别催、说自己有的是时间。
这是拖延的核心载体——你越不着急，对方越舍不得挂。可以顺势要求对方再讲一遍细节。""",

    ResponseMode.SOPHISTRY: """【本轮模态：诡辩】
你是有文化的人，会讲道理。用你自己的逻辑去追问或反驳：按理说应该是什么流程、以前办过类似的事是怎么办的、对方的说法哪里跟你知道的对不上。
不要显得咄咄逼人，是认真的、想把事情弄明白的那种较真。对方为了说服你，只能把更多细节说出来。""",

    ResponseMode.ELICIT: """【本轮模态：套话】
主动索取具体信息。要的是可被验证的硬信息：工号、姓名、所属单位全称、办公地址、回拨号码、收款账户、办理网点、网址。
可以包装成合理的关心——我要记下来、我要跟家里人商量、我要去网点核实。对方给得越具体，越有行动价值。""",
}

# 柔性模态：用于统计与调度权重判断是否跳过多轮
SOFT_MODES: frozenset[ResponseMode] = frozenset({
    ResponseMode.EMPATHY, ResponseMode.COMFORT,
    ResponseMode.PATIENCE, ResponseMode.REFLEXIVE,
})

# 刚性模态：连续出现会让对方警觉，调度时可做约束
HARD_MODES: frozenset[ResponseMode] = frozenset({
    ResponseMode.SOPHISTRY, ResponseMode.ELICIT,
})


@dataclass
class ModeScheduler:
    """模态轮询器。

    按 MODE_CYCLE 循环推进。提供两个约束：
    1. 刚性模态（诡辩/套话）不会连续出现——序列本身已保证，
       但在 HARVEST 阶段之外会跳过套话，避免过早暴露目的。
    2. 检测到对方起疑时，强制切到柔性模态降速。
    """

    index: int = 0
    history: list[ResponseMode] = field(default_factory=list)
    # 允许的模态集合，None 表示全部允许
    allowed: frozenset[ResponseMode] | None = None
    # 对方起疑时强制柔性
    force_soft: bool = False

    def next(self) -> ResponseMode:
        """取下一个模态并推进游标。"""
        for _ in range(len(MODE_CYCLE)):
            mode = MODE_CYCLE[self.index % len(MODE_CYCLE)]
            self.index += 1
            if self._is_eligible(mode):
                self.history.append(mode)
                return mode
        # 全部被过滤时退回柔性
        fallback = ResponseMode.PATIENCE
        self.history.append(fallback)
        return fallback

    def _is_eligible(self, mode: ResponseMode) -> bool:
        if self.force_soft and mode in HARD_MODES:
            return False
        if self.allowed is not None and mode not in self.allowed:
            return False
        # 刚性模态不与上一次重复（含被过滤前的历史）
        if mode in HARD_MODES and self.history and self.history[-1] == mode:
            return False
        return True

    def peek(self, n: int = 6) -> list[ResponseMode]:
        """预览接下来的 n 个模态，不推进游标。用于调试面板。"""
        out: list[ResponseMode] = []
        saved = self.index
        saved_hist = list(self.history)
        for _ in range(n):
            out.append(self.next())
        self.index = saved
        self.history = saved_hist
        return out

    def cycle(self, n: int = 12) -> Iterator[ResponseMode]:
        for _ in range(n):
            yield self.next()

    def stats(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for m in self.history:
            counts[MODE_LABELS[m]] = counts.get(MODE_LABELS[m], 0) + 1
        return counts

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "current": MODE_LABELS[self.history[-1]] if self.history else None,
            "history": [MODE_LABELS[m] for m in self.history],
            "stats": self.stats(),
        }


def build_mode_prompt(mode: ResponseMode, persona_name: str = "") -> str:
    """渲染单个模态的指令文本。"""
    return MODE_INSTRUCTIONS.get(mode, "")


def mode_cycle_labels() -> list[str]:
    return [MODE_LABELS[m] for m in MODE_CYCLE]
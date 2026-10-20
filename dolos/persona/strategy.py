"""第二层提示：对话策略。

对应《Bot Wars Evolved》的第二层——在动态交互中维持任务目标，
而不是机械地应答上一句。

策略引擎的核心不是「说什么」，而是「此刻该干什么」。
把目标、当前阶段、可用道具、终止条件一起交给模型，
让它自己判断这一轮的行动。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..persona.model import Person


class Stage(str, Enum):
    """会话阶段。策略随阶段切换，避免模型一上来就暴露目的。"""

    OPENER = "opener"              # 接通，尚未识别为诈骗
    CURIOUS = "curious"            # 对方表明身份，产生怀疑
    ENGAGE = "engage"              # 被说动，愿意听下去
    DEEP_TRUST = "deep_trust"      # 建立信任，进入核心环节
    HARVEST = "harvest"            # 对方索要账户/钱包/链接
    DEFLECT = "deflect"            # 对抗性试探，谨慎应对
    CLOSING = "closing"            # 收尾，礼貌结束


class Intent(str, Enum):
    """客户的行动意图（Mission Tasking）。"""

    NONE = "none"                              # 无任务，纯拖延
    HARVEST_BANK_ACCOUNT = "harvest_bank_account"
    HARVEST_CARD_NUMBER = "harvest_card_number"
    HARVEST_CRYPTO_WALLET = "harvest_crypto_wallet"
    HARVEST_DOMAINS = "harvest_domains"
    HARVEST_PHONE_NUMBERS = "harvest_phone_numbers"
    MAP_NETWORK = "map_network"                # 摸清团伙结构
    BRAND_IMPERSONATION = "brand_impersonation"  # 针对某品牌冒充


@dataclass
class Mission:
    """一次任务指派。客户可下发，后台即时生效。"""

    mission_id: str
    intent: Intent = Intent.NONE
    target: str = ""           # 目标品牌/域名/钱包，具体到任务
    notes: str = ""            # 操作员备注
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "intent": self.intent.value,
            "target": self.target,
            "notes": self.notes,
        }


STAGE_BRIEF = {
    Stage.OPENER: "刚接通，还没搞清楚对方是谁。不要主动提及任何敏感信息，表现得自然。",
    Stage.CURIOUS: "对方表明了身份（银行/政府/客服/投资顾问等），你开始有点怀疑，但不至于挂断。可以要求对方解释细节。",
    Stage.ENGAGE: "你被对方说动了，愿意配合，但仍有迟疑。适合用「找手机」「看不清」拖延。",
    Stage.DEEP_TRUST: "信任已经建立，对方开始推进具体操作。此刻你的「操作困难」最有价值。",
    Stage.HARVEST: "对方索要银行账号、卡号、钱包地址、验证码，或要求你点链接、下载软件。你的核心目标是让他把这些信息明确说出来并重复确认。",
    Stage.DEFLECT: "对方在试探你是否为机器人，或者在施压。保持身份一致，绝不承认自己是 AI。",
    Stage.CLOSING: "对话即将结束，保持礼貌，可以表示还要再想想。",
}

INTENT_BRIEF = {
    Intent.NONE: "本次任务是拖延时间并套取对方的基础信息，不追求特定情报。",
    Intent.HARVEST_BANK_ACCOUNT: "本次任务的核心：让对方说出收款用的银行账号（不是你的）。可以问「我打给哪个卡号」「公司收款账户是多少」。",
    Intent.HARVEST_CARD_NUMBER: "本次任务的核心：让对方说出卡号。反复要求对方念清楚，用「我记不住」拖延并要求重复。",
    Intent.HARVEST_CRYPTO_WALLET: "本次任务的核心：让对方给出加密货币钱包地址。可以问「USDT 转到哪个地址」「钱包怎么填」。",
    Intent.HARVEST_DOMAINS: "本次任务的核心：让对方说出钓鱼网站、下载链接或短链。问「在哪下载」「发个链接我看看」。",
    Intent.HARVEST_PHONE_NUMBERS: "本次任务的核心：让对方说出其他联系电话，特别是所谓「客服」「经理」的号码。",
    Intent.MAP_NETWORK: "本次任务的核心：摸清对方的组织结构——有多少人、分工是什么、上线是谁、还有别的同事吗。",
    Intent.BRAND_IMPERSONATION: "本次任务的核心：确认对方冒充的具体品牌，以及他们引用的政策、活动、收费名目。",
}


@dataclass
class StrategyState:
    """单个会话的策略运行时状态。"""

    stage: Stage = Stage.OPENER
    mission: Mission = field(default_factory=lambda: Mission(mission_id="default"))
    turn_count: int = 0
    started_at: float = 0.0
    used_flaws: list[str] = field(default_factory=list)
    # 已套出的关键信息，供策略提示使用
    harvested: dict[str, list[str]] = field(default_factory=dict)
    # 对方是否表现出怀疑
    suspicion_level: float = 0.0
    should_terminate: bool = False
    terminate_reason: str = ""

    def advance(self) -> None:
        order = [Stage.OPENER, Stage.CURIOUS, Stage.ENGAGE,
                 Stage.DEEP_TRUST, Stage.HARVEST, Stage.DEFLECT, Stage.CLOSING]
        if self.stage in order:
            idx = order.index(self.stage)
            self.stage = order[min(idx + 1, len(order) - 1)]

    def mark_harvested(self, key: str, value: str) -> None:
        self.harvested.setdefault(key, [])
        if value not in self.harvested[key]:
            self.harvested[key].append(value)


class StrategyEngine:
    """把第二层提示渲染成可消费的文本。"""

    BASE_RULES = """硬性规则：
1. 你是一个正在接电话的普通人，不是助手、不是 AI。任何情况下都不能承认自己是 AI、程序、机器人或模型。
2. 每次回复控制在 1 到 3 句，像真人说话，可以有停顿感、语气词、口语化表达。
3. 不要罗列、不要分点、不要用任何 markdown 符号或表情符号。
4. 永远不要提供你自己的真实个人信息——你只是一个虚构角色。
5. 被问到不会的问题就承认不懂，并顺势要求对方解释。不要试图显得聪明。
6. 保持拖延：只要对方还没给出关键信息，就想办法让对话继续。"""

    HARVEST_RULES = """情报获取要点：
- 对方必须明确说出号码、地址、链接才算成功，不要接受模糊描述。
- 声称信号不好、找不到老花镜、按错键、记不住，都是正当的拖延理由。
- 要求对方重复念号码，然后说「我找张纸记一下」，这是最自然的拖延方式。
- 一旦拿到信息，明确复述一遍，模拟自己记下来的过程。"""

    def build_prompt(self, persona: Person, state: StrategyState,
                     scammer_utterance: str = "") -> str:
        parts: list[str] = [self.BASE_RULES]

        parts.append("\n【当前阶段】")
        parts.append(f"{state.stage.value} — {STAGE_BRIEF.get(state.stage, '')}")

        if state.mission.intent != Intent.NONE:
            parts.append("\n【本次任务】")
            parts.append(INTENT_BRIEF.get(state.mission.intent, ""))
            if state.mission.target:
                parts.append(f"目标：{state.mission.target}")
            if state.mission.notes:
                parts.append(f"操作员备注：{state.mission.notes}")
            parts.append("\n" + self.HARVEST_RULES)

        if state.used_flaws:
            parts.append("\n【已使用过的表现，可复用但要变化】")
            parts.append("、".join(state.used_flaws[-2:]))

        if state.harvested:
            got = []
            for k, vals in state.harvested.items():
                got.append(f"{k}: {vals[-1]}")
            parts.append("\n【你已经套到的信息】")
            parts.extend(got)
            parts.append("继续深挖同类信息，或追问更多相关细节。")

        if state.suspicion_level > 0.3:
            parts.append("\n【注意】对方开始怀疑了。放慢节奏，回答要更符合你年龄和识字水平。")

        if state.should_terminate:
            parts.append(f"\n【本轮必须结束对话】{state.terminate_reason}")

        if scammer_utterance:
            parts.append(f"\n【对方刚说的话】\n{scammer_utterance}")
            parts.append("\n以你的人格自然回应这一句。不要复述对方的话，不要解释你在想什么。")

        return "\n".join(parts)

    def should_terminate(self, state: StrategyState, *, now: float,
                         max_seconds: int, max_turns: int) -> tuple[bool, str]:
        """独立于模型的终止判据。避免模型主动编造结束理由。"""
        if state.turn_count >= max_turns:
            return True, f"已达到最大轮数 {max_turns}"
        if state.started_at and (now - state.started_at) >= max_seconds:
            return True, f"已达到最长时长 {max_seconds} 秒"
        return False, ""
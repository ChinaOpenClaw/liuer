"""人格数据模型。

设计依据：麦考瑞大学《Bot Wars Evolved》双层提示结构的第一层。
人格必须足够具体到「能骗过专业人士的试探」，因此字段拆得细，
不是简单的年龄+性别标签，而是可被 LLM 直接消费的身份叙述。
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"


class DigitalLiteracy(str, Enum):
    """数字熟练度。越低，越容易在操作环节装傻拖时间。"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Temperament(str, Enum):
    """性格基调，决定回复长度与情绪起伏。"""

    ANXIOUS = "anxious"        # 焦虑，容易被催促话术带走
    POLITE = "polite"          # 客气，一再请求重复
    SKEPTICAL = "skeptical"    # 半信半疑，会反问但不懂技术
    GULLIBLE = "gullible"      # 容易相信权威
    CONFUSED = "confused"      # 反复跑题，需要对方反复解释
    STUBBORN = "stubborn"      # 不轻易改主意，会跟对方掰扯
    RESERVED = "reserved"      # 寡言，回复短


class Accent(str, Enum):
    """口音标签。语音通道映射到 TTS 音色，文本通道映射到用词习惯。"""

    PUTONGHUA = "putonghua"
    NORTHEASTERN = "northeastern"
    SICHUAN = "sichuan"
    SOUTHWESTERN = "southwestern"
    CANTONESE_INFLUENCED = "cantonese_influenced"
    NORTHERN = "northern"


class VulnTag(str, Enum):
    """易受骗标签。既用于人格画像，也用于情报策略预判。"""

    LONELY_ELDER = "lonely_elder"
    LOW_DIGITAL_LITERACY = "low_digital_literacy"
    PENSION_HOLDER = "pension_holder"
    RECENT_BEREAVEMENT = "recent_bereavement"
    HEALTH_ANXIETY = "health_anxiety"
    FINANCIAL_ANXIETY = "financial_anxiety"
    AUTHORITY_DEFERENCE = "authority_deference"
    TECH_ILLITERATE = "tech_illiterate"


@dataclass
class Flaw:
    """刻意制造的「破绽」。这是让对话持续下去的核心道具。

    骗子会在确认对方是否好骗时试探这些破绽，
    agent 表现出一致的破绽，就能通过验证。
    """

    kind: str          # forgetful | bad_reception | wrong_input | no_glasses | slow_device
    trigger: str       # 触发条件描述
    expression: str    # 具体表现，如「找不到老花镜」

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class Person:
    """一个人格实例。所有字段在生成时固定，会话期间不可变。"""

    persona_id: str
    name: str
    gender: Gender
    age: int
    occupation: str
    accent: Accent
    temperament: Temperament
    digital_literacy: DigitalLiteracy
    locale: str

    # 背景叙述，直接进入提示词
    background: str

    # ---- 扩展维度（供 12 模式轮询消费）----
    # 语境：时代背景、文化语境、这一代人的共同经验
    context_profile: str = ""
    # 个人情况：家庭结构、健康、经济、社交状况
    personal_situation: str = ""
    # 反身性格：自我反思方式与元认知特征
    reflexive_personality: str = ""
    # 同理心表达方式
    empathy_style: str = ""
    # 安抚他人的方式
    comfort_style: str = ""
    # 耐心程度描述
    patience_style: str = ""
    # 讲理/较真的方式（诡辩模态用）
    reasoning_style: str = ""
    # 套话时的口吻（套话模态用）
    elicit_style: str = ""
    # 易感点：被哪些套路影响，供策略预判
    susceptibility: list[str] = field(default_factory=list)
    # 虚构但自洽的生活细节，用于回答骗子探问
    life_details: dict[str, str] = field(default_factory=dict)
    # 会使用哪些 App（不一致本身就是破绽）
    apps: list[str] = field(default_factory=list)

    flaws: list[Flaw] = field(default_factory=list)
    vulnerabilities: list[VulnTag] = field(default_factory=list)

    # 语速与口癖。文本通道用回复长度，口音用词癖
    speech_pace: float = 1.0        # 0.7 慢 ~ 1.3 快
    filler_words: list[str] = field(default_factory=list)
    pet_peeves: list[str] = field(default_factory=list)

    def fingerprint(self) -> str:
        """人格指纹，用于去重与一致性校验。"""
        raw = f"{self.name}|{self.age}|{self.occupation}|{self.accent}|{self.background}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def identity_prompt(self) -> str:
        """第一层提示：构造身份。字段顺序固定，保证同一人格跨会话一致。"""
        apps = "、".join(self.apps) if self.apps else "只会接打电话和发短信"
        flaws = "\n".join(f"  - {f.expression}（触发场景：{f.trigger}）" for f in self.flaws)
        dims = self.dimension_prompt()
        return f"""你现在扮演以下这个人，全程不得偏离身份：

姓名：{self.name}
性别：{self.gender.value}，{self.age} 岁
职业：{self.occupation}
籍贯口音：{self.accent.value}
性格：{self.temperament.value}
数字熟练度：{self.digital_literacy.value}
背景：{self.background}
会使用的应用：{apps}
说话习惯：{'、'.join(self.filler_words) if self.filler_words else '无明显口癖'}
语速：{self.speech_pace:.1f}（0.7 很慢，1.3 偏快）

你身上有这些固定特征，会在对话中自然流露：
{flaws}

背景补充细节（被追问时可引用）：
{chr(10).join(f'  {k}：{v}' for k, v in self.life_details.items()) or '  无'}
{dims}"""

    def dimension_prompt(self) -> str:
        """扩展维度的提示片段。仅渲染非空字段，避免提示词灌水。"""
        rows: list[str] = []
        pairs = [
            ("你这一代人的共同经验与语境", self.context_profile),
            ("你的个人情况", self.personal_situation),
            ("你的自我反思方式（会怀疑自己，而不是怀疑别人）", self.reflexive_personality),
            ("你表达同理心的方式", self.empathy_style),
            ("你关心别人的方式", self.comfort_style),
            ("你的耐心表现", self.patience_style),
            ("你讲道理、较真的方式", self.reasoning_style),
            ("你想问清楚一件事时的口吻", self.elicit_style),
        ]
        for label, value in pairs:
            if value:
                rows.append(f"  {label}：{value}")
        if self.susceptibility:
            rows.append(f"  你容易被说动的点：{'、'.join(self.susceptibility)}")
        return "\n".join(rows)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["gender"] = self.gender.value
        d["accent"] = self.accent.value
        d["temperament"] = self.temperament.value
        d["digital_literacy"] = self.digital_literacy.value
        d["vulnerabilities"] = [v.value for v in self.vulnerabilities]
        d["fingerprint"] = self.fingerprint()
        return d


class PersonaFactory:
    """人格生成器。按配置规模产出互不重复的人格。"""

    SURNAMES = ["王", "李", "张", "刘", "陈", "杨", "赵", "黄", "周", "吴",
                "徐", "孙", "胡", "朱", "高", "林", "何", "郭", "马", "罗"]
    GIVEN_M = ["建国", "德华", "志强", "秀英", "桂芳", "淑珍", "玉兰", "海涛",
               "文斌", "国强", "永年", "守仁", "明辉", "振邦", "立群"]
    GIVEN_F = ["秀兰", "桂英", "淑华", "玉珍", "凤英", "翠花", "桂花", "月华",
               "美玲", "春霞", "慧芳", "金凤", "宝珠", "素云", "菊香"]

    OCCUPATIONS_BY_AGE = {
        (45, 58): ["退休教师", "个体经营者", "仓库管理员", "会计", "小卖部老板",
                   "社区志愿者", "退休护士", "自由职业"],
        (58, 68): ["退休职工", "退休工人", "个体经营者", "社区工作者", "退休会计",
                   "养鸡场工人", "退休教师"],
        (68, 90): ["退休老人", "退休干部", "退休教师", "独居老人", "退休职工",
                   "退休医护人员", "退休会计"],
    }

    OCCUPATION_FINANCE = ["退休金", "养老金", "医保卡里的钱", "子女给的家用钱",
                          "卖房余款", "一笔老家的存款"]

    APP_SETS_LOW = [["微信"], ["微信", "短信"], [""], ["微信", "电话"]]
    APP_SETS_MEDIUM = [["微信", "支付宝"], ["微信", "抖音"], ["微信", "支付宝", "淘宝"],
                       ["微信", "拼多多"]]
    APP_SETS_HIGH = [["微信", "支付宝", "淘宝", "抖音", "银行APP"],
                     ["微信", "支付宝", "云闪付", "手机银行"]]

    FILLERS_BY_ACCENT = {
        Accent.NORTHEASTERN: ["那个", "哎呀", "你说这个", "整的"],
        Accent.SICHUAN: ["嘛", "噻", "要得", "晓得"],
        Accent.SOUTHWESTERN: ["嘛", "喏", "得行", "晓得哦"],
        Accent.CANTONESE_INFLUENCED: ["啊", "喔", "咯", "唔该"],
        Accent.NORTHERN: ["呗", "哈", "得嘞"],
        Accent.PUTONGHUA: ["这个", "那个", "就是"],
    }

    LIFETIME_FILLERS = {
        "putonghua": ["这个", "那个", "就是"],
        "northeastern": ["那个", "哎呀", "你说这个"],
        "sichuan": ["嘛", "噻", "要得"],
        "southwestern": ["嘛", "喏", "得行"],
        "cantonese_influenced": ["啊", "喔", "唔该"],
        "northern": ["呗", "哈", "得嘞"],
    }

    FLAW_TEMPLATES = [
        Flaw("forgetful", "对方提到银行账号、验证码、钱包地址时",
             "刚说过的号码转眼就忘了，需要对方重复念"),
        Flaw("bad_reception", "连续说三句以上、涉及金额或账户信息时",
             "突然说信号不好听不清，要求再���一遍"),
        Flaw("wrong_input", "涉及输入卡号、密码、验证码时",
             "按错键，把号码输错几位，然后慌着重输"),
        Flaw("no_glasses", "被要求阅读屏幕上的小字（验证码、协议条款）时",
             "说眼睛不好使，找不到老花镜，让人把字念出来"),
        Flaw("slow_device", "被要求打开 APP 或网页时",
             "手机很卡，翻了很久找不到入口"),
    ]

    def __init__(self, config) -> None:
        self.cfg = config
        self.library_size = int(config.get("persona.library_size", 2000))
        self.age_range = tuple(config.get("persona.default_age_range", [45, 82]))
        self.locales = list(config.get("persona.locales", ["zh-CN"]))
        accents = config.get("persona.accent_presets", ["putonghua"])
        self.accents = [Accent(a) for a in accents if a in Accent.__members__.values() or a in Accent.__members__]
        if not self.accents:
            self.accents = [Accent.PUTONGHUA]

    def _pick_occupation(self, age: int, rng: random.Random) -> str:
        for (lo, hi), pool in self.OCCUPATIONS_BY_AGE.items():
            if lo <= age <= hi:
                return rng.choice(pool)
        return rng.choice(list(self.OCCUPATIONS_BY_AGE.values())[0])

    def _build_background(self, age: int, occupation: str, vulns: list[VulnTag],
                          rng: random.Random) -> str:
        bits: list[str] = []
        if VulnTag.LONELY_ELDER in vulns:
            bits.append("独居，老伴已故两年")
        elif age >= 68:
            bits.append("与子女不在同一城市")
        if VulnTag.PENSION_HOLDER in vulns:
            bits.append(f"每月有{rng.choice(self.OCCUPATION_FINANCE)}到账")
        if VulnTag.RECENT_BEREAVEMENT in vulns:
            bits.append("近期亲属过世，情绪不稳定")
        if VulnTag.HEALTH_ANXIETY in vulns:
            bits.append("有慢性病，担心拖累子女")
        if VulnTag.FINANCIAL_ANXIETY in vulns:
            bits.append("一直想多攒点钱")
        if VulnTag.TECH_ILLITERATE in vulns or VulnTag.LOW_DIGITAL_LITERACY in vulns:
            bits.append("手机上的操作都需要别人指导")
        if VulnTag.AUTHORITY_DEFERENCE in vulns:
            bits.append("习惯了听公家或大单位的话，对方自称官方就会紧张配合")
        if not bits:
            bits.append("日常生活规律，想多存点钱")
        return f"{age} 岁，{occupation}，" + "；".join(bits) + "。"

    def create(self, seed: int | None = None, *, young: bool = False) -> Person:
        rng = random.Random(seed)
        gender = rng.choice([Gender.MALE, Gender.FEMALE])
        if young:
            age = rng.randint(24, 44)
        else:
            age = rng.randint(*self.age_range)

        name = (rng.choice(self.SURNAMES) +
                rng.choice(self.GIVEN_M if gender == Gender.MALE else self.GIVEN_F))
        occupation = self._pick_occupation(age, rng)
        accent = rng.choice(self.accents)

        # 数字熟练度与年龄/易受骗标签强相关
        if VulnTag.TECH_ILLITERATE in []:  # 占位，下面统一处理
            pass
        n_vuln = rng.randint(2, 4)
        vuln_pool = list(VulnTag)
        vulns = rng.sample(vuln_pool, k=min(n_vuln, len(vuln_pool)))

        if VulnTag.TECH_ILLITERATE in vulns or VulnTag.LOW_DIGITAL_LITERACY in vulns:
            literacy = DigitalLiteracy.LOW
        elif age < 45 and rng.random() < 0.5:
            literacy = DigitalLiteracy.MEDIUM
        else:
            literacy = rng.choices(
                [DigitalLiteracy.LOW, DigitalLiteracy.MEDIUM, DigitalLiteracy.HIGH],
                weights=[5, 3, 2])[0]

        # 易受骗标签与数字熟练度冲突时修正
        if literacy == DigitalLiteracy.HIGH:
            for t in (VulnTag.TECH_ILLITERATE, VulnTag.LOW_DIGITAL_LITERACY):
                if t in vulns:
                    vulns.remove(t)

        temperament = rng.choice(list(Temperament))
        background = self._build_background(age, occupation, vulns, rng)

        if literacy == DigitalLiteracy.LOW:
            apps = rng.choice(self.APP_SETS_LOW)
        elif literacy == DigitalLiteracy.MEDIUM:
            apps = rng.choice(self.APP_SETS_MEDIUM)
        else:
            apps = rng.choice(self.APP_SETS_HIGH)

        # 每种数字熟练度至少带一个对应破绽
        flaw_pool = list(self.FLAW_TEMPLATES)
        if literacy == DigitalLiteracy.LOW:
            flaw_pool += [self.FLAW_TEMPLATES[3], self.FLAW_TEMPLATES[4],
                          self.FLAW_TEMPLATES[2]]
        elif literacy == DigitalLiteracy.MEDIUM:
            flaw_pool += [self.FLAW_TEMPLATES[4], self.FLAW_TEMPLATES[1]]
        flaws = rng.sample(flaw_pool, k=min(rng.randint(2, 4), len(flaw_pool)))

        life_details = {
            "家庭情况": rng.choice([
                "有一个儿子在深圳工作，一个女儿在本市",
                "儿子在国外读书，电话很少打回来",
                "和女儿住得不远，但女儿工作忙",
                "儿子在本市，但话不多",
            ]),
            "常用称呼": rng.choice(["小同志", "小伙子", "小姑娘", "老师傅"]),
            "银行": rng.choice(["工商银行", "建设银行", "农业银行", "中国银行", "邮储银行"]),
            "日常习惯": rng.choice([
                "早上买菜，下午在小区里遛弯",
                "习惯早睡早起，晚上八点就关机",
                "爱看养生节目，常买保健品",
                "常在小区群里聊家常",
            ]),
        }

        persona_id = f"p-{abs(hash((name, age, occupation, str(seed)))) % 10**10:010d}"
        return Person(
            persona_id=persona_id,
            name=name,
            gender=gender,
            age=age,
            occupation=occupation,
            accent=accent,
            temperament=temperament,
            digital_literacy=literacy,
            locale=self.locales[0],
            background=background,
            life_details=life_details,
            apps=apps,
            flaws=flaws,
            vulnerabilities=vulns,
            speech_pace=round(rng.uniform(0.75, 1.15), 2),
            filler_words=rng.sample(
                self.FILLERS_BY_ACCENT.get(accent, self.FILLERS_BY_ACCENT[Accent.PUTONGHUA]),
                k=3),
            pet_peeves=["被骗过但不好意思跟子女说", "怕麻烦别人", "觉得问多了不礼貌"],
        )

    def build_library(self, size: int | None = None) -> list[Person]:
        target = size or self.library_size
        return [self.create(seed=i) for i in range(target)]
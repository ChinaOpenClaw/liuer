"""情报提取：规则层。

规则必须在 LLM 之前跑。原因是抽取延迟直接决定情报的时效价值——
如果客户要在钱汇出前封堵，5 秒的 LLM 抽取就毫无意义。

规则层负责高精度类型：账号、卡号、钱包、域名、手机号。
LLM 层负责语义类：冒充的品牌、话术类型、团伙结构。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

# 中国大陆银行卡号：16-19 位
BANK_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){15,18}\d(?!\d)")
# BSB + 账号（澳大利亚格式）
BSB_ACCOUNT = re.compile(r"(?<!\d)\d{6}(?:[-\s]?\d{6,10})(?!\d)")
# 借记卡/信用卡号（13-19 位，带语义提示时置信度更高）
CARD_HINT = re.compile(r"(?:卡号|账号|账户|信用卡|储蓄卡|account|card)\D{0,8}(\d[\d \-]{10,24}\d)", re.I)

# 加密钱包地址
BTC_ADDR = re.compile(r"\b(?:bc1[a-z0-9]{25,62}|[13][a-km-zA-HJ-NP-Z1-9]{25,34})\b")
USDT_TRC20 = re.compile(r"\bT[1-9A-HJ-NP-Za-km-z]{33}\b")
USDT_ERC20 = re.compile(r"\b0x[a-fA-F0-9]{40}\b")
CRYPTO_ANY = re.compile(r"\b(?:bc1[a-z0-9]{20,62}|[13][a-km-zA-HJ-NP-Z1-9]{25,34}"
                        r"|T[1-9A-HJ-NP-Za-km-z]{33}|0x[a-fA-F0-9]{40})\b")

# 钓鱼域名 / 短链
URL = re.compile(r"https?://[^\s，。；！？、\"'）】\]]+", re.I)
DOMAIN = re.compile(r"\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
                    r"(?:com|net|org|io|co|xyz|top|vip|cn|info|biz|cc|ru|su|site|online|link|shop|club)\b",
                    re.I)
SHORTENER = re.compile(r"\b(?:bit\.ly|tinyurl\.com|t\.co|goo\.gl|is\.gd|cutt\.ly|"
                        r"rebrand\.ly|shorturl\.at)\b", re.I)
APK_URL = re.compile(r"(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?")   # 直连 IP，常见于恶意 APK 分发

# 联系方式
PHONE_CN = re.compile(r"(?<!\d)(?:\+?86[-\s]?)?1[3-9]\d{9}(?!\d)")
PHONE_INTL = re.compile(r"(?<!\d)\+\d{1,3}[\s-]?\d{6,14}(?!\d)")
EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# 转账指令线索，说明对方确实进入了收割阶段
HARVEST_TRIGGER = re.compile(
    r"(打款|转账|汇款|尾号|卡号|账户|收款|钱包地址|wallet|account|转至|汇到|打过去)", re.I)

# 常见被冒用品牌
KNOWN_BRANDS = [
    "visa", "mastercard", "paypal", "amazon", "apple", "google", "microsoft",
    "netflix", "facebook", "instagram", "whatsapp", "telegram", "微信", "支付宝",
    "淘宝", "京东", "拼多多", "抖音", "银行", "公安局", "检察院", "法院", "社保局",
    "医保局", "税务局", "证监会", "银保监", "联通", "移动", "电信",
    "顺丰", "圆通", "菜鸟", "抖音小店", "国家反诈中心", "ebay", "steam", "coinbase",
    "binance", "blockchain", "ledger", "western union", "西联",
]

# 诈骗话术类型分类
TACTIC_PATTERNS = {
    "authority_impersonation": re.compile(
        r"(公安局|公安|警察|警官|法院|检察院|反诈中心|银监|证监会|央行|税务|社保|医保|政府|部门|工号|警号)", re.I),
    "tech_support": re.compile(
        r"(技术支持|客服|维修|退款|退货|订单异常|系统故障|服务器|账号异常|解冻|风控)", re.I),
    "investment": re.compile(
        r"(投资|理财|基金|股票|内幕|涨停|收益|回报|带单|操盘|虚拟币|数字货币|挖矿|USDT|合约)", re.I),
    "lottery_prize": re.compile(
        r"(中奖|彩票|奖金|领奖|兑奖|红包|赔付|补偿)", re.I),
    "family_emergency": re.compile(
        r"(我是你儿子|我是你女儿|你孙子|出事了|急救|手术费|学费|绑架|出事|急需)", re.I),
    "romance": re.compile(
        r"(网恋|老伴|孤单|寂寞|陪陪我|喜欢你|心动|结婚)", re.I),
    "job_scam": re.compile(
        r"(招聘|兼职|刷单|返利|做任务|日结|高佣金|带货佣金|橱窗)", re.I),
    "debt_collection": re.compile(
        r"(欠款|催收|网贷|贷款|征信|逾期|还款|花呗|借呗)", re.I),
    "threat_legal": re.compile(
        r"(坐牢|判刑|起诉|传唤|罚款|拘留|通缉|违法|犯罪记录)", re.I),
}


@dataclass
class Finding:
    """一条规则层命中的情报。"""

    kind: str            # bank_account | card_number | crypto_wallet | domain | url | phone | email | ip
    value: str
    confidence: float
    context: str = ""
    source: str = "rule"
    extra: dict = field(default_factory=dict)

    def dedup_key(self) -> str:
        return f"{self.kind}:{self.value}"


def _norm_digits(s: str) -> str:
    return re.sub(r"[\s\-]", "", s)


def extract_rules(text: str, *, has_harvest_trigger: bool | None = None) -> list[Finding]:
    """从一段对话文本中抽取情报。"""
    out: list[Finding] = []
    seen: set[str] = set()

    def push(f: Finding) -> None:
        k = f.dedup_key()
        if k in seen:
            return
        seen.add(k)
        out.append(f)

    if has_harvest_trigger is None:
        has_harvest_trigger = bool(HARVEST_TRIGGER.search(text))

    # 银行账号 / 卡号：带语义提示的置信度更高
    for m in CARD_HINT.finditer(text):
        raw = m.group(1)
        digits = _norm_digits(raw)
        if digits.isdigit() and 13 <= len(digits) <= 19:
            push(Finding("card_number", digits, 0.9 if has_harvest_trigger else 0.7,
                         context=text[max(0, m.start() - 30):m.end() + 30]))

    for m in BANK_CARD.finditer(text):
        digits = _norm_digits(m.group(0))
        if 16 <= len(digits) <= 19 and digits not in {f.value for f in out}:
            push(Finding("bank_account", digits, 0.75 if has_harvest_trigger else 0.55,
                         context=text[max(0, m.start() - 30):m.end() + 30]))

    for m in BSB_ACCOUNT.finditer(text):
        digits = _norm_digits(m.group(0))
        if len(digits) in (10, 12, 16) and digits not in {f.value for f in out}:
            push(Finding("bank_account_bsb", digits, 0.6, context=text[max(0, m.start() - 30):m.end() + 30]))

    for m in CRYPTO_ANY.finditer(text):
        addr = m.group(0)
        kind = "crypto_wallet"
        if addr.startswith("0x"):
            kind = "crypto_wallet_erc20"
        elif addr.startswith("T"):
            kind = "crypto_wallet_trc20"
        elif addr.startswith("bc1"):
            kind = "crypto_wallet_btc"
        push(Finding(kind, addr, 0.92, context=text[max(0, m.start() - 30):m.end() + 30]))

    for m in URL.finditer(text):
        url = m.group(0).rstrip(".,;)")
        push(Finding("url", url, 0.8, context=text[max(0, m.start() - 30):m.end() + 30]))
        d = DOMAIN.search(url)
        if d:
            push(Finding("domain", d.group(0).lower(), 0.85))
        if SHORTENER.search(url):
            push(Finding("domain_shortlink", url, 0.7))

    for m in DOMAIN.finditer(text):
        dom = m.group(0).lower()
        if dom.lower() in {b.lower() for b in KNOWN_BRANDS}:
            continue
        push(Finding("domain", dom, 0.6))

    if re.search(r"\b(apk|安卓安装包|安装包|远程控制|远控|木马|\.apk)\b", text, re.I):
        for m in APK_URL.finditer(text):
            push(Finding("ip_endpoint", m.group(0), 0.65))

    for m in PHONE_CN.finditer(text):
        push(Finding("phone_cn", m.group(0), 0.8, context=text[max(0, m.start() - 30):m.end() + 30]))
    for m in PHONE_INTL.finditer(text):
        push(Finding("phone_intl", m.group(0), 0.6))
    for m in EMAIL.finditer(text):
        push(Finding("email", m.group(0).lower(), 0.75))

    return out


def detect_tactics(text: str) -> list[str]:
    """识别诈骗话术类型。返回命中的分类列表。"""
    hits: list[str] = []
    for name, pat in TACTIC_PATTERNS.items():
        if pat.search(text):
            hits.append(name)
    return hits


def detect_brand(text: str) -> list[str]:
    """识别被冒用的品牌。"""
    low = text.lower()
    return [b for b in KNOWN_BRANDS if b.lower() in low]


def detect_confidence_axis(text: str) -> str:
    """Scam Confidence 分类：deception / authority / distraction。

    对应官网仪表盘中出现的三类情绪操纵手法。
    """
    if re.search(r"(领导|局长|科长|警官|官方|正规|认证|工号|央行|银行)", text):
        return "authority"
    if re.search(r"(对不起|不好意思|遗憾|恐怕|无法|资格|账户异常|需要核实)", text):
        return "distraction"
    return "deception"


def scan(text: str) -> tuple[list[Finding], list[str], list[str], str, bool]:
    """一次性跑完全部规则。返回 (情报, 话术类型, 品牌, 情绪轴, 是否进入收割阶段)"""
    findings = extract_rules(text)
    tactics = detect_tactics(text)
    brands = detect_brand(text)
    axis = detect_confidence_axis(text)
    harvesting = bool(HARVEST_TRIGGER.search(text))
    return findings, tactics, brands, axis, harvesting
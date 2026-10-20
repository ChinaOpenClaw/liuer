"""接入侧风险检测。

与 Apate 的架构一致：检测由接入方（运营商/银行）完成，
LiuEr 只承接已判定为高风险的会话。

但本地部署场景下往往没有上游风控，因此这里提供一个轻量本地检测器：
基于规则打分，够单机使用，也能对接外部风控分数。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..intel.rules import TACTIC_PATTERNS, KNOWN_BRANDS


@dataclass
class RiskScore:
    score: float
    signals: list[str]

    @property
    def should_divert(self) -> bool:
        return self.score >= 0.6


class LocalRiskDetector:
    """规则打分器。不依赖模型，毫秒级。"""

    SUSPICIOUS_PATTERNS = [
        # 执法诱导是最强信号，命中即应转接
        (r"(安全账户|资金清查|银监|公安|反诈中心|案件|通缉|洗钱|涉案|涉嫌)", 0.45, "执法诱导"),
        (r"(退款|理赔|赔付|补偿|退税|补贴)", 0.2, "退款诱导"),
        (r"(投资|理财|内幕|带单|操盘|稳定收益|保本|高回报)", 0.35, "投资诱导"),
        (r"(账户异常|密码重置|冻结|解冻|验证身份|银行卡异常)", 0.25, "账户异常"),
        (r"(屏幕共享|远程控制|安装软件|下载APP|打开链接|安装包)", 0.3, "远程操控"),
        (r"(验证码|动态密码|短信验证码|密码)", 0.2, "索要凭据"),
        (r"(客服|售后|官方|专员|经理).{0,6}(退款|退货|订单)", 0.25, "冒充客服"),
    ]

    def score_text(self, text: str) -> RiskScore:
        signals: list[str] = []
        score = 0.0
        for pat, weight, label in self.SUSPICIOUS_PATTERNS:
            if re.search(pat, text, re.I):
                score += weight
                signals.append(label)

        # 话术类型命中加权
        for name, pat in TACTIC_PATTERNS.items():
            if pat.search(text):
                score += 0.15
                signals.append(f"话术:{name}")

        # 品牌冒充
        low = text.lower()
        for b in KNOWN_BRANDS:
            if b.lower() in low:
                score += 0.1
                signals.append(f"品牌:{b}")
                break

        # 号码类型：陌生/境外/虚拟号段
        if re.search(r"\+?(?:44|61|1|81|65|66|91)\d{8,12}", text.replace(" ", "")):
            score += 0.1
            signals.append("境外号码")

        return RiskScore(score=min(score, 1.0), signals=signals)

    def score_call(self, caller_number: str, transcript: str) -> RiskScore:
        rs = self.score_text(transcript)
        signals = list(rs.signals)
        score = rs.score
        # 未保存号码 + 疑似回拨线路
        if caller_number and re.search(r"^(0{2,}|\+)", caller_number):
            score = min(1.0, score + 0.15)
            signals.append("非常规来电格式")
        return RiskScore(score=score, signals=signals)
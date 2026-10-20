"""人格档案加载。

档案是 YAML 格式的单个人格定义，用于"以某个真实原型建模"的场景——
比如为家中易受骗的长辈建一个数字诱饵。

隐私约束：档案可能含真实个人背景。加载器会读取 profile.based_on_real_person
标记，为真时在日志中提示不得用于公开演示。除 template.yaml 外的档案
默认被 .gitignore 排除。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .model import Accent, DigitalLiteracy, Flaw, Gender, Person, Temperament

log = logging.getLogger(__name__)

_ENUM_MAP = {
    "gender": Gender,
    "accent": Accent,
    "temperament": Temperament,
    "digital_literacy": DigitalLiteracy,
}


@dataclass
class PersonaProfile:
    """从 YAML 载入的人格档案。"""

    data: dict[str, Any]
    path: Path
    opener: str = ""
    tactics_note: str = ""
    real_person: bool = False

    @property
    def codename(self) -> str:
        return str(self.data.get("codename", self.path.stem))

    def to_person(self) -> Person:
        """转成 Person 实例。缺失字段回落到默认值。"""
        import hashlib

        d = self.data
        seed = hashlib.sha256(
            f"{d.get('name','')}{d.get('birth_year','')}{d.get('occupation','')}".encode()
        ).hexdigest()[:8]

        flaws = [
            Flaw(kind=f.get("kind", "custom"),
                 trigger=f.get("trigger", ""),
                 expression=f.get("expression", ""))
            for f in (d.get("flaws") or [])
        ]

        def enum_of(key: str, default):
            raw = d.get(key)
            if raw is None:
                return default
            enum_cls = _ENUM_MAP[key]
            try:
                return enum_cls(raw)
            except ValueError:
                return default

        background = (
            f"{d.get('age','')} 岁，{d.get('occupation','')}，"
            f"{d.get('birth_year','')} 年生。{d.get('context_profile','').strip()}"
        ).strip()

        return Person(
            persona_id=f"prof-{seed}",
            name=str(d.get("name", "未命名")),
            gender=enum_of("gender", Gender.FEMALE),
            age=int(d.get("age", 60)),
            occupation=str(d.get("occupation", "")),
            accent=enum_of("accent", Accent.PUTONGHUA),
            temperament=enum_of("temperament", Temperament.POLITE),
            digital_literacy=enum_of("digital_literacy", DigitalLiteracy.MEDIUM),
            locale=str(d.get("locale", "zh-CN")),
            background=background,
            life_details=dict(d.get("life_details") or {}),
            apps=list(d.get("apps") or ["微信"]),
            flaws=flaws,
            vulnerabilities=[],
            speech_pace=float(d.get("speech_pace", 0.9)),
            filler_words=list(d.get("filler_words") or []),
            pet_peeves=list(d.get("pet_peeves") or ["怕麻烦别人"]),
            context_profile=str(d.get("context_profile", "")),
            personal_situation=str(d.get("personal_situation", "")),
            reflexive_personality=str(d.get("reflexive_personality", "")),
            empathy_style=str(d.get("empathy_style", "")),
            comfort_style=str(d.get("comfort_style", "")),
            patience_style=str(d.get("patience_style", "")),
            reasoning_style=str(d.get("reasoning_style", "")),
            elicit_style=str(d.get("elicit_style", "")),
            susceptibility=list(d.get("susceptibility") or []),
        )


def load_profile(path: str | Path) -> PersonaProfile:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"人格档案不存在: {p}")
    with open(p, "r", encoding="utf-8") as fp:
        raw = yaml.safe_load(fp) or {}
    data = raw.get("profile") or raw
    prof = PersonaProfile(
        data=data, path=p,
        opener=str(data.get("opener", "")),
        tactics_note=str(data.get("tactics_note", "")),
        real_person=bool(data.get("based_on_real_person", False)))
    if prof.real_person:
        log.warning("档案「%s」基于真实人物建模，禁止用于公开演示或对外分享",
                    prof.codename)
    return prof


def discover_profiles(dir_path: str | Path) -> list[Path]:
    """列出目录下所有可用于公开的档案（排除基于真实人物的）。"""
    d = Path(dir_path)
    if not d.exists():
        return []
    out: list[Path] = []
    for p in sorted(d.glob("*.yaml")):
        try:
            with open(p, "r", encoding="utf-8") as fp:
                data = (yaml.safe_load(fp) or {}).get("profile", {})
        except Exception:  # noqa: BLE001
            continue
        if data.get("based_on_real_person"):
            continue
        if data.get("codename") == "示例档案":
            continue
        out.append(p)
    return out
"""情报管道：规则层 + LLM 层 + 推送。

延迟是情报的生命线。管道按顺序设计：
规则层毫秒级，先跑，先推；LLM 层慢，只补充规则拿不到的语义信息。
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from ..intel.rules import Finding, detect_brand, detect_confidence_axis, detect_tactics, extract_rules
from ..intel.store import IntelStore

log = logging.getLogger(__name__)

LLM_EXTRACT_PROMPT = """从下面这段诈骗对话中提取结构化信息。只输出 JSON，不要任何解释文字。

对话内容：
\"\"\"
{text}
\"\"\"

输出格式：
{{"brand_impersonated": ["被冒用的品牌或机构，没有就空数组"],
 "tactic_type": ["话术类型，从这些里选：authority_impersonation, tech_support, investment, lottery_prize, family_emergency, romance, job_scam, debt_collection, threat_legal，没有就空数组"],
 "requested_account": "对方索要的收款账户或钱包地址，没有就空字符串",
 "requested_contact": "对方提供的其他联系电话或邮箱，没有就空字符串",
 "social_engineering_axis": "从 deception / authority / distraction 里选一个",
 "org_structure": "推断的组织结构描述，没有就空字符串",
 "next_lure": "对方下一步打算怎么推进，没有就空字符串"}}"""


class IntelPipeline:
    def __init__(self, config, store: IntelStore, llm) -> None:
        self.cfg = config
        self.store = store
        self.llm = llm
        self.min_confidence = float(config.get("intel.min_confidence", 0.55))
        self.push_channel = str(config.get("intel.push_channel", "console"))
        self.webhook_url = str(config.get("intel.webhook_url", ""))
        self.push_timeout_ms = int(config.get("intel.push_timeout_ms", 3000))
        # 抽取超时。9B 模型生成的速度约 40-70 tok/s，256 token 需数秒
        self.extract_timeout = float(config.get("llm.extract_timeout", 12))
        self._llm_enabled = bool(config.get("intel.enable_llm_extract", True))

    async def process_turn(self, text: str, *, client_id: str = "default",
                           session_id: str = "") -> dict[str, Any]:
        """处理一轮对话文本。返回本次推送的情报与语义标签。"""
        if not text or not text.strip():
            return {"findings": [], "tags": {}, "pushed": 0}

        # 1) 规则层，毫秒级
        findings = extract_rules(text)
        tactics = detect_tactics(text)
        brands = detect_brand(text)
        axis = detect_confidence_axis(text)

        # 2) LLM 层，仅对整段对话跑一次，带短超时
        llm_tags: dict[str, Any] = {}
        if self._llm_enabled:
            llm_tags = await self._llm_extract(text)

        if llm_tags:
            brands = list({*brands, *llm_tags.get("brand_impersonated", [])})[:6]
            tactics = list({*tactics, *llm_tags.get("tactic_type", [])})[:6]
            axis = llm_tags.get("social_engineering_axis") or axis

            # LLM 补出的账户没有规则层的正则保证，置信度打折。
            # 且必须先去重——规则层已抓到的值不要重复入库。
            existing = {re.sub(r"[\s\-]", "", f.value).lower() for f in findings}
            for key, kind in (("requested_account", "llm_reported_account"),
                              ("requested_contact", "llm_reported_contact")):
                val = (llm_tags.get(key) or "").strip()
                if not val or len(val) < 6:
                    continue
                norm = re.sub(r"[\s\-]", "", val).lower()
                if norm in existing:
                    continue
                findings.append(Finding(kind=kind, value=val, confidence=0.5,
                                        context=text[:300], source="llm"))
                existing.add(norm)

        # 3) 入库
        pushed: list[dict[str, Any]] = []
        for f in findings:
            rec, created = self.store.upsert(
                f, client_id=client_id, session_id=session_id,
                tactic=",".join(tactics), brand=",".join(brands), axis=axis)
            if rec.confidence >= self.min_confidence:
                pushed.append(rec.to_dict())

        if pushed:
            self.store.incr("intel_new", len(pushed))
            self.store.incr("intel_total")
            await self._push(pushed, session_id=session_id)

        self.store.incr("turns_processed")
        return {
            "findings": pushed,
            "tags": {"tactics": tactics, "brands": brands, "axis": axis, **llm_tags},
        }

    async def _llm_extract(self, text: str) -> dict[str, Any]:
        try:
            prompt = LLM_EXTRACT_PROMPT.format(text=text[:4000])
            raw = await asyncio.wait_for(
                self.llm.complete(prompt, temperature=0.1, max_tokens=256),
                timeout=self.extract_timeout)
        except asyncio.TimeoutError:
            log.warning("LLM 抽取超时，使用纯规则结果")
            return {}
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM 抽取失败: %s", exc)
            return {}
        return self._parse_json(raw)

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        if not raw:
            return {}
        raw = raw.strip()
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            data = json.loads(raw[start:end + 1])
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}

    async def _push(self, records: list[dict[str, Any]], *, session_id: str) -> None:
        if self.push_channel == "console":
            for r in records:
                log.info("[情报] %s | %s | 置信度 %.2f | 会话 %s",
                         r["kind"], r["value"], r["confidence"], session_id[:8])
        elif self.push_channel == "webhook" and self.webhook_url:
            try:
                await asyncio.wait_for(
                    self.llm.client.post(self.webhook_url,
                                         json={"records": records, "session_id": session_id}),
                    timeout=self.push_timeout_ms / 1000.0)
            except Exception as exc:  # noqa: BLE001
                log.warning("情报 webhook 推送失败: %s", exc)
        elif self.push_channel == "file":
            from pathlib import Path
            from ..config import PROJECT_ROOT
            out = PROJECT_ROOT / "data" / "intel_outbox.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "a", encoding="utf-8") as fp:
                for r in records:
                    fp.write(json.dumps(r, ensure_ascii=False) + "\n")
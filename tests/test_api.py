"""API 接口测试。用 FastAPI TestClient，不依赖外部端口与代理。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from liuer.api import create_server  # noqa: E402

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
    server = create_server()
    client = TestClient(server)

    print("\n=== 1. 健康检查 ===")
    r = client.get("/api/health")
    check("health 返回 200", r.status_code == 200, str(r.status_code))
    h = r.json()
    check("含部署模式 local", h.get("deployment", {}).get("mode") == "local")
    check("人格库已就绪", h.get("persona", {}).get("library_size", 0) > 0)
    check("返回 LLM 状态", "llm" in h)
    check("返回语音状态", "voice" in h)

    print("\n=== 2. 风险检测 ===")
    r = client.post("/api/detect", json={
        "text": "您好，这里是公安局反诈中心，您的账户涉嫌洗钱，请配合调查"})
    d = r.json()
    check("检测接口可用", r.status_code == 200)
    check("识别为高风险", d.get("should_divert") is True, str(d))
    check("给出信号", len(d.get("signals", [])) > 0, str(d.get("signals")))

    d2 = client.post("/api/detect", json={"text": "你好，今天天气不错"}).json()
    check("正常对话不转接", d2.get("should_divert") is False, str(d2))

    print("\n=== 3. 人格预览 ===")
    p = client.get("/api/persona/preview?count=3").json()
    check("返回 3 个人格", p.get("count") == 3, str(p.get("count")))
    first = p["personas"][0]
    check("含姓名年龄口音", all(k in first for k in ("name", "age", "accent")))
    check("含破绽列表", len(first.get("flaws", [])) >= 2)
    check("指纹唯一", len({x["fingerprint"] for x in p["personas"]}) == 3)
    check("不同人格不同姓名", len({x["name"] for x in p["personas"]}) > 1)

    print("\n=== 4. 对话链路 ===")
    r = client.post("/api/chat", json={
        "peer": "+8613800138000",
        "text": "请把款转到安全账户，账号 6222021234567890123",
        "channel": "web"})
    check("对话接口返回 200", r.status_code == 200, r.text[:300])
    d = r.json()
    sid = d.get("session_id")
    check("分配了会话", bool(sid))
    check("返回人格信息", d.get("persona") is not None)
    check("阶段已推进", d.get("stage") is not None)

    r2 = client.post("/api/chat", json={
        "peer": "+8613800138000", "text": "USDT 地址是 TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t",
        "session_id": sid, "channel": "web"})
    check("同会话可续接", r2.json().get("session_id") == sid)
    check("复用同一人格", r2.json().get("persona", {}).get("name") ==
          d.get("persona", {}).get("name"))

    print("\n=== 5. 任务下发 ===")
    r = client.post("/api/mission", json={
        "peer": "tester", "intent": "harvest_crypto_wallet",
        "target": "某交易所", "notes": "优先套 USDT TRC20 地址"})
    check("任务接口可用", r.status_code == 200, r.text[:200])
    check("返回任务 ID", bool(r.json().get("mission_id")))
    check("可列出任务", len(client.get("/api/mission").json()["missions"]) > 0)

    r = client.post("/api/mission", json={"peer": "x", "intent": "不存在的意图"})
    check("非法 intent 被拒", r.status_code == 400, str(r.status_code))

    print("\n=== 6. 情报链路 ===")
    r = client.get("/api/intel?limit=50")
    check("情报接口可用", r.status_code == 200)
    records = r.json()["records"]
    check("已抓到情报", len(records) > 0, str(len(records)))
    kinds = {x["kind"] for x in records}
    check("含卡号", "card_number" in kinds, str(kinds))
    check("含钱包地址", any(k.startswith("crypto_wallet") for k in kinds), str(kinds))
    if records:
        check("值已解密", len(records[0]["value"]) > 5)
        check("有置信度", 0 < records[0]["confidence"] <= 1)

    st = client.get("/api/intel/stats").json()
    check("统计接口可用", st.get("total", 0) > 0, str(st.get("total")))
    check("按类型统计", len(st.get("by_kind", {})) > 0)

    print("\n=== 7. 会话查询 ===")
    r = client.get("/api/sessions?limit=10")
    check("会话列表可用", r.status_code == 200 and len(r.json()["sessions"]) > 0)
    detail = client.get(f"/api/sessions/{sid}")
    check("会话详情可用", detail.status_code == 200)
    check("详情含转录", len(detail.json().get("transcript", [])) > 0)
    check("不存在的会话返回 404", client.get("/api/sessions/nope").status_code == 404)

    print("\n=== 8. 语音通道（缺依赖时应降级） ===")
    vh = client.get("/api/health").json().get("voice", {})
    check("给出降级标记", "degraded" in vh)
    check("给出降级说明", bool(vh.get("note")))
    r = client.post("/api/voice", json={"peer": "voice-test", "text": "测试语音通道"})
    check("语音接口可用", r.status_code == 200, r.text[:200])
    check("返回语音指标", "voice_metrics" in r.json())
    vm = r.json()["voice_metrics"]
    check("指标含延迟", "avg_latency_ms" in vm)
    check("指标含组件名", "components" in vm)

    print("\n=== 9. 蜜罐 ===")
    hn = client.get("/api/honeypot").json()
    check("蜜罐接口可用", "enabled" in hn)
    check("默认不启用", hn.get("enabled") is False)

    print("\n=== 10. 看板 ===")
    r = client.get("/")
    check("看板可访问", r.status_code == 200 and "LiuEr" in r.text)

    print(f"\n{'='*46}\n  通过 {PASS}  失败 {FAIL}\n{'='*46}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
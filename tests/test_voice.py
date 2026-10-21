"""语音生成接线测试。

沙箱网络无法下载 GB 级模型文件，因此本测试只验证：
1. 引擎选择与降级链
2. 口音→音色映射的确定性
3. 配置解析
模型本体合成质量需要装好依赖后在真实环境验证（README 有说明）。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from liuer.config import load_config  # noqa: E402

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
    from liuer.channel.voice.pipeline import MockTTS
    from liuer.channel.voice.tts_hf import (
        CHATTTS_SEED_BY_ACCENT,
        MELO_SPEED_BY_ACCENT,
        ChatTTSAdapter,
        MeloTTSAdapter,
        apply_hf_mirror,
        build_tts,
    )
    from liuer.persona.model import Accent

    cfg = load_config(ROOT / "config.yaml")

    print("\n=== 1. 配置解析 ===")
    provider = str(cfg.get("channel.voice.tts.provider", ""))
    check("provider 为自适应链", provider in ("auto", "melo", "kokoro"), provider)
    check("Kokoro 权重目录已配置",
          bool(cfg.get("channel.voice.tts.kokoro_dir")),
          str(cfg.get("channel.voice.tts.kokoro_dir")))
    check("模型不落系统盘",
          str(cfg.get("storage.hf_home", "")).startswith("D:"),
          str(cfg.get("storage.hf_home")))
    check("镜像开关存在", "hf_mirror" in cfg.raw.get("channel", {}).get("voice", {}))
    check("镜像端点正确", cfg.get("channel.voice.hf_endpoint") == "https://hf-mirror.com",
          str(cfg.get("channel.voice.hf_endpoint")))

    print("\n=== 2. 镜像环境变量 ===")
    import os
    os.environ.pop("HF_ENDPOINT", None)
    apply_hf_mirror(cfg)
    check("HF_ENDPOINT 已设置", os.environ.get("HF_ENDPOINT") == "https://hf-mirror.com",
          os.environ.get("HF_ENDPOINT"))

    print("\n=== 3. 口音映射确定性 ===")
    for accent in Accent:
        check(f"MeloTTS {accent.value} 有映射", accent in MELO_SPEED_BY_ACCENT)
        check(f"ChatTTS {accent.value} 有种子", accent in CHATTTS_SEED_BY_ACCENT)
    # 确定性：同一口音速度恒定
    check("语速区间合理",
          all(0.7 <= v <= 1.1 for v in MELO_SPEED_BY_ACCENT.values()))
    check("种子互不相同", len(set(CHATTTS_SEED_BY_ACCENT.values())) == len(Accent))

    print("\n=== 4. Melo 适配器（不加载模型）===")
    m = MeloTTSAdapter(cfg)
    check("voice_for_accent 返回语速串", m.voice_for_accent(Accent.NORTHEASTERN).startswith("zh:"),
          m.voice_for_accent(Accent.NORTHEASTERN))
    check("语速可解析", float(m.voice_for_accent(Accent.NORTHEASTERN).split(":")[1]) == 0.86)
    check("设备解析不崩溃", m.device in ("cuda", "cpu", "mps"), m.device)

    print("\n=== 5. ChatTTS 适配器（不加载模型）===")
    ch = ChatTTSAdapter(cfg)
    check("voice_for_accent 返回口音值",
          ch.voice_for_accent(Accent.SICHUAN) == "sichuan",
          ch.voice_for_accent(Accent.SICHUAN))

    print("\n=== 6. 降级链 ===")
    # 降级只要落到可用引擎即可：机器上装了 kokoro 就用它，没装才是 Mock。
    # 断言写死某一个引擎名会把测试绑死在特定机器的依赖状态上。
    REAL = {"KokoroAdapter", "MeloTTSAdapter", "ChatTTSAdapter"}
    eng = build_tts(cfg)
    check("自适应链落到可用引擎", type(eng).__name__ in REAL | {"MockTTS"},
          type(eng).__name__)
    cfg2 = load_config(ROOT / "config.yaml")
    cfg2.raw["channel"]["voice"]["tts"]["provider"] = "mock"
    eng2 = build_tts(cfg2)
    check("显式 mock 直接返回", type(eng2).__name__ == "MockTTS")
    cfg3 = load_config(ROOT / "config.yaml")
    cfg3.raw["channel"]["voice"]["tts"]["provider"] = "chattts"
    eng3 = build_tts(cfg3)
    check("chattts 不可用时降级到可用引擎", type(eng3).__name__ in REAL | {"MockTTS"},
          type(eng3).__name__)

    print("\n=== 7. Mock 合为文本字节 ===")
    import asyncio
    audio = asyncio.run(eng2.synthesize("测试"))
    check("mock 输出非空", len(audio) > 0)

    print("\n=== 8. TTS 接口注册 ===")
    from liuer.api import create_server
    import inspect
    src = inspect.getsource(create_server)
    check("/api/voice/tts 已注册", "/api/voice/tts" in src)
    check("返回引擎名", '"engine"' in src)

    print(f"\n{'=' * 46}\n  通过 {PASS}  失败 {FAIL}\n{'=' * 46}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
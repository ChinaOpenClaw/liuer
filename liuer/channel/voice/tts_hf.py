"""HuggingFace 语音合成适配器。

选型依据（8GB 显存、LLM 已占用约 3GB 的前提）：

- MeloTTS：CPU 实时推理，零显存，MIT 许可。与本地 LLM 互不抢资源，
  作为默认方案。
- ChatTTS：2-3GB 显存，对话感最强——自带语气词、停顿、笑声，
  最贴合老年人格。但 CC-BY-NC 许可仅限个人/研究用途，
  且权重带刻意注入的噪声，不适合商用产品。
- CosyVoice2-0.5B 需 6-8GB 显存，与 LLM 无法共存于 8GB 卡，不纳入。

模型获取：首次运行时由各自库从 HuggingFace 自动下载。
国内网络通过 HF_ENDPOINT 镜像加速（config: channel.voice.hf_mirror）。

诚实声明：本模块在沙箱内只做了接线与降级逻辑的单元测试，
模型本体的合成质量未实测（沙箱网络无法下载 GB 级模型文件）。
"""
from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Any

from .pipeline import MockTTS, TTSEngine
from ...persona.model import Accent

log = logging.getLogger(__name__)

# 各口音对应的音色参数。
# MeloTTS 内置中文音色只有一个 ZH，用语速区分口音气质；
# ChatTTS 用确定性随机种子生成说话人向量。
MELO_SPEED_BY_ACCENT: dict[Accent, float] = {
    Accent.PUTONGHUA: 0.92,
    Accent.NORTHERN: 0.90,
    Accent.NORTHEASTERN: 0.86,
    Accent.SICHUAN: 0.94,
    Accent.SOUTHWESTERN: 0.94,
    Accent.CANTONESE_INFLUENCED: 0.90,
}

CHATTTS_SEED_BY_ACCENT: dict[Accent, int] = {
    Accent.PUTONGHUA: 2027,
    Accent.NORTHERN: 411,
    Accent.NORTHEASTERN: 1966,
    Accent.SICHUAN: 8080,
    Accent.SOUTHWESTERN: 512,
    Accent.CANTONESE_INFLUENCED: 2249,
}

# 老年说话风格：多停顿、口语化、少笑
CHATTTS_REFINE_PROMPT = "[oral_2][laugh_0][break_6]"


def apply_hf_mirror(config) -> None:
    """国内镜像加速。必须在模型库 import 之前设置。"""
    enabled = bool(config.get("channel.voice.hf_mirror", True))
    endpoint = str(config.get("channel.voice.hf_endpoint", "https://hf-mirror.com"))
    if enabled and endpoint:
        os.environ.setdefault("HF_ENDPOINT", endpoint)
        log.info("HuggingFace 镜像已启用: %s", endpoint)


def _wav_bytes(samples, sample_rate: int) -> bytes:
    """张量/数组 → WAV 字节流。"""
    import numpy as np
    import soundfile as sf

    if hasattr(samples, "detach"):
        samples = samples.detach().cpu().numpy()
    arr = np.asarray(samples, dtype="float32")
    if arr.ndim > 1:
        arr = arr.squeeze()
    buf = io.BytesIO()
    sf.write(buf, arr, sample_rate, format="WAV")
    return buf.getvalue()


def _pick_device(config) -> str:
    requested = str(config.get("channel.voice.tts.device", "auto")).lower()
    if requested != "auto":
        return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class MeloTTSAdapter(TTSEngine):
    """MeloTTS：CPU 实时，MIT 许可，零显存占用。

    安装（需先装 espeak-ng 并配置 PATH）：
        pip install git+https://github.com/myshell-ai/MeloTTS.git
    模型约几百 MB，首次调用时自动从 HuggingFace 下载。
    """

    name = "melo"

    def __init__(self, config) -> None:
        self.config = config
        self.device = _pick_device(config)
        self._model = None
        self._speaker_id: int | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        apply_hf_mirror(self.config)
        from melo.api import TTS  # 延迟导入，缺依赖时走降级

        self._model = TTS(language="ZH", device=self.device)
        spk2id = self._model.hps.data.spk2id
        self._speaker_id = spk2id.get("ZH", next(iter(spk2id.values()), 0))
        log.info("MeloTTS 已加载 device=%s", self.device)

    def voice_for_accent(self, accent: Accent) -> str:
        # MeloTTS 中文音色单一，用语速体现口音气质差异
        return f"zh:{MELO_SPEED_BY_ACCENT.get(accent, 0.92):.2f}"

    async def synthesize(self, text: str, *, voice: str | None = None) -> bytes:
        import tempfile

        self._ensure_loaded()
        speed = 0.92
        if voice and voice.startswith("zh:"):
            try:
                speed = float(voice.split(":", 1)[1])
            except ValueError:
                pass
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tmp = Path(tf.name)
        try:
            self._model.tts_to_file(text, self._speaker_id, str(tmp), speed=speed)
            return tmp.read_bytes()
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass


class ChatTTSAdapter(TTSEngine):
    """ChatTTS：对话感最强，显存 2-3GB。

    许可证 CC-BY-NC：仅限个人/研究用途，商用前需自行获得授权。
    安装：
        pip install chattts
    模型约 1GB，首次运行时自动从 HuggingFace 下载。
    """

    name = "chattts"

    def __init__(self, config) -> None:
        self.config = config
        self.device = _pick_device(config)
        self._chat = None
        self._speakers: dict[Accent, Any] = {}

    def _ensure_loaded(self) -> None:
        if self._chat is not None:
            return
        apply_hf_mirror(self.config)
        import torch
        import ChatTTS

        chat = ChatTTS.Chat()
        chat.load(source="huggingface", compile=False, device=self.device)
        self._chat = chat
        log.info("ChatTTS 已加载 device=%s", self.device)

    def _speaker_for(self, accent: Accent) -> Any:
        import torch

        if accent not in self._speakers:
            seed = CHATTTS_SEED_BY_ACCENT.get(accent, 2027)
            torch.manual_seed(seed)
            self._speakers[accent] = self._chat.sample_random_speaker()
        return self._speakers[accent]

    def voice_for_accent(self, accent: Accent) -> str:
        # 返回口音标识即可，真正的音色向量在 synthesize 时按口音取
        return accent.value

    async def synthesize(self, text: str, *, voice: str | None = None) -> bytes:
        self._ensure_loaded()
        import ChatTTS

        accent = Accent(voice) if voice in Accent.__members__.values() \
            or (voice and voice in [a.value for a in Accent]) else Accent.PUTONGHUA
        params_refine = ChatTTS.Chat.RefineTextPrompt(prompt=CHATTTS_REFINE_PROMPT)
        params_infer = ChatTTS.Chat.InferCodePrompt(
            spk_emb=self._speaker_for(accent),
            temperature=0.3, top_P=0.7, top_K=20,
        )
        wavs = self._chat.synthesize(
            [text], params_refine_text=params_refine, params_infer_code=params_infer)
        return _wav_bytes(wavs[0], 24000)


def build_tts(config) -> TTSEngine:
    """按配置选择 TTS 引擎，缺依赖时逐级降级，保证语音通道可用。"""
    provider = str(config.get("channel.voice.tts.provider", "melo")).lower()
    chain: list[str]
    if provider == "mock":
        chain = ["mock"]
    elif provider == "edge":
        chain = ["edge", "mock"]
    elif provider == "chattts":
        chain = ["chattts", "melo", "mock"]
    else:  # melo 为默认
        chain = ["melo", "chattts" if provider == "auto" else "mock"]

    tried: list[str] = []
    for name in chain:
        tried.append(name)
        if name == "mock":
            log.info("TTS 降级为 Mock（无真实合成）")
            return MockTTS(str(config.get("channel.voice.tts.voice", "zh-CN-XiaoxiaoNeural")))
        try:
            if name == "melo":
                eng = MeloTTSAdapter(config)
                eng._ensure_loaded()
            elif name == "chattts":
                eng = ChatTTSAdapter(config)
                eng._ensure_loaded()
            elif name == "edge":
                # edge-tts 走微软在线接口，离线部署不可用，直接降级
                log.warning("edge-tts 需要访问微软在线接口，离线部署不可用，降级")
                continue
            log.info("TTS 引擎: %s", name)
            return eng
        except ImportError as exc:
            log.warning("TTS %s 依赖缺失: %s", name, exc)
        except Exception as exc:  # noqa: BLE001
            log.warning("TTS %s 初始化失败: %s", name, exc)
    log.error("所有 TTS 引擎不可用（尝试过 %s），使用 Mock", tried)
    return MockTTS("zh-CN-XiaoxiaoNeural")
"""本地语音识别（faster-whisper）。

在 GPU 上运行 Whisper，把音频片段转为文字。
模型首次使用会自动下载到本地缓存。
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("asr")


def _register_cuda_dlls() -> None:
    """把 pip 安装的 NVIDIA CUDA 运行库加入 DLL 搜索路径。

    Windows 上 ctranslate2 需要 cublas64_12.dll / cudnn*.dll，
    但驱动不自带。安装了 nvidia-cublas-cu12 / nvidia-cudnn-cu12 后，
    这些 DLL 位于 site-packages/nvidia/*/bin，需手动加入搜索路径。
    """
    if os.name != "nt":
        return
    try:
        import site

        roots = []
        for sp in site.getsitepackages() + [site.getusersitepackages()]:
            nv = Path(sp) / "nvidia"
            if nv.is_dir():
                roots.append(nv)
        for nv in roots:
            for sub in nv.iterdir():
                bin_dir = sub / "bin"
                if bin_dir.is_dir():
                    try:
                        os.add_dll_directory(str(bin_dir))
                        log.debug("已注册 CUDA DLL 路径: %s", bin_dir)
                    except Exception:
                        pass
                    # 同时加入 PATH，兼容旧式加载
                    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
    except Exception as e:  # pragma: no cover
        log.debug("注册 CUDA DLL 失败: %s", e)


_register_cuda_dlls()


@dataclass
class ASRResult:
    text: str
    language: str
    probability: float
    duration: float


class WhisperASR:
    def __init__(
        self,
        model: str = "small",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "",
        beam_size: int = 1,
        initial_prompt: str = "",
    ):
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.language = language or None
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt or None
        self._model = None
        self._lock = threading.Lock()
        self._loading = False
        self._ctx = None  # 可复用的 prompt tokens（预热后）

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """加载模型（可能耗时数十秒，首次会下载）。"""
        if self._model is not None or self._loading:
            return
        self._loading = True
        try:
            from faster_whisper import WhisperModel

            log.info("加载 Whisper 模型 %s (%s/%s)...", self.model_name, self.device, self.compute_type)
            try:
                self._model = WhisperModel(
                    self.model_name,
                    device=self.device,
                    compute_type=self.compute_type,
                    cpu_threads=4,
                    num_workers=1,
                )
            except Exception as e:
                log.warning("GPU 加载失败(%s)，回退 CPU int8", e)
                self.device = "cpu"
                self.compute_type = "int8"
                self._model = WhisperModel(
                    self.model_name, device="cpu", compute_type="int8", cpu_threads=4
                )
            log.info("Whisper 模型加载完成")
        finally:
            self._loading = False

    def transcribe(
        self, audio: np.ndarray, is_partial: bool = False, prompt: str | None = None
    ) -> Optional[ASRResult]:
        """识别音频。

        is_partial=True 时是增量识别（赶时间，用最贪心的参数、不做语言检测以省时间），
        否则按正式参数识别。
        """
        if self._model is None:
            self.load()
        if self._model is None:
            return None
        audio = np.ascontiguousarray(audio.astype(np.float32))
        with self._lock:
            segments, info = self._model.transcribe(
                audio,
                language=self.language,
                beam_size=1 if is_partial else self.beam_size,
                best_of=1,
                temperature=0.0 if is_partial else [0.0, 0.2, 0.4],
                vad_filter=False,
                initial_prompt=prompt or self.initial_prompt,
                condition_on_previous_text=False,
                without_timestamps=True,
                word_timestamps=False,
            )
            text = "".join(seg.text for seg in segments).strip()
        if not text:
            return None
        return ASRResult(
            text=text,
            language=getattr(info, "language", "") or "",
            probability=float(getattr(info, "language_probability", 0.0) or 0.0),
            duration=len(audio) / 16000,
        )

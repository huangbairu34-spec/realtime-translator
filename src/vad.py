"""语音活动检测与分段。

把连续的音频流切成"一句话"的片段，交给识别引擎。
支持两种模式：
  - energy: 纯能量阈值，零依赖、够用、极快。
  - webrtc: 使用 webrtcvad（若安装）更精准。
两种模式都实现"连续静音 N ms 判定句末 + 最长时长强制切分"。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Segment:
    audio: np.ndarray   # float32 单声道
    duration: float     # 秒


class Segmenter:
    def __init__(
        self,
        sample_rate: int = 16000,
        silence_ms: int = 600,
        min_speech_ms: int = 400,
        max_segment_s: float = 12.0,
        mode: str = "energy",
        energy_threshold: float = 0.012,
    ):
        self.sr = sample_rate
        self.silence_blocks = max(1, int(silence_ms / 100))     # 假设输入块 100ms
        self.min_speech_samples = int(sample_rate * min_speech_ms / 1000)
        self.max_segment_samples = int(sample_rate * max_segment_s)
        self.mode = mode
        self.energy_threshold = energy_threshold
        self._buf: list[np.ndarray] = []
        self._silence_run = 0
        self._speech_samples = 0
        self._has_speech = False

        self._vad = None
        if mode == "webrtc":
            try:
                import webrtcvad

                self._vad = webrtcvad.Vad(2)
            except Exception:
                self._vad = None  # 回退到 energy

    # ---------- 判定单个块是否含语音 ----------
    def _is_speech(self, block: np.ndarray) -> bool:
        if self._vad is not None:
            try:
                pcm = (np.clip(block, -1, 1) * 32767).astype(np.int16).tobytes()
                # webrtcvad 只接受 10/20/30ms 帧；按 10ms 拆分判定
                frame_len = int(self.sr * 0.01) * 2  # 字节数
                votes = 0
                total = 0
                for i in range(0, len(pcm) - frame_len + 1, frame_len):
                    total += 1
                    if self._vad.is_speech(pcm[i : i + frame_len], self.sr):
                        votes += 1
                return total > 0 and votes / total > 0.3
            except Exception:
                pass
        # energy 回退
        return float(np.sqrt(np.mean(block ** 2))) > self.energy_threshold

    # ---------- 输入一个块，返回可能完成的句子 ----------
    def push(self, block: np.ndarray) -> Optional[Segment]:
        speech = self._is_speech(block)
        if speech:
            self._has_speech = True
            self._silence_run = 0
            self._speech_samples += len(block)
            self._buf.append(block)
        else:
            if self._has_speech:
                # 保留静音尾巴，帮助识别收尾
                self._buf.append(block)
                self._silence_run += 1

        # 句末判定
        end_by_silence = self._has_speech and self._silence_run >= self.silence_blocks
        total = sum(len(b) for b in self._buf)
        end_by_length = total >= self.max_segment_samples

        if end_by_silence or end_by_length:
            return self.flush(force=end_by_length)

        # 长时间无语音，清空防止内存累积
        if not self._has_speech and len(self._buf) > 50:
            self._buf.clear()
        return None

    def flush(self, force: bool = False) -> Optional[Segment]:
        if not self._buf:
            self._reset()
            return None
        audio = np.concatenate(self._buf).astype(np.float32)
        speech_len = self._speech_samples
        self._reset()
        if not force and speech_len < self.min_speech_samples:
            return None  # 太短，视为噪声丢弃
        if len(audio) == 0:
            return None
        return Segment(audio=audio, duration=len(audio) / self.sr)

    def _reset(self) -> None:
        self._buf = []
        self._silence_run = 0
        self._speech_samples = 0
        self._has_speech = False

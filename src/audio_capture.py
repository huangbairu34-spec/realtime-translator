"""系统音频回环采集。

抓取"电脑正在播放的声音"（YouTube 直播等），归一化为 16kHz 单声道 float32。
优先使用 soundcard 的 WASAPI loopback（Windows 上最可靠，无需开启"立体声混音"）。
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import soundcard as sc
except Exception:  # pragma: no cover
    sc = None


@dataclass
class AudioDevice:
    id: str
    name: str
    is_default: bool = False


def list_loopback_devices() -> list[AudioDevice]:
    """列出可用于回环采集的输出设备（扬声器/耳机）。"""
    if sc is None:
        return []
    devices: list[AudioDevice] = []
    try:
        default_spk = sc.default_speaker()
        default_name = default_spk.name if default_spk else None
    except Exception:
        default_name = None

    try:
        for mic in sc.all_microphones(include_loopback=True):
            # 只保留回环设备（即"扬声器回环"），过滤真实麦克风
            if getattr(mic, "isloopback", False):
                devices.append(
                    AudioDevice(
                        id=str(mic.id),
                        name=mic.name,
                        is_default=(mic.name == default_name),
                    )
                )
    except Exception:
        pass
    return devices


class LoopbackCapture:
    """后台线程持续读取系统播放音频，切块放入队列。

    读完的块为 float32、单声道、16kHz、长度 block_size。
    """

    def __init__(
        self,
        device_name: str,
        sample_rate: int = 16000,
        channels: int = 2,
        block_ms: int = 100,
    ):
        self.device_name = device_name
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = int(sample_rate * block_ms / 1000)
        self.queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=300)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[str] = None

    # ---------- 生命周期 ----------
    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="audio-capture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    @property
    def error(self) -> Optional[str]:
        return self._error

    # ---------- 内部 ----------
    def _find_mic(self):
        # 匹配回环设备
        for mic in sc.all_microphones(include_loopback=True):
            if getattr(mic, "isloopback", False) and (
                mic.name == self.device_name or str(mic.id) == self.device_name
            ):
                return mic
        # 退而求其次：按名字模糊匹配任意设备
        for mic in sc.all_microphones(include_loopback=True):
            if self.device_name and self.device_name in mic.name:
                return mic
        return None

    def _run(self) -> None:
        if sc is None:
            self._error = "soundcard 未安装"
            return
        try:
            mic = self._find_mic()
            if mic is None:
                self._error = f"未找到音频设备: {self.device_name}"
                return
            with mic.recorder(
                samplerate=self.sample_rate, channels=self.channels, blocksize=self.block_size
            ) as rec:
                while not self._stop.is_set():
                    data = rec.record(numframes=self.block_size)
                    # data: (frames, channels) float32 in [-1,1]
                    if data is None or len(data) == 0:
                        continue
                    mono = data.mean(axis=1).astype(np.float32) if data.ndim > 1 else data.astype(np.float32)
                    mono = np.clip(mono, -1.0, 1.0)
                    try:
                        self.queue.put_nowait(mono)
                    except queue.Full:
                        try:
                            self.queue.get_nowait()  # 丢最旧的，保证实时
                            self.queue.put_nowait(mono)
                        except queue.Empty:
                            pass
        except Exception as e:  # pragma: no cover
            self._error = f"音频采集异常: {e}"

    def read(self, timeout: float = 0.5) -> Optional[np.ndarray]:
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

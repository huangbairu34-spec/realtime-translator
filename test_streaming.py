"""流式架构延迟测试。

不依赖真实网络音频：用一段程序生成的"类语音"音频（调幅噪声脉冲序列，
有明确的"说话段"和"静音段"），按实时节奏喂给 Pipeline 的采集缓冲，
测量：
  1) 从"开始说话"到第一次 partial 字幕出现的时间（首字延迟）
  2) 从"说完"到 final 定稿的时间（收尾延迟）

由于类语音不能被 Whisper 识别成文字，这里用 stub ASR 度量"调度延迟"，
即架构本身的固有延迟（不含识别时间）。
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from src.config import Config
from src.pipeline import Pipeline, SubtitleEvent


class StubASR:
    """模拟一个有固定耗时的识别引擎。"""

    def __init__(self, cost_s=0.25, text="hello world this is a test"):
        self.cost_s = cost_s
        self.text = text
        self.device = "stub"

    def load(self):
        pass

    def transcribe(self, audio, is_partial=False, prompt=None):
        time.sleep(self.cost_s)
        from src.asr import ASRResult

        # 用音频长度决定文本长度，模拟"说得越多识别出越多"
        n = max(1, int(len(audio) / 16000 * 3))
        return ASRResult(text=(self.text + " ") * n, language="en", probability=1.0, duration=len(audio) / 16000)


class StubTranslator:
    def __init__(self, cost_s=0.3):
        self.cost_s = cost_s

    def translate(self, text, lang=""):
        time.sleep(self.cost_s)
        return "译:" + text[:30]

    def translate_stream(self, text, lang="", on_delta=None):
        for ch in "译:" + text[:30]:
            if on_delta:
                on_delta(ch)
            time.sleep(self.cost_s / 32)
        return "译:" + text[:30]


def run_test():
    cfg = Config()
    # 强制流式参数
    cfg.set("streaming.enabled", True, save=False)
    cfg.set("streaming.interval_ms", 700, save=False)
    cfg.set("streaming.commit_silence_ms", 300, save=False)
    cfg.set("vad.energy_threshold", 0.01, save=False)

    events: list[tuple[float, SubtitleEvent]] = []
    t_start = [0.0]

    def on_sub(ev: SubtitleEvent):
        events.append((time.time() - t_start[0], ev))

    p = Pipeline(cfg, on_sub, lambda s: None, lambda e: print("ERR", e))
    p.asr = StubASR(cost_s=0.2)
    p.translator = StubTranslator(cost_s=0.3)

    sr = 16000
    block_ms = 60
    block = int(sr * block_ms / 1000)

    # 构造音频：1s 静音 + 4s 语音(幅度>阈值) + 1.5s 静音
    def gen():
        t = np.arange(block) / sr
        sil = (np.random.randn(block) * 0.002).astype(np.float32)
        voice = (np.sin(2 * np.pi * 200 * t) * 0.3).astype(np.float32)
        return sil, voice

    sil, voice = gen()

    # 启动流水线各线程（跳过 capture，手动喂）
    p._running = True
    p._stats["started"] = time.time()
    t_start[0] = time.time()
    p.asr.load()
    threading.Thread(target=p._stream_loop, daemon=True).start()
    threading.Thread(target=p._translate_loop, daemon=True).start()

    # 实时喂入
    def feed(arr, duration):
        n = int(duration * 1000 / block_ms)
        for _ in range(n):
            with p._buf_lock:
                p._buf.append(arr)
                if arr is voice:
                    p._last_voice_ts = time.time()
                    p._sentence_active = True
            time.sleep(block_ms / 1000)

    t_speak_start = None
    # 静音1s
    for _ in range(int(1.0 * 1000 / block_ms)):
        with p._buf_lock:
            p._buf.append(sil)
        time.sleep(block_ms / 1000)
    # 语音4s —— 记录说话开始时刻
    t_speak_start = time.time()
    for _ in range(int(4.0 * 1000 / block_ms)):
        with p._buf_lock:
            p._buf.append(voice)
            p._last_voice_ts = time.time()
            p._sentence_active = True
        time.sleep(block_ms / 1000)
    t_speak_end = time.time()
    # 静音1.5s 触发定稿
    for _ in range(int(1.5 * 1000 / block_ms)):
        with p._buf_lock:
            p._buf.append(sil)
        time.sleep(block_ms / 1000)

    time.sleep(2.0)
    p._running = False

    print("\n===== 流式延迟测试结果 =====")
    print(f"说话开始 → 说话结束: {t_speak_end - t_speak_start:.2f}s")
    first_partial = None
    first_final = None
    for dt, ev in events:
        mark = ""
        if ev.kind == "partial" and first_partial is None:
            first_partial = dt
            mark = "  ← 首字上屏"
        if ev.kind == "final" and first_final is None:
            first_final = dt
            mark = "  ← 定稿"
        print(f"  t+{dt:5.2f}s  [{ev.kind:7s}] src={ev.source[:32]:32s}{mark}")
    if first_partial:
        # 首字延迟 = 首个 partial 相对说话开始的延迟
        lat = events[0][0]  # t_start 起点 ≈ 喂入开始
        print(f"\n首字延迟(说话开始→首次partial): 约 {(first_partial - (t_speak_start - t_start[0]))*1000:.0f} ms")
    if first_final:
        print(f"收尾延迟(说话结束→定稿): 约 {((first_final) - (t_speak_end - t_start[0]))*1000:.0f} ms")

    # 断言：partial 出现应早于 final，且都有
    kinds = [ev.kind for _, ev in events]
    print(f"\n事件序列: {kinds}")
    assert "partial" in kinds, "没有产生增量字幕！"
    assert "final" in kinds, "没有产生定稿字幕！"
    assert kinds.index("partial") < kinds.index("final"), "partial 应在 final 之前"
    print("✅ 流式链路工作正常：增量字幕先于定稿出现")


if __name__ == "__main__":
    run_test()

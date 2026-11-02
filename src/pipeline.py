"""实时翻译流水线（流式增量版）。

核心思想：不等整句说完，边播边识别、边翻译、边上屏。

线程模型：
  capture 线程  ── 持续读音频块 ──▶ 共享环形缓冲（只存"当前这一句"的音频）
  stream  线程  ── 每 interval_ms ──▶ 对缓冲区做增量识别
                                     ├─ 有结果 → 立即翻译（流式）→ 回调 partial
                                     └─ 检测到句末静音 → 定稿 → 回调 final，清空缓冲
  UI 侧用 partial(临时) + final(定稿) 区分显示，避免闪烁。

延迟构成（说话 → 首字上屏）：
  识别间隔(≤0.9s，通常说到一半就出) + 识别(0.2~0.5s) + 翻译首字(0.3~0.6s) ≈ 1~1.5s
  相比旧版"整句+静音判定+串行"的 2~5s 显著改善。
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .asr import ASRResult, WhisperASR
from .audio_capture import LoopbackCapture
from .config import Config
from .translator import Translator
from .vad import Segmenter

log = logging.getLogger("pipeline")


@dataclass
class SubtitleEvent:
    """一次字幕更新。

    kind:
      "partial" —— 增量识别/翻译的临时结果（会随后续结果刷新）
      "final"   —— 一句定稿（不再变化，可写入历史）
      "status"  —— 状态文本
    """

    source: str
    translated: str
    language: str = ""
    kind: str = "final"


class Pipeline:
    def __init__(
        self,
        cfg: Config,
        on_subtitle: Callable[[SubtitleEvent], None],
        on_status: Callable[[str], None],
        on_error: Callable[[str], None],
    ):
        self.cfg = cfg
        self.on_subtitle = on_subtitle
        self.on_status = on_status
        self.on_error = on_error

        self._running = False
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

        self.capture: Optional[LoopbackCapture] = None
        self.asr: Optional[WhisperASR] = None
        self.translator: Optional[Translator] = None

        # 流式缓冲（只放当前这句）
        self._buf: list[np.ndarray] = []
        self._buf_lock = threading.Lock()
        self._last_voice_ts = 0.0       # 最后一次检测到"有声音"的时间
        self._sentence_active = False
        self._tail_silence = 0          # 当前句尾部静音块计数

        # 识别结果缓存
        self._last_partial_text = ""
        self._last_partial_lang = ""
        self._committed_texts: list[str] = []   # 本句已定稿的若干段（长句切分时）

        self._translate_q: "queue.Queue[tuple[str, str, bool]]" = queue.Queue(maxsize=8)
        self._stats = {"segments": 0, "translated": 0, "started": 0.0, "first_latency": 0.0}

    # ==================== 生命周期 ====================
    def start(self) -> None:
        if self._running:
            return
        self._stop.clear()

        self.capture = LoopbackCapture(
            device_name=self.cfg.get("audio.device", ""),
            sample_rate=self.cfg.get("audio.sample_rate", 16000),
            channels=self.cfg.get("audio.channels", 2),
            block_ms=self.cfg.get("audio.block_ms", 60),
        )
        self.asr = WhisperASR(
            model=self.cfg.get("asr.model", "small"),
            device=self.cfg.get("asr.device", "cuda"),
            compute_type=self.cfg.get("asr.compute_type", "float16"),
            language=self.cfg.get("asr.language", ""),
            beam_size=self.cfg.get("asr.beam_size", 1),
            initial_prompt=self.cfg.get("asr.initial_prompt", ""),
        )
        if self.cfg.get("translate.enabled", True):
            self.translator = Translator(
                base_url=self.cfg.get("translate.base_url", ""),
                api_key=self.cfg.get("translate.api_key", ""),
                model=self.cfg.get("translate.model", ""),
                target_lang=self.cfg.get("translate.target_lang", "zh"),
                fallback_lang=self.cfg.get("translate.fallback_lang", "en"),
                temperature=self.cfg.get("translate.temperature", 0.3),
                timeout=self.cfg.get("translate.timeout", 20),
                context_turns=self.cfg.get("translate.context_turns", 2),
            )

        self._running = True
        self._stats["started"] = time.time()
        # 独立线程：识别+翻译（避免阻塞音频采集）
        threading.Thread(target=self._stream_loop, name="stream", daemon=True).start()
        # 翻译串行线程（保证顺序 & 复用上下文）
        threading.Thread(target=self._translate_loop, name="translate", daemon=True).start()
        # 预热模型
        threading.Thread(target=self._warmup_and_start, name="warmup", daemon=True).start()
        self.on_status("正在加载识别模型...")

    def _warmup_and_start(self) -> None:
        try:
            t0 = time.time()
            self.asr.load()
            log.info("模型就绪，耗时 %.1fs", time.time() - t0)
            self.capture.start()
            self._threads = [
                threading.Thread(target=self._capture_loop, name="capture", daemon=True)
            ]
            self._threads[0].start()
            self.on_status("已启动 · 正在聆听")
        except Exception as e:
            log.exception("启动失败")
            self.on_error(f"启动失败：{e}")
            self._running = False

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self._stop.set()
        if self.capture:
            self.capture.stop()
        for t in self._threads:
            if t.is_alive():
                t.join(timeout=1)
        self.on_status("已暂停")

    def is_running(self) -> bool:
        return self._running

    def stats(self) -> dict:
        return dict(self._stats)

    # ==================== 采集线程 ====================
    def _capture_loop(self) -> None:
        """读音频 → 判断有无声音 → 维护"当前句"缓冲。"""
        energy_th = self.cfg.get("vad.energy_threshold", 0.010)
        block_ms = self.cfg.get("audio.block_ms", 60)
        sr = self.cfg.get("audio.sample_rate", 16000)
        # 静音尾巴最多保留 ~0.5s，避免缓冲无限增长拖慢识别
        max_tail_blocks = max(1, int(500 / block_ms))

        while self._running and not self._stop.is_set():
            if self.capture and self.capture.error:
                self.on_error(self.capture.error)
                break
            block = self.capture.read(timeout=0.5)
            if block is None:
                continue
            rms = float(np.sqrt(np.mean(block ** 2)))
            has_voice = rms > energy_th
            now = time.time()
            with self._buf_lock:
                if has_voice:
                    self._last_voice_ts = now
                    self._sentence_active = True
                    self._tail_silence = 0
                    self._buf.append(block)
                elif self._sentence_active:
                    # 只保留有限的静音尾巴
                    self._tail_silence = getattr(self, "_tail_silence", 0) + 1
                    if self._tail_silence <= max_tail_blocks:
                        self._buf.append(block)

    # ==================== 流式识别线程 ====================
    def _stream_loop(self) -> None:
        """流式主循环：高频检查句末（低延迟定稿），按间隔做增量识别。"""
        interval = self.cfg.get("streaming.interval_ms", 900) / 1000.0
        min_partial = self.cfg.get("streaming.min_partial_s", 0.6)
        max_partial = self.cfg.get("streaming.max_partial_s", 7.0)
        commit_silence = self.cfg.get("streaming.commit_silence_ms", 350) / 1000.0
        keep_ctx = self.cfg.get("streaming.keep_context_s", 1.0)
        sr = self.cfg.get("audio.sample_rate", 16000)

        tick = 0.12  # 120ms 检查一次，保证句末能迅速定稿
        last_recognize = 0.0

        while self._running and not self._stop.is_set():
            time.sleep(tick)
            now = time.time()

            with self._buf_lock:
                active = self._sentence_active
                has_audio = len(self._buf) > 0
                silent_for = now - self._last_voice_ts if self._last_voice_ts else 0
                audio = np.concatenate(self._buf) if has_audio else None
            if audio is None:
                continue
            dur = len(audio) / sr

            # 情况一：句末静音达标 → 立即定稿（不等下一个识别间隔）
            if active and silent_for >= commit_silence and dur >= min_partial:
                self._do_recognize(audio, is_final=True)
                with self._buf_lock:
                    self._buf = []
                    self._sentence_active = False
                    self._tail_silence = 0
                    self._last_partial_text = ""
                    self._committed_texts = []
                last_recognize = now
                continue

            # 情况二：到点做增量识别
            if active and dur >= min_partial and (now - last_recognize) >= interval:
                self._do_recognize(audio, is_final=False)
                last_recognize = now
                # 缓冲太长则滑动窗口，保留最近 keep_ctx 秒
                if dur >= max_partial:
                    with self._buf_lock:
                        keep = int(sr * keep_ctx)
                        tail = np.concatenate(self._buf)[-keep:]
                        self._buf = [tail]

    def _do_recognize(self, audio: np.ndarray, is_final: bool) -> None:
        """对一段音频识别 + 翻译，产出 partial/final 事件。"""
        # 长句切分拼接：若之前有已定稿片段，用它们拼前缀帮助上下文
        prompt = " ".join(self._committed_texts[-2:]) if self._committed_texts else None
        try:
            result = self.asr.transcribe(audio, is_partial=not is_final, prompt=prompt)
        except Exception as e:
            log.warning("识别异常: %s", e)
            result = None

        if result is None or not result.text.strip():
            # 定稿时若本次识别为空，但之前有增量结果，则用增量结果定稿，
            # 避免"说过了却什么都没留下"。
            if is_final and self._last_partial_text:
                self._translate_q.put_nowait(
                    (self._last_partial_text, self._last_partial_lang or "", True)
                )
            return

        text = result.text.strip()
        if text == self._last_partial_text and not is_final:
            return  # 无变化，不重复翻译
        self._last_partial_text = text
        self._last_partial_lang = result.language
        self._stats["segments"] += 1
        if self._stats["first_latency"] == 0:
            self._stats["first_latency"] = time.time() - self._stats["started"]

        # 翻译交给独立线程，保证顺序
        try:
            self._translate_q.put_nowait((text, result.language, is_final))
        except queue.Full:
            # 队列满 → 丢掉旧的 partial
            try:
                self._translate_q.get_nowait()
                self._translate_q.put_nowait((text, result.language, is_final))
            except queue.Empty:
                pass

    def _drain_latest(self, first: tuple[str, str, bool]) -> tuple[str, str, bool]:
        """合并积压：只保留最新的 partial，避免翻译过时内容浪费算力。

        final 不可丢弃（必须定稿）。
        """
        latest = first
        while latest[2] is False:  # 当前是 partial，看看有没有更新的
            try:
                nxt = self._translate_q.get_nowait()
            except queue.Empty:
                break
            if nxt[2] is True:  # 遇到 final，先把 final 存下，返回 final
                return nxt
            latest = nxt
        return latest

    # ==================== 翻译线程 ====================
    def _translate_loop(self) -> None:
        stream_enabled = self.cfg.get("streaming.enabled", True) and self.cfg.get(
            "translate.stream", True
        )
        while self._running or not self._translate_q.empty():
            try:
                text, lang, is_final = self._translate_q.get(timeout=0.5)
            except queue.Empty:
                if not self._running:
                    break
                continue

            # 合并积压：跳过过时的 partial，只翻最新的
            text, lang, is_final = self._drain_latest((text, lang, is_final))

            if self.translator is None:
                self.on_subtitle(SubtitleEvent(text, "", lang, "final" if is_final else "partial"))
                continue

            # 增量阶段也做流式翻译（预览），让译文边听边出；
            # 定稿时会用完整文本再翻一次并锁定。
            speculative = self.cfg.get("translate.speculative", True)
            if not is_final and not speculative:
                # 不做预览翻译时，直接显示原文，等定稿再翻
                self.on_subtitle(SubtitleEvent(text, text, lang, "partial"))
                continue

            if stream_enabled:
                acc: list[str] = []

                def on_delta(piece: str) -> None:
                    acc.append(piece)
                    # 预览用 partial，定稿也用 partial 逐步刷新，最后发 final
                    self.on_subtitle(SubtitleEvent(text, "".join(acc), lang, "partial"))

                translated = self.translator.translate_stream(text, lang, on_delta=on_delta)
            else:
                translated = self.translator.translate(text, lang)

            if translated:
                self._stats["translated"] += 1
            if is_final:
                self._committed_texts.append(text)
            self.on_subtitle(
                SubtitleEvent(text, translated or text, lang, "final" if is_final else "partial")
            )

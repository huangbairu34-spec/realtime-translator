"""大模型翻译模块。

通过 OpenAI 兼容接口（OpenAI / DeepSeek / 硅基流动 / 通义 / Moonshot 等）翻译字幕。
特点：
  - 源语言自动，译为目标语言；若源==目标则译成 fallback。
  - 携带最近几句作为上下文，保证术语与人称一致。
  - 翻译失败时原样返回，不阻塞流水线。
"""
from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Callable, Optional

log = logging.getLogger("translate")

LANG_NAMES = {
    "zh": "简体中文",
    "en": "English",
    "ja": "日本語",
    "ko": "한국어",
    "ru": "Русский",
    "fr": "Français",
    "de": "Deutsch",
    "es": "Español",
}


class Translator:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        target_lang: str = "zh",
        fallback_lang: str = "en",
        temperature: float = 0.3,
        timeout: float = 20,
        context_turns: int = 2,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.target_lang = target_lang
        self.fallback_lang = fallback_lang
        self.temperature = temperature
        self.timeout = timeout
        self.context_turns = context_turns
        self._client = None
        self._lock = threading.Lock()
        self._history: deque[tuple[str, str]] = deque(maxlen=max(0, context_turns))

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.base_url, api_key=self.api_key or "EMPTY", timeout=self.timeout
            )
        return self._client

    def reset(self) -> None:
        self._history.clear()

    def _build_messages(self, text: str, src_lang: str) -> list[dict]:
        target = self.target_lang
        if src_lang and src_lang.split("-")[0] == target.split("-")[0]:
            target = self.fallback_lang
        target_name = LANG_NAMES.get(target, target)

        system = (
            "你是一个专业的实时字幕翻译引擎。请把用户给出的直播/视频口语文本翻译成"
            f"{target_name}。要求：\n"
            "1) 只输出翻译结果，不要任何解释、引号或前后缀；\n"
            "2) 保持口语自然流畅，符合中文表达习惯；\n"
            "3) 人名、专有名词、术语要准确并保持全篇一致；\n"
            "4) 若输入本身已是目标语言，则原样返回。"
        )
        messages = [{"role": "system", "content": system}]
        for src, dst in list(self._history)[-self.context_turns:]:
            messages.append({"role": "user", "content": src})
            messages.append({"role": "assistant", "content": dst})
        messages.append({"role": "user", "content": text})
        return messages

    def translate(self, text: str, src_lang: str = "") -> str:
        text = text.strip()
        if not text:
            return ""
        try:
            client = self._get_client()
            with self._lock:
                messages = self._build_messages(text, src_lang)
            resp = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )
            out = (resp.choices[0].message.content or "").strip()
            if out:
                self._history.append((text, out))
                return out
            return text
        except Exception as e:
            log.warning("翻译失败: %s", e)
            return ""  # 失败返回空，调用方决定回退

    def translate_stream(
        self,
        text: str,
        src_lang: str = "",
        on_delta: Optional[Callable[[str], None]] = None,
    ) -> str:
        """流式翻译：每收到一个增量片段就回调 on_delta(增量文本)。

        返回完整译文。流式让用户"边翻边看"，感知延迟大幅降低。
        若流式不可用，回退为一次性翻译。
        """
        text = text.strip()
        if not text:
            return ""
        try:
            client = self._get_client()
            with self._lock:
                messages = self._build_messages(text, src_lang)
            stream = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                stream=True,
            )
            parts: list[str] = []
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                piece = getattr(delta, "content", None)
                if piece:
                    parts.append(piece)
                    if on_delta:
                        on_delta(piece)
            out = "".join(parts).strip()
            if out:
                self._history.append((text, out))
                return out
            return ""
        except Exception as e:
            log.warning("流式翻译失败(%s)，回退一次性翻译", e)
            out = self.translate(text, src_lang)
            if out and on_delta:
                on_delta(out)
            return out

    def test_connection(self) -> tuple[bool, str]:
        """设置面板里"测试连接"用。"""
        try:
            out = self.translate("Hello, this is a connection test.")
            if out:
                return True, f"连接成功，示例输出：{out}"
            return False, "接口返回为空，请检查模型名/Key/额度"
        except Exception as e:
            return False, f"连接失败：{e}"

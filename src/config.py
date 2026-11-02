"""配置加载与管理。

读取 config/config.yaml，提供点号访问（如 cfg.get("asr.model")），
并支持把运行时改动写回文件。
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "config.yaml"


class Config:
    def __init__(self, path: Path = CONFIG_PATH):
        self.path = path
        self._lock = threading.RLock()
        self._data: dict[str, Any] = {}
        self.load()

    def load(self) -> "Config":
        with self._lock:
            if self.path.exists():
                with open(self.path, "r", encoding="utf-8") as f:
                    self._data = yaml.safe_load(f) or {}
            else:
                self._data = {}
        return self

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """cfg.get('asr.model', 'medium')"""
        with self._lock:
            node: Any = self._data
            for part in dotted_key.split("."):
                if not isinstance(node, dict) or part not in node:
                    return default
                node = node[part]
            return node if node is not None else default

    def set(self, dotted_key: str, value: Any, save: bool = True) -> None:
        with self._lock:
            parts = dotted_key.split(".")
            node = self._data
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
            if save:
                self._write()

    def update_many(self, mapping: dict[str, Any], save: bool = True) -> None:
        with self._lock:
            for k, v in mapping.items():
                parts = k.split(".")
                node = self._data
                for part in parts[:-1]:
                    node = node.setdefault(part, {})
                node[parts[-1]] = v
            if save:
                self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".yaml.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                self._data, f, allow_unicode=True, sort_keys=False, default_flow_style=False
            )
        os.replace(tmp, self.path)

    @property
    def data(self) -> dict[str, Any]:
        return self._data


# 全局单例
config = Config()

"""独立主窗口：完整对话历史。

与悬浮字幕条并行存在，显示所有历史记录（原文 + 译文 + 时间戳 + 语言），
支持滚动回看、右键复制、导出为 txt、清空。
"""
from __future__ import annotations

import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QFont, QGuiApplication
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

LANG_ZH = {"en": "英语", "ja": "日语", "zh": "中文", "ko": "韩语", "": "自动"}


class HistoryWindow(QMainWindow):
    request_settings = pyqtSignal()
    request_toggle = pyqtSignal()
    request_clear = pyqtSignal()

    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self._records: deque[tuple] = deque(maxlen=2000)
        self._auto_scroll = True

        self.setWindowTitle("实时翻译 · 对话记录")
        self.resize(720, 480)
        self._init_ui()

    # ---------- UI ----------
    def _init_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 6)

        # 工具条
        bar = QHBoxLayout()
        self.btn_toggle = QPushButton("开始 / 暂停")
        self.btn_toggle.clicked.connect(self.request_toggle.emit)
        btn_clear = QPushButton("清空")
        btn_clear.clicked.connect(self.clear)
        btn_export = QPushButton("导出 TXT")
        btn_export.clicked.connect(self.export_txt)
        btn_settings = QPushButton("设置")
        btn_settings.clicked.connect(self.request_settings.emit)
        bar.addWidget(self.btn_toggle)
        bar.addWidget(btn_clear)
        bar.addWidget(btn_export)
        bar.addWidget(btn_settings)
        bar.addStretch(1)
        self.lang_label = QLabel("源语言：自动")
        self.lang_label.setStyleSheet("color:#888;")
        bar.addWidget(self.lang_label)
        layout.addLayout(bar)

        # 历史显示区
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setFont(QFont("Microsoft YaHei UI", 11))
        self.text.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.text.customContextMenuRequested.connect(self._show_menu)
        self.text.setStyleSheet(
            "QPlainTextEdit{background:#1b1b1f;color:#e8e8e8;border:1px solid #333;border-radius:6px;padding:8px;}"
        )
        layout.addWidget(self.text, 1)

        # 状态栏
        sb = QStatusBar()
        self.status_label = QLabel("就绪")
        sb.addWidget(self.status_label)
        self.setStatusBar(sb)

    # ---------- 记录 ----------
    def add_record(self, source: str, translated: str, language: str = "") -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self._records.append((ts, source, translated, language))
        lang = LANG_ZH.get(language.split("-")[0], language or "自动")
        self.lang_label.setText(f"源语言：{lang}")

        block = f"[{ts}]  {source}\n"
        if translated and translated != source:
            block += f"      → {translated}\n"
        block += "\n"
        self.text.appendPlainText(block)
        if self._auto_scroll:
            self.text.verticalScrollBar().setValue(self.text.verticalScrollBar().maximum())

    def clear(self) -> None:
        self._records.clear()
        self.text.clear()

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def set_running(self, running: bool) -> None:
        self.btn_toggle.setText("暂停" if running else "开始")

    # ---------- 导出 ----------
    def export_txt(self) -> None:
        if not self._records:
            self.set_status("没有可导出的记录")
            return
        default = f"transcript_{datetime.now():%Y%m%d_%H%M%S}.txt"
        path, _ = QFileDialog.getSaveFileName(
            self, "导出对话记录", str(Path.home() / "Desktop" / default), "文本文件 (*.txt)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                for ts, src, dst, lang in self._records:
                    f.write(f"[{ts}] {src}\n")
                    if dst and dst != src:
                        f.write(f"    → {dst}\n")
                    f.write("\n")
            self.set_status(f"已导出：{path}")
        except Exception as e:
            self.set_status(f"导出失败：{e}")

    # ---------- 右键 ----------
    def _show_menu(self, pos) -> None:
        menu = QMenu(self)
        act_copy = QAction("复制选中", self)
        act_copy.triggered.connect(self.text.copy)
        act_all = QAction("复制全部", self)
        act_all.triggered.connect(self._copy_all)
        act_auto = QAction("自动滚动到最新", self)
        act_auto.setCheckable(True)
        act_auto.setChecked(self._auto_scroll)
        act_auto.triggered.connect(self._set_auto_scroll)
        menu.addAction(act_copy)
        menu.addAction(act_all)
        menu.addSeparator()
        menu.addAction(act_auto)
        menu.exec(self.text.mapToGlobal(pos))

    def _copy_all(self) -> None:
        QGuiApplication.clipboard().setText(self.text.toPlainText())

    def _set_auto_scroll(self, checked: bool) -> None:
        self._auto_scroll = checked

    # ---------- 关闭时隐藏而非退出 ----------
    def closeEvent(self, event):  # noqa: N802
        event.ignore()
        self.hide()

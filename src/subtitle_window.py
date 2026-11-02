"""悬浮字幕条窗口。

无边框、置顶、半透明，显示"原文 + 译文"，可拖动、调透明度，
支持鼠标穿透（不挡直播操作）。译文为大字，原文为小字浅色。
"""
from __future__ import annotations

from collections import deque
from typing import Optional

from PyQt6.QtCore import Qt, QPoint, pyqtSignal, QTimer
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QMenu


class SubtitleWindow(QWidget):
    request_settings = pyqtSignal()
    request_toggle = pyqtSignal()
    request_clear = pyqtSignal()
    request_quit = pyqtSignal()

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._drag_pos: Optional[QPoint] = None
        self._lines: deque[tuple[str, str]] = deque(maxlen=cfg.get("subtitle.max_lines", 4))

        self._init_window()
        self._init_ui()
        self._apply_style()

    # ---------- 窗口基础 ----------
    def _init_window(self) -> None:
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_menu)
        self._apply_geometry()
        if self.cfg.get("subtitle.click_through", False):
            self._enable_click_through(True)

    def _apply_geometry(self) -> None:
        screen = QApplication.primaryScreen().availableGeometry()
        wr = float(self.cfg.get("subtitle.width_ratio", 0.7))
        w = int(screen.width() * wr)
        h = int(float(self.cfg.get("subtitle.font_size", 20)) * 4.5)
        bottom = int(self.cfg.get("subtitle.bottom_margin", 90))
        x = screen.x() + (screen.width() - w) // 2
        y = screen.y() + screen.height() - bottom - h
        self.setGeometry(x, y, w, h)

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 16, 28, 16)
        layout.setSpacing(4)

        self.source_label = QLabel("")
        self.source_label.setWordWrap(True)
        self.source_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.trans_label = QLabel("实时翻译已就绪 · 右键显示菜单")
        self.trans_label.setWordWrap(True)
        self.trans_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.source_label)
        layout.addWidget(self.trans_label)
        layout.addStretch(1)

    def _apply_style(self) -> None:
        fam = self.cfg.get("subtitle.font_family", "Microsoft YaHei UI")
        fs = int(self.cfg.get("subtitle.font_size", 20))
        color = self.cfg.get("subtitle.color", "#FFFFFF")
        src_color = self.cfg.get("subtitle.source_color", "#9FB3C8")

        f_trans = QFont(fam, fs)
        f_trans.setBold(True)
        self.trans_label.setFont(f_trans)
        self.trans_label.setStyleSheet(f"color:{color}; background:transparent;")

        f_src = QFont(fam, max(10, int(fs * 0.72)))
        self.source_label.setFont(f_src)
        self.source_label.setStyleSheet(f"color:{src_color}; background:transparent;")
        self.source_label.setVisible(bool(self.cfg.get("subtitle.show_source", True)))

    # ---------- 半透明背景 + 圆角 ----------
    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        opacity = float(self.cfg.get("subtitle.bg_opacity", 0.55))
        bg = QColor(self.cfg.get("subtitle.bg_color", "#000000"))
        bg.setAlphaF(max(0.0, min(1.0, opacity)))
        path = QPainterPath()
        path.addRoundedRect(0, 0, self.width(), self.height(), 16, 16)
        painter.fillPath(path, bg)

    # ---------- 字幕更新 ----------
    def push_subtitle(self, source: str, translated: str, kind: str = "final") -> None:
        """kind: 'partial'=增量临时结果(半透明), 'final'=定稿"""
        if not translated:
            translated = source
        self.source_label.setText(source)
        self.trans_label.setText(translated)
        # 增量中用较低不透明度提示"还在翻"，定稿后恢复
        if kind == "partial":
            self.trans_label.setStyleSheet(
                f"color:{self.cfg.get('subtitle.color', '#FFFFFF')}; background:transparent; opacity:0.7;"
            )
        else:
            self.trans_label.setStyleSheet(
                f"color:{self.cfg.get('subtitle.color', '#FFFFFF')}; background:transparent;"
            )
            self._lines.append((source, translated))

    def set_status(self, text: str) -> None:
        self.trans_label.setText(text)

    def clear(self) -> None:
        self._lines.clear()
        self.source_label.setText("")
        self.trans_label.setText("")

    # ---------- 鼠标穿透 ----------
    def _enable_click_through(self, enable: bool) -> None:
        flags = self.windowFlags()
        if enable:
            self.setWindowFlag(Qt.WindowType.WindowTransparentForInput, True)
        else:
            self.setWindowFlag(Qt.WindowType.WindowTransparentForInput, False)
        self.setWindowFlags(flags)
        self.show()

    def apply_settings(self) -> None:
        """设置面板保存后调用，实时生效。"""
        self._apply_geometry()
        self._apply_style()
        self._enable_click_through(bool(self.cfg.get("subtitle.click_through", False)))
        self.update()

    # ---------- 拖动 ----------
    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._drag_pos and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):  # noqa: N802
        self._drag_pos = None

    def wheelEvent(self, event):  # noqa: N802
        # 滚轮调字体大小
        delta = event.angleDelta().y()
        fs = int(self.cfg.get("subtitle.font_size", 20))
        fs = max(12, min(48, fs + (1 if delta > 0 else -1)))
        self.cfg.set("subtitle.font_size", fs)
        self._apply_geometry()
        self._apply_style()

    # ---------- 右键菜单 ----------
    def _show_menu(self, pos: QPoint) -> None:
        menu = QMenu(self)
        menu.addAction("开始 / 暂停翻译", self.request_toggle.emit)
        menu.addAction("清空字幕", self.request_clear.emit)
        menu.addSeparator()
        act_through = menu.addAction("鼠标穿透")
        act_through.setCheckable(True)
        act_through.setChecked(bool(self.cfg.get("subtitle.click_through", False)))
        act_through.triggered.connect(self._toggle_click_through)
        menu.addAction("设置...", self.request_settings.emit)
        menu.addSeparator()
        menu.addAction("退出", self.request_quit.emit)
        menu.exec(self.mapToGlobal(pos))

    def _toggle_click_through(self, checked: bool) -> None:
        self.cfg.set("subtitle.click_through", checked)
        self._enable_click_through(checked)

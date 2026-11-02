"""实时翻译软件 —— 主程序。

用法：
    python main.py                # 直接启动
    python main.py --select       # 启动时先弹出设备/设置窗口
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 允许从项目根直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from src.config import config
from src.history_window import HistoryWindow
from src.pipeline import Pipeline
from src.settings_dialog import SettingsDialog
from src.subtitle_window import SubtitleWindow

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("main")


class Bridge(QObject):
    """把工作线程的回调转到 Qt 主线程。"""

    subtitle = pyqtSignal(str, str, str, str)
    status = pyqtSignal(str)
    error = pyqtSignal(str)


class App:
    def __init__(self):
        self.qt_app = QApplication(sys.argv)
        self.qt_app.setQuitOnLastWindowClosed(False)

        self.bridge = Bridge()
        self.window = SubtitleWindow(config)
        self.history = HistoryWindow(config)
        self.pipeline: Pipeline | None = None
        self._settings_dialog: SettingsDialog | None = None
        self._last_lang: str = ""

        self.bridge.subtitle.connect(self._on_subtitle)
        self.bridge.status.connect(self._on_status)
        self.bridge.error.connect(self._on_error)

        self.window.request_settings.connect(self.open_settings)
        self.window.request_toggle.connect(self.toggle)
        self.window.request_clear.connect(self._clear_all)
        self.window.request_quit.connect(self.quit)

        self.history.request_settings.connect(self.open_settings)
        self.history.request_toggle.connect(self.toggle)
        self.history.request_clear.connect(self._clear_all)

        self._init_tray()
        self.window.show()
        if config.get("ui.show_history", True):
            self.history.show()

    # ---------- 字幕分发 ----------
    def _on_subtitle(self, source: str, translated: str, language: str = "", kind: str = "final") -> None:
        self._last_lang = language
        # 悬浮字幕条：partial 与 final 都实时刷新
        self.window.push_subtitle(source, translated, kind)
        # 对话记录：只在定稿时写入，避免重复
        if kind == "final":
            self.history.add_record(source, translated, language)

    def _on_status(self, text: str) -> None:
        self.window.set_status(text)
        self.history.set_status(text)

    def _clear_all(self) -> None:
        self.window.clear()
        self.history.clear()

    # ---------- 系统托盘 ----------
    def _init_tray(self) -> None:
        self.tray = QSystemTrayIcon(self.qt_app)
        self.tray.setToolTip("实时翻译")
        menu = QMenu()
        act_toggle = QAction("开始 / 暂停", self.qt_app)
        act_toggle.triggered.connect(self.toggle)
        act_history = QAction("显示 / 隐藏对话记录", self.qt_app)
        act_history.triggered.connect(self.toggle_history)
        act_settings = QAction("设置...", self.qt_app)
        act_settings.triggered.connect(self.open_settings)
        act_clear = QAction("清空字幕", self.qt_app)
        act_clear.triggered.connect(self._clear_all)
        act_quit = QAction("退出", self.qt_app)
        act_quit.triggered.connect(self.quit)
        menu.addAction(act_toggle)
        menu.addAction(act_history)
        menu.addAction(act_settings)
        menu.addAction(act_clear)
        menu.addSeparator()
        menu.addAction(act_quit)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(
            lambda r: self.toggle() if r == QSystemTrayIcon.ActivationReason.DoubleClick else None
        )
        self.tray.show()

    def toggle_history(self) -> None:
        if self.history.isVisible():
            self.history.hide()
        else:
            self.history.show()
            self.history.raise_()
            self.history.activateWindow()

    # ---------- 控制 ----------
    def start(self, autostart: bool = True) -> None:
        if self.pipeline and self.pipeline.is_running():
            return
        if not config.get("audio.device", ""):
            self.window.set_status("未选择音频设备 · 右键→设置")
        self.pipeline = Pipeline(
            cfg=config,
            on_subtitle=lambda s: self.bridge.subtitle.emit(
                s.source, s.translated, s.language, s.kind
            ),
            on_status=lambda t: self.bridge.status.emit(t),
            on_error=lambda e: self.bridge.error.emit(e),
        )
        self.pipeline.start()
        self.history.set_running(True)

    def stop(self) -> None:
        if self.pipeline:
            self.pipeline.stop()
        self.history.set_running(False)

    def toggle(self) -> None:
        if self.pipeline and self.pipeline.is_running():
            self.stop()
        else:
            self.start()

    def open_settings(self) -> None:
        dlg = SettingsDialog(config, parent=None)
        dlg.settings_saved.connect(self.window.apply_settings)
        dlg.settings_saved.connect(self._restart_if_running)
        self._settings_dialog = dlg
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _restart_if_running(self) -> None:
        if self.pipeline and self.pipeline.is_running():
            self.window.set_status("设置已更新 · 正在重启引擎...")
            self.stop()
            self.start()

    def _on_error(self, msg: str) -> None:
        log.error(msg)
        self.window.set_status(f"⚠️ {msg}")

    def quit(self) -> None:
        self.stop()
        self.tray.hide()
        self.qt_app.quit()

    def run(self) -> None:
        self.start()
        sys.exit(self.qt_app.exec())


def main() -> None:
    parser = argparse.ArgumentParser(description="实时翻译")
    parser.add_argument("--select", action="store_true", help="启动时先打开设置")
    args = parser.parse_args()

    app = App()
    if args.select:
        app.open_settings()
    app.run()


if __name__ == "__main__":
    main()

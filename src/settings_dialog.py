"""设置面板。

分四个标签页：音频 / 识别 / 翻译 / 字幕。保存后写回 config.yaml 并热应用。
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .audio_capture import list_loopback_devices
from .translator import Translator

PRESETS = {
    "DeepSeek": ("https://api.deepseek.com/v1", "deepseek-chat"),
    "OpenAI": ("https://api.openai.com/v1", "gpt-4o-mini"),
    "硅基流动": ("https://api.siliconflow.cn/v1", "Qwen/Qwen2.5-7B-Instruct"),
    "通义千问": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus"),
    "Moonshot": ("https://api.moonshot.cn/v1", "moonshot-v1-8k"),
}


class SettingsDialog(QDialog):
    settings_saved = pyqtSignal()

    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.setWindowTitle("实时翻译 · 设置")
        self.setMinimumWidth(560)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_audio_tab(), "音频")
        self.tabs.addTab(self._build_asr_tab(), "识别")
        self.tabs.addTab(self._build_translate_tab(), "翻译")
        self.tabs.addTab(self._build_subtitle_tab(), "字幕")
        layout.addWidget(self.tabs)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self._save)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    # ---------- 音频 ----------
    def _build_audio_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self.device_combo = QComboBox()
        self.device_combo.setEditable(True)
        devs = list_loopback_devices()
        if devs:
            for d in devs:
                label = d.name + ("  [默认]" if d.is_default else "")
                self.device_combo.addItem(label, d.name)
        else:
            self.device_combo.addItem("(未检测到回环设备，请手动输入)", "")
        cur = self.cfg.get("audio.device", "")
        if cur:
            idx = self.device_combo.findData(cur)
            if idx >= 0:
                self.device_combo.setCurrentIndex(idx)
            else:
                self.device_combo.setEditText(cur)
        form.addRow("系统音频设备：", self.device_combo)
        form.addRow(QLabel("<small>选择你的扬声器/耳机（回环设备），即直播声音的输出端</small>"))
        btn = QPushButton("刷新设备列表")
        btn.clicked.connect(self._refresh_devices)
        form.addRow(btn)
        return w

    def _refresh_devices(self) -> None:
        self.device_combo.clear()
        for d in list_loopback_devices():
            label = d.name + ("  [默认]" if d.is_default else "")
            self.device_combo.addItem(label, d.name)

    # ---------- 识别 ----------
    def _build_asr_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        self.model_combo = QComboBox()
        for m in ["tiny", "base", "small", "medium", "large-v3", "distil-large-v3"]:
            self.model_combo.addItem(m)
        self.model_combo.setCurrentText(self.cfg.get("asr.model", "medium"))
        form.addRow("Whisper 模型：", self.model_combo)
        form.addRow(QLabel("<small>4060 8G 推荐 medium（快准平衡）或 large-v3（最准）</small>"))

        self.device_combo2 = QComboBox()
        self.device_combo2.addItems(["cuda", "cpu"])
        self.device_combo2.setCurrentText(self.cfg.get("asr.device", "cuda"))
        form.addRow("计算设备：", self.device_combo2)

        self.ctype_combo = QComboBox()
        self.ctype_combo.addItems(["float16", "int8_float16", "int8"])
        self.ctype_combo.setCurrentText(self.cfg.get("asr.compute_type", "float16"))
        form.addRow("精度：", self.ctype_combo)

        self.lang_combo = QComboBox()
        self.lang_combo.addItem("自动检测", "")
        for code, name in [("en", "英语"), ("ja", "日语"), ("zh", "中文"), ("ko", "韩语")]:
            self.lang_combo.addItem(name, code)
        cur = self.cfg.get("asr.language", "")
        idx = self.lang_combo.findData(cur)
        self.lang_combo.setCurrentIndex(idx if idx >= 0 else 0)
        form.addRow("源语言：", self.lang_combo)

        self.prompt_edit = QLineEdit(self.cfg.get("asr.initial_prompt", ""))
        self.prompt_edit.setPlaceholderText("可选：主播名、专业术语，帮助识别更准")
        form.addRow("提示词：", self.prompt_edit)

        # --- 实时性调节 ---
        self.stream_enabled = QCheckBox("流式增量识别（边播边出字，强烈建议开）")
        self.stream_enabled.setChecked(bool(self.cfg.get("streaming.enabled", True)))
        form.addRow(self.stream_enabled)

        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(300, 3000)
        self.interval_spin.setSingleStep(100)
        self.interval_spin.setSuffix(" ms")
        self.interval_spin.setValue(int(self.cfg.get("streaming.interval_ms", 900)))
        form.addRow("识别间隔：", self.interval_spin)

        self.commit_spin = QSpinBox()
        self.commit_spin.setRange(150, 1200)
        self.commit_spin.setSingleStep(50)
        self.commit_spin.setSuffix(" ms")
        self.commit_spin.setValue(int(self.cfg.get("streaming.commit_silence_ms", 350)))
        form.addRow("句末静音判定：", self.commit_spin)
        form.addRow(QLabel("<small>识别间隔越小越实时（但更吃GPU）；句末静音越小定稿越快</small>"))
        return w

    # ---------- 翻译 ----------
    def _build_translate_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        self.tr_enabled = QCheckBox("启用大模型翻译")
        self.tr_enabled.setChecked(bool(self.cfg.get("translate.enabled", True)))
        form.addRow(self.tr_enabled)

        self.preset_combo = QComboBox()
        self.preset_combo.addItem("自定义")
        for name in PRESETS:
            self.preset_combo.addItem(name)
        self.preset_combo.currentTextChanged.connect(self._apply_preset)
        form.addRow("服务商预设：", self.preset_combo)

        self.base_edit = QLineEdit(self.cfg.get("translate.base_url", ""))
        form.addRow("API Base URL：", self.base_edit)

        self.key_edit = QLineEdit(self.cfg.get("translate.api_key", ""))
        self.key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("API Key：", self.key_edit)

        self.model_edit = QLineEdit(self.cfg.get("translate.model", ""))
        form.addRow("模型名：", self.model_edit)

        self.target_combo = QComboBox()
        for code, name in [("zh", "中文"), ("en", "英文"), ("ja", "日文"), ("ko", "韩文")]:
            self.target_combo.addItem(name, code)
        idx = self.target_combo.findData(self.cfg.get("translate.target_lang", "zh"))
        self.target_combo.setCurrentIndex(idx if idx >= 0 else 0)
        form.addRow("翻译为：", self.target_combo)

        test_btn = QPushButton("测试连接")
        test_btn.clicked.connect(self._test_translate)
        form.addRow(test_btn)
        self.test_label = QLabel("")
        self.test_label.setWordWrap(True)
        form.addRow(self.test_label)

        self.speculative = QCheckBox("增量预览翻译（边听边出译文，延迟最低）")
        self.speculative.setChecked(bool(self.cfg.get("translate.speculative", True)))
        form.addRow(self.speculative)
        self.stream_check = QCheckBox("流式翻译（逐字上屏）")
        self.stream_check.setChecked(bool(self.cfg.get("translate.stream", True)))
        form.addRow(self.stream_check)
        return w

    def _apply_preset(self, name: str) -> None:
        if name in PRESETS:
            base, model = PRESETS[name]
            self.base_edit.setText(base)
            self.model_edit.setText(model)

    def _test_translate(self) -> None:
        self.test_label.setText("测试中...")
        tr = Translator(
            base_url=self.base_edit.text().strip(),
            api_key=self.key_edit.text().strip(),
            model=self.model_edit.text().strip(),
            target_lang=self.target_combo.currentData() or "zh",
        )
        ok, msg = tr.test_connection()
        color = "#2ecc71" if ok else "#e74c3c"
        self.test_label.setStyleSheet(f"color:{color};")
        self.test_label.setText(msg)

    # ---------- 字幕 ----------
    def _build_subtitle_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        self.font_size = QSpinBox()
        self.font_size.setRange(12, 48)
        self.font_size.setValue(int(self.cfg.get("subtitle.font_size", 20)))
        form.addRow("译文字号：", self.font_size)

        self.opacity = QDoubleSpinBox()
        self.opacity.setRange(0.0, 1.0)
        self.opacity.setSingleStep(0.05)
        self.opacity.setValue(float(self.cfg.get("subtitle.bg_opacity", 0.55)))
        form.addRow("背景不透明度：", self.opacity)

        self.width_ratio = QDoubleSpinBox()
        self.width_ratio.setRange(0.3, 1.0)
        self.width_ratio.setSingleStep(0.05)
        self.width_ratio.setValue(float(self.cfg.get("subtitle.width_ratio", 0.7)))
        form.addRow("宽度占比：", self.width_ratio)

        self.bottom_margin = QSpinBox()
        self.bottom_margin.setRange(0, 600)
        self.bottom_margin.setValue(int(self.cfg.get("subtitle.bottom_margin", 90)))
        form.addRow("距底部像素：", self.bottom_margin)

        self.show_source = QCheckBox("同时显示原文")
        self.show_source.setChecked(bool(self.cfg.get("subtitle.show_source", True)))
        form.addRow(self.show_source)

        self.click_through = QCheckBox("鼠标穿透（不挡操作，但无法拖动窗口）")
        self.click_through.setChecked(bool(self.cfg.get("subtitle.click_through", False)))
        form.addRow(self.click_through)
        return w

    # ---------- 保存 ----------
    def _save(self) -> None:
        self.cfg.update_many(
            {
                "audio.device": self.device_combo.currentData() or self.device_combo.currentText(),
                "asr.model": self.model_combo.currentText(),
                "asr.device": self.device_combo2.currentText(),
                "asr.compute_type": self.ctype_combo.currentText(),
                "asr.language": self.lang_combo.currentData(),
                "asr.initial_prompt": self.prompt_edit.text().strip(),
                "streaming.enabled": self.stream_enabled.isChecked(),
                "streaming.interval_ms": self.interval_spin.value(),
                "streaming.commit_silence_ms": self.commit_spin.value(),
                "translate.enabled": self.tr_enabled.isChecked(),
                "translate.base_url": self.base_edit.text().strip(),
                "translate.api_key": self.key_edit.text().strip(),
                "translate.model": self.model_edit.text().strip(),
                "translate.target_lang": self.target_combo.currentData(),
                "translate.stream": self.stream_check.isChecked(),
                "translate.speculative": self.speculative.isChecked(),
                "subtitle.font_size": self.font_size.value(),
                "subtitle.bg_opacity": round(self.opacity.value(), 2),
                "subtitle.width_ratio": round(self.width_ratio.value(), 2),
                "subtitle.bottom_margin": self.bottom_margin.value(),
                "subtitle.show_source": self.show_source.isChecked(),
                "subtitle.click_through": self.click_through.isChecked(),
            }
        )
        QMessageBox.information(self, "已保存", "设置已保存，部分项在下次开始翻译时生效。")
        self.settings_saved.emit()
        self.accept()

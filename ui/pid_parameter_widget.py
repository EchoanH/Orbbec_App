"""无相机、跟踪器、PID 或串口所有权的共享 PID 参数控件。"""

from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from gimbal.pid import PIDDiagnostics
from gimbal.pid_config import (
    DEFAULT_CONFIG_PATH,
    PIDConfigError,
    get_default_pid_config,
    load_pid_config,
    save_pid_config,
    validate_pid_config,
)


PAN_SIGN = -1
TILT_SIGN = -1

_UI_TO_CONFIG_KEYS = {
    "pan_kp": "pan_kp",
    "pan_ki": "pan_ki",
    "pan_kd": "pan_kd",
    "tilt_kp": "tilt_kp",
    "tilt_ki": "tilt_ki",
    "tilt_kd": "tilt_kd",
    "deadzone_x": "deadzone_x",
    "deadzone_y": "deadzone_y",
    "control_interval": "control_interval_s",
    "max_jog": "max_jog_deg",
    "integral_limit": "integral_limit",
}


class ParameterControl(QWidget):
    value_changed = pyqtSignal(float)

    def __init__(self, title, minimum, maximum, value, decimals, step,
                 parent=None):
        super(ParameterControl, self).__init__(parent)
        self._scale = 10 ** int(decimals)
        title_label = QLabel(title)
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(
            int(round(minimum * self._scale)),
            int(round(maximum * self._scale)),
        )
        self._spinbox = QDoubleSpinBox()
        self._spinbox.setDecimals(decimals)
        self._spinbox.setRange(minimum, maximum)
        self._spinbox.setSingleStep(step)
        self._spinbox.setFixedWidth(88)
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 1, 0, 1)
        layout.addWidget(title_label, 0, 0, 1, 2)
        layout.addWidget(self._slider, 1, 0)
        layout.addWidget(self._spinbox, 1, 1)
        self._slider.valueChanged.connect(self._on_slider_changed)
        self._spinbox.valueChanged.connect(self._on_spinbox_changed)
        self.set_value(value)

    def value(self):
        return float(self._spinbox.value())

    def set_value(self, value):
        self._spinbox.setValue(float(value))

    def _on_slider_changed(self, value):
        spin_value = value / float(self._scale)
        self._spinbox.blockSignals(True)
        self._spinbox.setValue(spin_value)
        self._spinbox.blockSignals(False)
        self.value_changed.emit(spin_value)

    def _on_spinbox_changed(self, value):
        self._slider.blockSignals(True)
        self._slider.setValue(int(round(value * self._scale)))
        self._slider.blockSignals(False)
        self.value_changed.emit(float(value))


class AxisDiagnostics(QGroupBox):
    FIELD_NAMES = (
        ("error", "当前误差"),
        ("p_term", "比例项 P"),
        ("i_term", "积分项 I"),
        ("d_term", "微分项 D"),
        ("raw_output", "PID 原始输出"),
        ("output", "PID 限幅输出"),
        ("accumulator", "小数累积量"),
        ("jog", "实际 JOG 指令"),
    )

    def __init__(self, title, parent=None):
        super(AxisDiagnostics, self).__init__(title, parent)
        self._values = {}
        grid = QGridLayout(self)
        grid.setVerticalSpacing(3)
        for row, (key, caption) in enumerate(self.FIELD_NAMES):
            grid.addWidget(QLabel(caption), row, 0)
            value = QLabel("0.000")
            value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            value.setStyleSheet("color: #9fe8cf; font-family: monospace;")
            grid.addWidget(value, row, 1)
            self._values[key] = value

    def update_sample(self, sample, final_jog):
        self._values["error"].setText("%+.4f" % sample.error)
        self._values["p_term"].setText("%+.4f" % sample.p_term)
        self._values["i_term"].setText("%+.4f" % sample.i_term)
        self._values["d_term"].setText("%+.4f" % sample.d_term)
        self._values["raw_output"].setText("%+.4f" % sample.raw_output)
        self._values["output"].setText("%+.4f" % sample.output)
        self._values["accumulator"].setText("%+.4f" % sample.accumulator)
        self._values["jog"].setText("%+d°" % int(final_jog))

    def set_accumulator(self, value):
        self._values["accumulator"].setText("%+.4f" % float(value))

    def reset_values(self):
        self.update_sample(PIDDiagnostics(), 0)


class PIDParameterWidget(QWidget):
    """只管理 PID 参数显示、配置持久化和诊断值。"""

    values_changed = pyqtSignal(object)
    reset_requested = pyqtSignal()
    configuration_action_started = pyqtSignal()
    feedback_changed = pyqtSignal(str)
    config_source_changed = pyqtSignal(str)

    def __init__(self, config_path=None, initial_values=None,
                 initial_source=None, auto_load=True, parent=None):
        super(PIDParameterWidget, self).__init__(parent)
        self._config_path = config_path
        self._applying = False
        self.parameters = {}
        self._build_ui()
        if initial_values is not None:
            self.apply_pid_config(
                initial_values,
                source_text=initial_source or "当前参数来源：运行中参数",
                emit_change=False)
        elif auto_load:
            self.load_saved_parameters(announce=False)
        else:
            self.apply_pid_config(
                get_default_pid_config(),
                source_text="当前参数来源：默认配置",
                emit_change=False)

    @property
    def config_path(self):
        path = (Path(self._config_path) if self._config_path is not None
                else DEFAULT_CONFIG_PATH)
        return path

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        defaults = get_default_pid_config()
        pan_group = QGroupBox("水平轴 PAN PID 参数")
        pan_layout = QVBoxLayout(pan_group)
        tilt_group = QGroupBox("俯仰轴 TILT PID 参数")
        tilt_layout = QVBoxLayout(tilt_group)
        specs = (
            ("pan_kp", "水平轴比例系数 Kp", 0.0, 10.0, 2, 0.05,
             pan_layout),
            ("pan_ki", "水平轴积分系数 Ki", 0.0, 2.0, 3, 0.01,
             pan_layout),
            ("pan_kd", "水平轴微分系数 Kd", 0.0, 2.0, 3, 0.01,
             pan_layout),
            ("tilt_kp", "俯仰轴比例系数 Kp", 0.0, 10.0, 2, 0.05,
             tilt_layout),
            ("tilt_ki", "俯仰轴积分系数 Ki", 0.0, 2.0, 3, 0.01,
             tilt_layout),
            ("tilt_kd", "俯仰轴微分系数 Kd", 0.0, 2.0, 3, 0.01,
             tilt_layout),
        )
        for key, title, minimum, maximum, decimals, step, layout in specs:
            control = ParameterControl(
                title, minimum, maximum, defaults[key], decimals, step)
            control.value_changed.connect(self._on_value_changed)
            self.parameters[key] = control
            layout.addWidget(control)
        axis_row = QHBoxLayout()
        axis_row.addWidget(pan_group)
        axis_row.addWidget(tilt_group)
        root.addLayout(axis_row)

        common_group = QGroupBox("公共控制参数（未启用微分低通滤波）")
        common_layout = QVBoxLayout(common_group)
        common_specs = (
            ("deadzone_x", "水平死区", 0.0, 0.30, 3, 0.01),
            ("deadzone_y", "垂直死区", 0.0, 0.30, 3, 0.01),
            ("control_interval", "控制周期（秒）", 0.08, 0.50, 3, 0.01),
            ("max_jog", "最大单次转角（°）", 1.0, 5.0, 0, 1.0),
            ("integral_limit", "积分限幅", 0.0, 5.0, 2, 0.1),
        )
        for key, title, minimum, maximum, decimals, step in common_specs:
            config_key = _UI_TO_CONFIG_KEYS[key]
            control = ParameterControl(
                title, minimum, maximum, defaults[config_key], decimals, step)
            control.value_changed.connect(self._on_value_changed)
            self.parameters[key] = control
            common_layout.addWidget(control)
        root.addWidget(common_group)

        diagnostics_row = QHBoxLayout()
        self.pan_diagnostics = AxisDiagnostics("水平轴 PAN 实时诊断")
        self.tilt_diagnostics = AxisDiagnostics("俯仰轴 TILT 实时诊断")
        diagnostics_row.addWidget(self.pan_diagnostics)
        diagnostics_row.addWidget(self.tilt_diagnostics)
        root.addLayout(diagnostics_row)

        export_group = QGroupBox("当前最终参数（保存后供动态跟踪使用）")
        export_layout = QVBoxLayout(export_group)
        self.parameter_text = QPlainTextEdit()
        self.parameter_text.setReadOnly(True)
        self.parameter_text.setMaximumHeight(190)
        self.config_source_label = QLabel("当前参数来源：默认配置")
        self.config_source_label.setWordWrap(True)
        self.config_source_label.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.config_source_label.setStyleSheet("color: #9fe8cf;")
        self.feedback_label = QLabel("")
        self.feedback_label.setWordWrap(True)
        self.feedback_label.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Preferred)
        button_grid = QGridLayout()
        self.save_button = QPushButton("保存当前参数")
        self.save_button.setStyleSheet(
            "background: #2f9e72; color: white; font-weight: 700; padding: 8px;")
        self.restore_button = QPushButton("恢复已保存参数")
        self.default_button = QPushButton("恢复默认参数")
        self.reset_button = QPushButton("PID 状态清零")
        copy_button = QPushButton("复制参数")
        self.save_button.clicked.connect(self.save_current_parameters)
        self.restore_button.clicked.connect(self.load_saved_parameters)
        self.default_button.clicked.connect(self.restore_default_parameters)
        self.reset_button.clicked.connect(self._request_reset)
        copy_button.clicked.connect(self._copy_parameters)
        for index, button in enumerate((
                self.save_button, self.restore_button,
                self.default_button, self.reset_button, copy_button)):
            button_grid.addWidget(button, index // 2, index % 2)
        export_layout.addWidget(self.parameter_text)
        export_layout.addWidget(self.config_source_label)
        export_layout.addWidget(self.feedback_label)
        export_layout.addLayout(button_grid)
        root.addWidget(export_group)

    def value(self, key):
        return self.parameters[key].value()

    def current_pid_config(self):
        values = {
            config_key: self.value(ui_key)
            for ui_key, config_key in _UI_TO_CONFIG_KEYS.items()
        }
        values["pan_sign"] = PAN_SIGN
        values["tilt_sign"] = TILT_SIGN
        return validate_pid_config(values)

    def apply_pid_config(self, values, source_text=None, emit_change=True):
        values = validate_pid_config(values)
        self._applying = True
        try:
            for ui_key, config_key in _UI_TO_CONFIG_KEYS.items():
                self.parameters[ui_key].set_value(values[config_key])
        finally:
            self._applying = False
        self._update_parameter_text()
        if source_text is not None:
            self._set_source(source_text)
        if emit_change:
            self.values_changed.emit(dict(values))

    def load_saved_parameters(self, _checked=False, announce=True):
        self.configuration_action_started.emit()
        result = load_pid_config(self._config_path)
        source_text = (
            "当前参数来源：已保存配置"
            if result.source == "saved"
            else "当前参数来源：默认配置")
        self.apply_pid_config(result.values, source_text=source_text)
        if result.error:
            self._set_feedback(result.error)
        elif announce:
            self._set_feedback(source_text)
        return result

    def restore_default_parameters(self, _checked=False):
        self.configuration_action_started.emit()
        source_text = "当前参数来源：默认配置（尚未保存）"
        self.apply_pid_config(
            get_default_pid_config(), source_text=source_text)
        self._set_feedback("已恢复默认参数；点击保存后写入配置")

    def save_current_parameters(self, _checked=False):
        self.configuration_action_started.emit()
        try:
            path = save_pid_config(
                self.current_pid_config(), self._config_path).resolve()
        except PIDConfigError as exc:
            self._set_feedback("参数保存失败：%s" % exc)
            return False
        self._set_source("当前参数来源：已保存配置")
        self._set_feedback("参数保存成功\n配置文件：%s" % path)
        return True

    def update_diagnostics(self, pan_sample, pan_jog,
                           tilt_sample, tilt_jog):
        self.pan_diagnostics.update_sample(pan_sample, pan_jog)
        self.tilt_diagnostics.update_sample(tilt_sample, tilt_jog)

    def set_accumulator(self, axis_name, value):
        if axis_name == "PAN":
            self.pan_diagnostics.set_accumulator(value)
        elif axis_name == "TILT":
            self.tilt_diagnostics.set_accumulator(value)

    def reset_diagnostics(self):
        self.pan_diagnostics.reset_values()
        self.tilt_diagnostics.reset_values()

    def _on_value_changed(self, _value):
        self._update_parameter_text()
        if not self._applying:
            self.values_changed.emit(self.current_pid_config())

    def _parameter_export_text(self):
        values = self.current_pid_config()
        return (
            "PAN_KP = %.3f\nPAN_KI = %.3f\nPAN_KD = %.3f\n"
            "TILT_KP = %.3f\nTILT_KI = %.3f\nTILT_KD = %.3f\n"
            "GIMBAL_DEADZONE_X = %.3f\nGIMBAL_DEADZONE_Y = %.3f\n"
            "GIMBAL_CONTROL_INTERVAL_S = %.3f\n"
            "GIMBAL_MAX_JOG_DEG = %.1f\nGIMBAL_INTEGRAL_LIMIT = %.3f"
        ) % (
            values["pan_kp"], values["pan_ki"], values["pan_kd"],
            values["tilt_kp"], values["tilt_ki"], values["tilt_kd"],
            values["deadzone_x"], values["deadzone_y"],
            values["control_interval_s"], values["max_jog_deg"],
            values["integral_limit"],
        )

    def _update_parameter_text(self):
        self.parameter_text.setPlainText(self._parameter_export_text())

    def _set_source(self, source_text):
        self.config_source_label.setText(
            "%s\n配置文件：%s" %
            (source_text, self.config_path.resolve()))
        self.config_source_changed.emit(source_text)

    def _set_feedback(self, text):
        self.feedback_label.setText(text)
        self.feedback_changed.emit(text)

    def _copy_parameters(self):
        QApplication.clipboard().setText(self.parameter_text.toPlainText())
        self._set_feedback("参数已复制；复制操作未保存配置")

    def _request_reset(self):
        self.reset_diagnostics()
        self.reset_requested.emit()
        self._set_feedback("PID 状态与小数累计器已清零")


class PIDParameterDialog(QDialog):
    """动态跟踪页使用的单例非模态参数工具窗口。"""

    def __init__(self, config_path=None, initial_values=None,
                 initial_source=None, parent=None):
        super(PIDParameterDialog, self).__init__(parent)
        self.setWindowTitle("动态目标跟踪 · PID 调试")
        self.setWindowFlags(self.windowFlags() | Qt.Tool)
        self.setModal(False)
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.resize(680, 820)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.parameter_widget = PIDParameterWidget(
            config_path=config_path,
            initial_values=initial_values,
            initial_source=initial_source,
            auto_load=initial_values is None,
            parent=self)
        scroll.setWidget(self.parameter_widget)
        layout.addWidget(scroll)

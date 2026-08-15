"""主 GUI 与独立入口共用的云台 PID 调试页面。"""

import time
from collections import deque

import cv2
from PyQt5.QtCore import QEvent, Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from gimbal.controller import GimbalWorker
from gimbal.pid import (
    PIDAxis,
    PIDDiagnostics,
    parse_gimbal_angles,
    safe_jog_for_angle,
)
from inference.target_tracker import TargetTracker, normalized_bbox_center

from .base_page import BasePage


PAN_SIGN = -1
TILT_SIGN = -1

PAN_SAFE_MIN = 60.0
PAN_SAFE_MAX = 120.0
TILT_SAFE_MIN = 60.0
TILT_SAFE_MAX = 120.0
TRACKING_FRAME_TIMEOUT_S = 0.75

INITIAL_VALUES = {
    "pan_kp": 4.0,
    "pan_ki": 0.0,
    "pan_kd": 0.25,
    "tilt_kp": 4.0,
    "tilt_ki": 0.0,
    "tilt_kd": 0.25,
    "deadzone_x": 0.13,
    "deadzone_y": 0.13,
    "control_interval": 0.18,
    "max_jog": 2.0,
    "integral_limit": 1.0,
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


class GimbalPIDTunerPage(BasePage):
    page_title = "云台 PID 调试"
    page_hint = "框选目标 · 运行时 PID 调参 · 软件安全限位"

    def __init__(self, parent=None, gimbal_worker_factory=None):
        super(GimbalPIDTunerPage, self).__init__(parent)
        self._gimbal_worker_factory = (
            gimbal_worker_factory if gimbal_worker_factory is not None
            else GimbalWorker)
        self._active = False
        self._tracker = TargetTracker()
        self._pan_pid = PIDAxis()
        self._tilt_pid = PIDAxis()
        self._latest_bgr = None
        self._source_size = None
        self._selecting = False
        self._selection_start = None
        self._selection_current = None
        self._normalized_center = None
        self._tracking_state = TargetTracker.IDLE
        self._frame_times = deque(maxlen=10)
        self._display_fps = 0.0
        self._last_frame_time = None

        self._gimbal_worker = None
        self._gimbal_connected = False
        self._gimbal_status = "云台未连接"
        self._pan_angle = None
        self._tilt_angle = None
        self._auto_enabled = False
        self._pending_auto = {}
        self._last_pid_time = None
        self._last_pan_sample = PIDDiagnostics()
        self._last_tilt_sample = PIDDiagnostics()

        self._build_tuner_ui()
        self._control_timer = QTimer(self)
        self._control_timer.setTimerType(Qt.PreciseTimer)
        self._control_timer.timeout.connect(self._control_tick)
        self._parameters_changed()
        self.video_label.setMouseTracking(True)
        self.video_label.setCursor(Qt.CrossCursor)
        self.video_label.installEventFilter(self)
        self._update_start_enabled()
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._shutdown_gimbal)

    @property
    def gimbal_worker(self):
        return self._gimbal_worker

    @property
    def auto_enabled(self):
        return self._auto_enabled

    def _build_tuner_ui(self):
        base_layout = self.layout()
        video_panel = base_layout.takeAt(0).widget()
        target_status = QLabel("未选择目标 · 归一化 X -- · Y -- · 帧率 0.0")
        target_status.setObjectName("resultLabel")
        target_status.setFixedHeight(34)
        target_status.setStyleSheet("font-size: 13px; padding: 5px;")
        video_panel.layout().addWidget(target_status)
        self.target_status = target_status

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(video_panel)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(620)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right = QWidget()
        self.right_layout = QVBoxLayout(right)
        self.right_layout.setContentsMargins(8, 0, 8, 0)
        scroll.setWidget(right)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        base_layout.addWidget(splitter, 1)

        self._build_action_buttons()
        self._build_parameter_controls()
        self._build_diagnostics()
        self._build_parameter_export()
        self.right_layout.addStretch(1)

    def _build_action_buttons(self):
        group = QGroupBox("控制与安全")
        grid = QGridLayout(group)
        self.start_button = QPushButton("开始自动控制")
        self.stop_button = QPushButton("立即停止")
        self.center_button = QPushButton("云台回中")
        self.clear_button = QPushButton("清除目标")
        self.reset_button = QPushButton("PID状态清零")
        self.get_button = QPushButton("读取云台角度")
        self.start_button.setStyleSheet(
            "background: #2f9e72; color: white; font-weight: 700; padding: 9px;")
        self.stop_button.setStyleSheet(
            "background: #c83d4d; color: white; font-weight: 800; padding: 9px;")
        for index, button in enumerate((
                self.start_button, self.stop_button, self.center_button,
                self.clear_button, self.reset_button, self.get_button)):
            grid.addWidget(button, index // 2, index % 2)
        self.start_button.clicked.connect(self._start_auto_control)
        self.stop_button.clicked.connect(
            lambda: self._stop_auto_control("用户立即停止"))
        self.center_button.clicked.connect(self._center_gimbal)
        self.clear_button.clicked.connect(self.clear_target)
        self.reset_button.clicked.connect(self._reset_pid_state)
        self.get_button.clicked.connect(self._request_get)

        self.angle_label = QLabel(
            "STM32：水平角=--  俯仰角=--\n"
            "安全范围：水平轴 PAN 60～120° · 俯仰轴 TILT 60～120°\n"
            "方向：PAN=-1 / TILT=-1 · 画面超时保护：0.75 秒")
        self.angle_label.setWordWrap(True)
        self.angle_label.setStyleSheet("color: #9fe8cf; padding: 4px;")
        grid.addWidget(self.angle_label, 3, 0, 1, 2)
        self.control_status = QLabel("自动控制：关闭")
        self.control_status.setWordWrap(True)
        grid.addWidget(self.control_status, 4, 0, 1, 2)
        self.right_layout.addWidget(group)

    def _build_parameter_controls(self):
        self.parameters = {}
        pan_group = QGroupBox("水平轴 PAN PID 参数")
        pan_layout = QVBoxLayout(pan_group)
        tilt_group = QGroupBox("俯仰轴 TILT PID 参数")
        tilt_layout = QVBoxLayout(tilt_group)
        specs = (
            ("pan_kp", "水平轴比例系数 Kp", 0.0, 10.0, 2, 0.05, pan_layout),
            ("pan_ki", "水平轴积分系数 Ki", 0.0, 2.0, 3, 0.01, pan_layout),
            ("pan_kd", "水平轴微分系数 Kd", 0.0, 2.0, 3, 0.01, pan_layout),
            ("tilt_kp", "俯仰轴比例系数 Kp", 0.0, 10.0, 2, 0.05, tilt_layout),
            ("tilt_ki", "俯仰轴积分系数 Ki", 0.0, 2.0, 3, 0.01, tilt_layout),
            ("tilt_kd", "俯仰轴微分系数 Kd", 0.0, 2.0, 3, 0.01, tilt_layout),
        )
        for key, title, minimum, maximum, decimals, step, layout in specs:
            control = ParameterControl(
                title, minimum, maximum, INITIAL_VALUES[key], decimals, step)
            control.value_changed.connect(self._parameters_changed)
            self.parameters[key] = control
            layout.addWidget(control)
        row = QHBoxLayout()
        row.addWidget(pan_group)
        row.addWidget(tilt_group)
        self.right_layout.addLayout(row)

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
            control = ParameterControl(
                title, minimum, maximum, INITIAL_VALUES[key], decimals, step)
            control.value_changed.connect(self._parameters_changed)
            self.parameters[key] = control
            common_layout.addWidget(control)
        self.right_layout.addWidget(common_group)

    def _build_diagnostics(self):
        self.pan_diagnostics = AxisDiagnostics("水平轴 PAN 实时诊断")
        self.tilt_diagnostics = AxisDiagnostics("俯仰轴 TILT 实时诊断")
        self.right_layout.addWidget(self.pan_diagnostics)
        self.right_layout.addWidget(self.tilt_diagnostics)

    def _build_parameter_export(self):
        group = QGroupBox("当前最终参数（只读，不写正式页面）")
        layout = QVBoxLayout(group)
        self.parameter_text = QPlainTextEdit()
        self.parameter_text.setReadOnly(True)
        self.parameter_text.setMaximumHeight(235)
        copy_button = QPushButton("复制参数")
        copy_button.clicked.connect(self._copy_parameters)
        layout.addWidget(self.parameter_text)
        layout.addWidget(copy_button)
        self.right_layout.addWidget(group)

    def _value(self, key):
        return self.parameters[key].value()

    def _parameters_changed(self, _value=None):
        if "control_interval" not in self.parameters:
            return
        interval_ms = max(
            1, int(round(self._value("control_interval") * 1000.0)))
        self._control_timer.setInterval(interval_ms)
        self.parameter_text.setPlainText(self._parameter_export_text())

    def _parameter_export_text(self):
        return (
            "PAN_KP = %.3f\nPAN_KI = %.3f\nPAN_KD = %.3f\n"
            "TILT_KP = %.3f\nTILT_KI = %.3f\nTILT_KD = %.3f\n"
            "GIMBAL_DEADZONE_X = %.3f\nGIMBAL_DEADZONE_Y = %.3f\n"
            "GIMBAL_CONTROL_INTERVAL_S = %.3f\n"
            "GIMBAL_MAX_JOG_DEG = %.1f\nGIMBAL_INTEGRAL_LIMIT = %.3f"
        ) % (
            self._value("pan_kp"), self._value("pan_ki"),
            self._value("pan_kd"), self._value("tilt_kp"),
            self._value("tilt_ki"), self._value("tilt_kd"),
            self._value("deadzone_x"), self._value("deadzone_y"),
            self._value("control_interval"), self._value("max_jog"),
            self._value("integral_limit"),
        )

    def _copy_parameters(self):
        QApplication.clipboard().setText(self.parameter_text.toPlainText())
        self.control_status.setText("参数已复制；正式页面未被修改")

    def process_frame(self, bgr_frame, bgr_display=None, depth_frame=None):
        now = time.monotonic()
        self._last_frame_time = now
        self._frame_times.append(now)
        if len(self._frame_times) >= 2:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            if elapsed > 0.0:
                self._display_fps = (len(self._frame_times) - 1) / elapsed
        self._latest_bgr = bgr_frame
        source_height, source_width = bgr_frame.shape[:2]
        self._source_size = (source_width, source_height)

        bbox = None
        if self._active and self._tracker.state == TargetTracker.TRACKING:
            bbox = self._tracker.update(bgr_frame)
            if bbox is None:
                self._tracking_state = TargetTracker.LOST
                self._normalized_center = None
                self._stop_auto_control(
                    "目标丢失：已停止自动输出并清空 PID")
            else:
                self._tracking_state = TargetTracker.TRACKING
                self._normalized_center = normalized_bbox_center(
                    bbox, bgr_frame.shape)
        else:
            self._tracking_state = self._tracker.state

        display = bgr_display if bgr_display is not None else bgr_frame
        display_height, display_width = display.shape[:2]
        target_width, target_height, target_scale = self.compute_target_size(
            display_width, display_height)
        if target_scale < 1.0:
            rendered = cv2.resize(
                display, (target_width, target_height),
                interpolation=cv2.INTER_LINEAR)
        else:
            rendered = display
        has_overlay = (
            bbox is not None or
            (self._selecting and self._selection_start is not None and
             self._selection_current is not None))
        if has_overlay and rendered is display:
            rendered = display.copy()
        scale_x = rendered.shape[1] / float(source_width)
        scale_y = rendered.shape[0] / float(source_height)
        if bbox is not None:
            x, y, width, height = bbox
            p1 = (int(round(x * scale_x)), int(round(y * scale_y)))
            p2 = (int(round((x + width) * scale_x)),
                  int(round((y + height) * scale_y)))
            cv2.rectangle(rendered, p1, p2, (0, 255, 0), 2)
            center = (
                int(round((x + width / 2.0) * scale_x)),
                int(round((y + height / 2.0) * scale_y)),
            )
            cv2.drawMarker(
                rendered, center, (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        if (self._selecting and self._selection_start is not None and
                self._selection_current is not None):
            x1, y1 = self._selection_start
            x2, y2 = self._selection_current
            cv2.rectangle(
                rendered,
                (int(round(x1 * scale_x)), int(round(y1 * scale_y))),
                (int(round(x2 * scale_x)), int(round(y2 * scale_y))),
                (0, 255, 255), 2,
            )
        self._update_target_status()
        self._update_start_enabled()
        return rendered, self._combined_status()

    def eventFilter(self, watched, event):
        if watched is self.video_label:
            if (event.type() == QEvent.MouseButtonPress and
                    event.button() == Qt.LeftButton):
                point = self._label_to_source(event.pos(), clamp=False)
                if point is None:
                    return True
                self._stop_auto_control("重新框选：已清空 PID")
                self._tracker.clear()
                self._normalized_center = None
                self._selecting = True
                self._selection_start = point
                self._selection_current = point
                self._tracking_state = TargetTracker.IDLE
                return True
            if event.type() == QEvent.MouseMove and self._selecting:
                point = self._label_to_source(event.pos(), clamp=True)
                if point is not None:
                    self._selection_current = point
                return True
            if (event.type() == QEvent.MouseButtonRelease and
                    event.button() == Qt.LeftButton and self._selecting):
                point = self._label_to_source(event.pos(), clamp=True)
                if point is not None:
                    self._selection_current = point
                self._finish_selection()
                return True
        return super(GimbalPIDTunerPage, self).eventFilter(watched, event)

    def _label_to_source(self, position, clamp):
        pixmap = self.video_label.pixmap()
        if (pixmap is None or pixmap.isNull() or
                self._source_size is None):
            return None
        content = self.video_label.contentsRect()
        offset_x = content.x() + (content.width() - pixmap.width()) / 2.0
        offset_y = content.y() + (content.height() - pixmap.height()) / 2.0
        image_x = float(position.x()) - offset_x
        image_y = float(position.y()) - offset_y
        if not clamp and (image_x < 0.0 or image_y < 0.0 or
                          image_x >= pixmap.width() or
                          image_y >= pixmap.height()):
            return None
        image_x = max(0.0, min(float(pixmap.width() - 1), image_x))
        image_y = max(0.0, min(float(pixmap.height() - 1), image_y))
        source_width, source_height = self._source_size
        return (
            image_x * source_width / float(pixmap.width()),
            image_y * source_height / float(pixmap.height()),
        )

    def _finish_selection(self):
        self._selecting = False
        if (self._latest_bgr is None or self._selection_start is None or
                self._selection_current is None):
            return
        start_x, start_y = self._selection_start
        end_x, end_y = self._selection_current
        roi = (min(start_x, end_x), min(start_y, end_y),
               abs(end_x - start_x), abs(end_y - start_y))
        initialized = self._tracker.initialize(self._latest_bgr, roi)
        self._selection_start = None
        self._selection_current = None
        if initialized:
            self._tracking_state = TargetTracker.TRACKING
            self._normalized_center = normalized_bbox_center(
                self._tracker.bbox, self._latest_bgr.shape)
            self.control_status.setText(
                "目标已选择；请手动点击开始自动控制")
        else:
            self._tracking_state = TargetTracker.IDLE
            self._normalized_center = None
            self.control_status.setText("框选无效：目标框至少 16×16 像素")
        self._update_target_status()
        self._update_start_enabled()

    def _update_target_status(self):
        if self._normalized_center is None:
            coordinates = "归一化 X -- · Y --"
        else:
            coordinates = "归一化 X %+.3f · Y %+.3f" % (
                self._normalized_center)
        state_text = {
            TargetTracker.IDLE: "未选择目标",
            TargetTracker.TRACKING: "跟踪中",
            TargetTracker.LOST: "目标丢失",
        }.get(self._tracking_state, str(self._tracking_state))
        self.target_status.setText(
            "%s · %s · 帧率 %.1f" % (
                state_text, coordinates, self._display_fps))

    def _combined_status(self):
        state_text = {
            TargetTracker.IDLE: "未选择目标",
            TargetTracker.TRACKING: "跟踪中",
            TargetTracker.LOST: "目标丢失",
        }.get(self._tracking_state, str(self._tracking_state))
        return "%s · %s" % (state_text, self._gimbal_status)

    def _start_gimbal_worker(self):
        if self._gimbal_worker is not None:
            if self._gimbal_worker.isRunning():
                return
            self._gimbal_worker.deleteLater()
            self._gimbal_worker = None
        worker = self._gimbal_worker_factory(parent=self)
        worker.connection_changed.connect(self._on_gimbal_connection)
        worker.status_changed.connect(self._on_gimbal_status)
        worker.response_received.connect(self._on_gimbal_response)
        worker.command_completed.connect(self._on_command_completed)
        worker.error_occurred.connect(self._on_gimbal_error)
        worker.finished.connect(self._on_gimbal_finished)
        self._gimbal_worker = worker
        worker.start()

    def _start_auto_control(self):
        if not self._active:
            return
        if self._tracker.state != TargetTracker.TRACKING:
            QMessageBox.warning(self, "无法启动", "请先在画面中框选目标。")
            return
        if not self._gimbal_connected:
            QMessageBox.warning(self, "无法启动", "云台串口未连接。")
            return
        if self._pan_angle is None or self._tilt_angle is None:
            QMessageBox.warning(
                self, "无法启动", "尚未获得真实角度，请先读取角度或执行云台回中。")
            return
        if not (PAN_SAFE_MIN <= self._pan_angle <= PAN_SAFE_MAX and
                TILT_SAFE_MIN <= self._tilt_angle <= TILT_SAFE_MAX):
            QMessageBox.warning(
                self, "无法启动", "当前角度位于软件安全范围外，请先处理或回中。")
            return
        self._reset_pid_state(update_status=False)
        self._auto_enabled = True
        self._gimbal_worker.set_auto_enabled(True)
        self._last_pid_time = time.monotonic()
        self.control_status.setText("自动控制：运行中")
        self._update_start_enabled()

    def _stop_auto_control(self, reason="自动控制：关闭"):
        self._auto_enabled = False
        worker = self._gimbal_worker
        if worker is not None:
            worker.set_auto_enabled(False)
        self._pending_auto.clear()
        self._last_pid_time = None
        self._reset_pid_state(update_status=False)
        self.control_status.setText(reason)
        self._update_start_enabled()

    def _reset_pid_state(self, update_status=True):
        self._pan_pid.reset()
        self._tilt_pid.reset()
        self._pending_auto.clear()
        self._last_pid_time = None
        self._last_pan_sample = PIDDiagnostics()
        self._last_tilt_sample = PIDDiagnostics()
        self.pan_diagnostics.reset_values()
        self.tilt_diagnostics.reset_values()
        if update_status:
            self.control_status.setText("PID 状态与小数累计器已清零")

    def clear_target(self):
        self._stop_auto_control("目标已清除；自动控制关闭")
        self._tracker.clear()
        self._normalized_center = None
        self._tracking_state = TargetTracker.IDLE
        self._selecting = False
        self._selection_start = None
        self._selection_current = None
        self._update_target_status()
        self._update_start_enabled()

    def _request_get(self):
        self._stop_auto_control("读取角度前已停止自动控制")
        worker = self._gimbal_worker
        if worker is not None and worker.isRunning():
            worker.request_get()

    def _center_gimbal(self):
        self._stop_auto_control("云台回中：已停止并清空 PID")
        worker = self._gimbal_worker
        if worker is not None and worker.isRunning():
            worker.request_center()

    def _control_tick(self):
        if not self._active or not self._auto_enabled or self._pending_auto:
            return
        if (self._tracker.state != TargetTracker.TRACKING or
                self._normalized_center is None):
            self._stop_auto_control("目标不可用：已停止自动输出")
            return
        if (not self._gimbal_connected or self._pan_angle is None or
                self._tilt_angle is None):
            self._stop_auto_control("角度或串口不可用：已立即停止")
            return
        now = time.monotonic()
        if (self._last_frame_time is None or
                now - self._last_frame_time > TRACKING_FRAME_TIMEOUT_S):
            self._stop_auto_control("跟踪画面超时：已立即停止")
            return
        if self._last_pid_time is None:
            self._last_pid_time = now
            return
        dt = now - self._last_pid_time
        self._last_pid_time = now
        nx, ny = self._normalized_center
        integral_limit = self._value("integral_limit")
        max_jog = self._value("max_jog")
        self._last_pan_sample = self._pan_pid.update(
            nx, dt, self._value("pan_kp"), self._value("pan_ki"),
            self._value("pan_kd"), self._value("deadzone_x"),
            integral_limit, max_jog, max_jog)
        self._last_tilt_sample = self._tilt_pid.update(
            ny, dt, self._value("tilt_kp"), self._value("tilt_ki"),
            self._value("tilt_kd"), self._value("deadzone_y"),
            integral_limit, max_jog, max_jog)
        pan_jog = self._safe_signed_jog(
            "PAN", self._last_pan_sample.jog, PAN_SIGN,
            self._pan_angle, PAN_SAFE_MIN, PAN_SAFE_MAX, self._pan_pid)
        tilt_jog = self._safe_signed_jog(
            "TILT", self._last_tilt_sample.jog, TILT_SIGN,
            self._tilt_angle, TILT_SAFE_MIN, TILT_SAFE_MAX, self._tilt_pid)
        self.pan_diagnostics.update_sample(self._last_pan_sample, pan_jog)
        self.tilt_diagnostics.update_sample(self._last_tilt_sample, tilt_jog)
        if pan_jog:
            self._pending_auto["PAN"] = (
                self._pan_pid, pan_jog * PAN_SIGN)
        if tilt_jog:
            self._pending_auto["TILT"] = (
                self._tilt_pid, tilt_jog * TILT_SIGN)
        worker = self._gimbal_worker
        if (pan_jog or tilt_jog) and worker is not None:
            worker.submit_auto_jog(pan_jog, tilt_jog)

    def _safe_signed_jog(self, axis_name, logical_jog, direction_sign,
                         current_angle, safe_min, safe_max, pid_axis):
        requested = int(logical_jog) * int(direction_sign)
        safe = safe_jog_for_angle(
            current_angle, requested, safe_min, safe_max)
        if requested and safe != requested:
            self.control_status.setText(
                "%s 软件安全限位生效：请求 %+d°，允许 %+d°" %
                (axis_name, requested, safe))
            if safe == 0:
                pid_axis.clear_accumulator()
        return safe

    def _on_command_completed(self, command, _response):
        parts = command.split()
        if len(parts) != 4 or parts[:2] != ["GIMBAL", "JOG"]:
            return
        axis_name = parts[2]
        pending = self._pending_auto.pop(axis_name, None)
        if pending is None:
            return
        pid_axis, logical_jog = pending
        pid_axis.consume_jog(logical_jog)
        if axis_name == "PAN":
            self.pan_diagnostics.set_accumulator(self._pan_pid.accumulator)
        elif axis_name == "TILT":
            self.tilt_diagnostics.set_accumulator(self._tilt_pid.accumulator)

    def _on_gimbal_connection(self, connected):
        self._gimbal_connected = bool(connected)
        if connected:
            self._gimbal_status = "云台已连接，等待角度"
        else:
            self._gimbal_status = "云台未连接"
            if self._active:
                self._stop_auto_control("串口断开：已立即停止")
        self._update_start_enabled()

    def _on_gimbal_status(self, text):
        self._gimbal_status = text

    def _on_gimbal_response(self, response):
        angles = parse_gimbal_angles(response)
        if angles is None:
            self._pan_angle = None
            self._tilt_angle = None
            self.angle_label.setText(
                "STM32：回复未包含水平角/俯仰角，自动控制已禁止\n%s" % response)
            if self._auto_enabled:
                self._stop_auto_control("角度解析失败：已立即停止")
        else:
            self._pan_angle, self._tilt_angle = angles
            self.angle_label.setText(
                "STM32：水平角=%.2f°  俯仰角=%.2f°\n"
                "安全范围：水平轴 PAN %.0f～%.0f° · "
                "俯仰轴 TILT %.0f～%.0f°\n"
                "方向：PAN=-1 / TILT=-1 · 画面超时保护：%.2f 秒" % (
                    self._pan_angle, self._tilt_angle,
                    PAN_SAFE_MIN, PAN_SAFE_MAX,
                    TILT_SAFE_MIN, TILT_SAFE_MAX,
                    TRACKING_FRAME_TIMEOUT_S))
        self._update_start_enabled()

    def _on_gimbal_error(self, text):
        self._pan_angle = None
        self._tilt_angle = None
        self._gimbal_connected = False
        self._gimbal_status = text
        self._stop_auto_control("串口异常：已立即停止自动控制")

    def _on_gimbal_finished(self):
        worker = self.sender()
        if worker is self._gimbal_worker:
            self._gimbal_worker = None
        worker.deleteLater()
        self._gimbal_connected = False
        self._update_start_enabled()

    def _update_start_enabled(self):
        ready = (
            self._active and not self._auto_enabled and
            self._tracker.state == TargetTracker.TRACKING and
            self._gimbal_connected and
            self._pan_angle is not None and
            self._tilt_angle is not None)
        self.start_button.setEnabled(ready)

    def on_camera_error(self, text):
        self._stop_auto_control("相机异常：已停止自动控制")
        self.video_label.setText("摄像头未连接\n\n%s" % text)

    def on_activated(self):
        if self._active:
            return
        self._active = True
        self._frame_times.clear()
        self._display_fps = 0.0
        self._last_frame_time = None
        self._control_timer.start()
        self._start_gimbal_worker()
        self.control_status.setText("自动控制：关闭；请先框选目标")
        self._update_start_enabled()

    def on_deactivated(self):
        self._active = False
        self._control_timer.stop()
        self.clear_target()
        self._shutdown_gimbal()

    def _shutdown_gimbal(self):
        self._stop_auto_control("页面停用：自动控制已停止")
        worker = self._gimbal_worker
        if worker is not None:
            worker.stop()
            if worker.isRunning():
                worker.wait()
            if worker is self._gimbal_worker:
                self._gimbal_worker = None
            worker.deleteLater()
        self._gimbal_connected = False
        self._gimbal_status = "云台未连接"
        self._pan_angle = None
        self._tilt_angle = None
        self._pending_auto.clear()
        self._update_start_enabled()

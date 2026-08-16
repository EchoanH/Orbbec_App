"""鼠标框选任意目标，并可选通过云台自动跟随。"""

import time

import cv2
from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtWidgets import (QApplication, QFrame, QHBoxLayout, QLabel,
                             QPushButton)

from gimbal.controller import GimbalWorker
from gimbal.pid import PIDAxis, parse_gimbal_angles, safe_jog_for_angle
from gimbal.pid_config import get_default_pid_config, load_pid_config
from inference.target_tracker import TargetTracker, normalized_bbox_center
from ui.draw_utils import draw_text_box_bgr

from .base_page import BasePage


PAN_SIGN = -1
TILT_SIGN = -1

PAN_SAFE_MIN = 60.0
PAN_SAFE_MAX = 120.0
TILT_SAFE_MIN = 60.0
TILT_SAFE_MAX = 120.0


class TargetTrackingPage(BasePage):
    page_title = "动态目标跟踪"
    page_hint = "鼠标框选目标 · 光流跟踪 · 可选云台跟随"

    def __init__(self, parent=None, gimbal_worker_factory=None,
                 config_path=None):
        super(TargetTrackingPage, self).__init__(parent)
        self._gimbal_worker_factory = (
            gimbal_worker_factory if gimbal_worker_factory is not None
            else GimbalWorker)
        self._config_path = config_path
        self._active = False
        self._tracker = TargetTracker()
        self._pan_pid = PIDAxis()
        self._tilt_pid = PIDAxis()
        self._pid_config = get_default_pid_config()
        self._config_status = "PID 参数：默认配置"
        self._latest_bgr = None
        self._source_size = None
        self._selecting = False
        self._selection_start = None
        self._selection_current = None
        self._normalized_center = None
        self._tracking_status = "未选择目标"
        self._gimbal_status = "云台未连接"
        self._gimbal_connected = False
        self._gimbal_worker = None
        self._pan_angle = None
        self._tilt_angle = None
        self._pending_auto = {}
        self._last_pid_time = None
        self._build_controls()
        self._load_pid_parameters()
        self.video_label.setMouseTracking(True)
        self.video_label.setCursor(Qt.CrossCursor)
        self.video_label.installEventFilter(self)
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._shutdown_gimbal)

    def _build_controls(self):
        panel = QFrame()
        panel.setObjectName("formPanel")
        controls = QHBoxLayout(panel)
        controls.setContentsMargins(16, 10, 16, 10)
        controls.setSpacing(10)

        hint = QLabel("在画面中按住鼠标左键拖框选择目标")
        hint.setObjectName("resultLabel")
        controls.addWidget(hint)
        controls.addStretch(1)

        self.coordinate_label = QLabel("目标坐标 X -- · Y --")
        self.coordinate_label.setObjectName("resultLabel")
        controls.addWidget(self.coordinate_label)

        self.clear_button = QPushButton("清除目标")
        self.clear_button.setObjectName("primaryButton")
        self.clear_button.clicked.connect(self.clear_target)
        controls.addWidget(self.clear_button)

        self.follow_button = QPushButton("云台跟随：关闭")
        self.follow_button.setObjectName("primaryButton")
        self.follow_button.setCheckable(True)
        self.follow_button.toggled.connect(self._on_follow_toggled)
        controls.addWidget(self.follow_button)

        self.center_button = QPushButton("云台回中")
        self.center_button.setObjectName("primaryButton")
        self.center_button.clicked.connect(self._center_gimbal)
        controls.addWidget(self.center_button)
        self.layout().addWidget(panel)

    def process_frame(self, bgr_frame, bgr_display=None, depth_frame=None):
        self._latest_bgr = bgr_frame
        source_height, source_width = bgr_frame.shape[:2]
        self._source_size = (source_width, source_height)
        display_frame = (bgr_display if bgr_display is not None
                         else bgr_frame)

        bbox = None
        if self._active and self._tracker.state == TargetTracker.TRACKING:
            bbox = self._tracker.update(bgr_frame)
            if bbox is None:
                self._tracking_status = "目标丢失"
                self._normalized_center = None
                self._stop_auto_control()
            else:
                self._normalized_center = normalized_bbox_center(
                    bbox, bgr_frame.shape)
                self._tracking_status = "跟踪中"
                self._maybe_control_gimbal(self._normalized_center)

        display_height, display_width = display_frame.shape[:2]
        target_width, target_height, target_scale = self.compute_target_size(
            display_width, display_height)
        if target_scale < 1.0:
            rendered = cv2.resize(
                display_frame, (target_width, target_height),
                interpolation=cv2.INTER_LINEAR)
        else:
            rendered = display_frame

        has_overlay = (bbox is not None or
                       (self._selecting and self._selection_start is not None and
                        self._selection_current is not None))
        if has_overlay and rendered is display_frame:
            rendered = display_frame.copy()

        scale_x = rendered.shape[1] / float(source_width)
        scale_y = rendered.shape[0] / float(source_height)
        if bbox is not None:
            x, y, width, height = bbox
            x1 = int(round(x * scale_x))
            y1 = int(round(y * scale_y))
            x2 = int(round((x + width) * scale_x))
            y2 = int(round((y + height) * scale_y))
            cv2.rectangle(rendered, (x1, y1), (x2, y2), (0, 255, 0), 2)
            center_x = int(round((x + width / 2.0) * scale_x))
            center_y = int(round((y + height / 2.0) * scale_y))
            cv2.drawMarker(
                rendered, (center_x, center_y), (0, 255, 0),
                markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

        if (self._selecting and self._selection_start is not None and
                self._selection_current is not None):
            start_x, start_y = self._selection_start
            current_x, current_y = self._selection_current
            cv2.rectangle(
                rendered,
                (int(round(start_x * scale_x)),
                 int(round(start_y * scale_y))),
                (int(round(current_x * scale_x)),
                 int(round(current_y * scale_y))),
                (0, 255, 255), 2)

        if self._normalized_center is not None:
            nx, ny = self._normalized_center
            coordinate_text = "目标坐标 X %.3f · Y %.3f" % (nx, ny)
            self._set_coordinate_text(coordinate_text)
            rendered = draw_text_box_bgr(
                rendered, coordinate_text, 20, 20, font_size=14,
                text_color=(8, 19, 31),
                background_color=(74, 158, 255))
        else:
            self._set_coordinate_text("目标坐标 X -- · Y --")
        return rendered, self._combined_status()

    def eventFilter(self, watched, event):
        if watched is self.video_label:
            if (event.type() == QEvent.MouseButtonPress and
                    event.button() == Qt.LeftButton):
                point = self._label_to_source(event.pos(), clamp=False)
                if point is None:
                    return True
                self._tracker.clear()
                self._normalized_center = None
                self._stop_auto_control()
                self._selecting = True
                self._selection_start = point
                self._selection_current = point
                self._tracking_status = "未选择目标"
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
        return super(TargetTrackingPage, self).eventFilter(watched, event)

    def _label_to_source(self, position, clamp):
        pixmap = self.video_label.pixmap()
        if (pixmap is None or pixmap.isNull() or
                self._source_size is None):
            return None
        content = self.video_label.contentsRect()
        pixmap_width = pixmap.width()
        pixmap_height = pixmap.height()
        offset_x = content.x() + (content.width() - pixmap_width) / 2.0
        offset_y = content.y() + (content.height() - pixmap_height) / 2.0
        image_x = float(position.x()) - offset_x
        image_y = float(position.y()) - offset_y
        if not clamp and (image_x < 0.0 or image_y < 0.0 or
                          image_x >= pixmap_width or
                          image_y >= pixmap_height):
            return None
        image_x = max(0.0, min(float(pixmap_width - 1), image_x))
        image_y = max(0.0, min(float(pixmap_height - 1), image_y))
        source_width, source_height = self._source_size
        source_x = image_x * source_width / float(pixmap_width)
        source_y = image_y * source_height / float(pixmap_height)
        return (max(0.0, min(float(source_width - 1), source_x)),
                max(0.0, min(float(source_height - 1), source_y)))

    def _finish_selection(self):
        self._selecting = False
        if (self._selection_start is None or
                self._selection_current is None or
                self._latest_bgr is None):
            self._selection_start = None
            self._selection_current = None
            return
        start_x, start_y = self._selection_start
        end_x, end_y = self._selection_current
        roi = (min(start_x, end_x), min(start_y, end_y),
               abs(end_x - start_x), abs(end_y - start_y))
        initialized = self._tracker.initialize(self._latest_bgr, roi)
        self._selection_start = None
        self._selection_current = None
        if initialized:
            self._tracking_status = "跟踪中"
            self._normalized_center = normalized_bbox_center(
                self._tracker.bbox, self._latest_bgr.shape)
        else:
            self._tracking_status = "未选择目标"
            self._normalized_center = None

    def clear_target(self):
        self._tracker.clear()
        self._selecting = False
        self._selection_start = None
        self._selection_current = None
        self._normalized_center = None
        self._tracking_status = "未选择目标"
        self._stop_auto_control()

    def _set_coordinate_text(self, text):
        if self.coordinate_label.text() != text:
            self.coordinate_label.setText(text)

    def _combined_status(self):
        return "%s · %s · %s" % (
            self._tracking_status, self._gimbal_status, self._config_status)

    def _start_gimbal_worker(self):
        if self._gimbal_worker is not None:
            if not self._gimbal_worker.isFinished():
                return
            self._gimbal_worker.deleteLater()
            self._gimbal_worker = None
        worker = self._gimbal_worker_factory(parent=self)
        worker.status_changed.connect(self._on_gimbal_status)
        worker.connection_changed.connect(self._on_gimbal_connection)
        worker.response_received.connect(self._on_gimbal_response)
        worker.command_completed.connect(self._on_command_completed)
        worker.error_occurred.connect(self._on_gimbal_error)
        worker.finished.connect(self._on_gimbal_finished)
        self._gimbal_worker = worker
        worker.start()

    def _on_gimbal_status(self, text):
        self._gimbal_status = text
        if text == "云台已连接":
            self._gimbal_connected = True

    def _on_gimbal_connection(self, connected):
        self._gimbal_connected = bool(connected)
        if connected:
            self._gimbal_status = "云台已连接，等待角度"
        elif (self._active and
              not self._gimbal_status.startswith("通信异常") and
              not self._gimbal_status.startswith("云台未连接")):
            self._gimbal_status = "云台未连接"
        if not connected:
            self._pan_angle = None
            self._tilt_angle = None
            self._stop_auto_control()

    def _on_gimbal_error(self, text):
        if text.startswith("云台未连接"):
            self._gimbal_status = "云台未连接"
        else:
            self._gimbal_status = text
        self._gimbal_connected = False
        self._pan_angle = None
        self._tilt_angle = None
        self._stop_auto_control()
        if self.follow_button.isChecked():
            self.follow_button.setChecked(False)

    def _on_gimbal_finished(self):
        worker = self.sender()
        if worker is self._gimbal_worker:
            self._gimbal_worker = None
        worker.deleteLater()
        self._pan_angle = None
        self._tilt_angle = None
        self._stop_auto_control()

    def _on_follow_toggled(self, enabled):
        self.follow_button.setText(
            "云台跟随：开启" if enabled else "云台跟随：关闭")
        if enabled:
            self._reset_pid_state()
            self._start_gimbal_worker()
            if (self._gimbal_worker is not None and self._gimbal_connected and
                    self._pan_angle is not None and
                    self._tilt_angle is not None and
                    self._tracker.state == TargetTracker.TRACKING):
                self._gimbal_worker.set_auto_enabled(True)
            elif self._pan_angle is None or self._tilt_angle is None:
                self._gimbal_status = "等待云台真实角度，尚未启动自动控制"
        else:
            self._stop_auto_control()

    def _center_gimbal(self):
        self._stop_auto_control()
        if self.follow_button.isChecked():
            self.follow_button.setChecked(False)
        self._start_gimbal_worker()
        if self._gimbal_worker is not None:
            self._gimbal_worker.request_center()

    def _maybe_control_gimbal(self, normalized_center):
        worker = self._gimbal_worker
        if (normalized_center is None or worker is None or
                not self.follow_button.isChecked() or
                not self._gimbal_connected or self._pending_auto or
                self._pan_angle is None or self._tilt_angle is None):
            return
        now = time.monotonic()
        if self._last_pid_time is None:
            self._last_pid_time = now
            return
        dt = now - self._last_pid_time
        if dt < self._pid_config["control_interval_s"]:
            return
        self._last_pid_time = now
        nx, ny = normalized_center
        integral_limit = self._pid_config["integral_limit"]
        max_jog = self._pid_config["max_jog_deg"]
        try:
            pan_sample = self._pan_pid.update(
                nx, dt, self._pid_config["pan_kp"],
                self._pid_config["pan_ki"], self._pid_config["pan_kd"],
                self._pid_config["deadzone_x"], integral_limit,
                max_jog, max_jog)
            tilt_sample = self._tilt_pid.update(
                ny, dt, self._pid_config["tilt_kp"],
                self._pid_config["tilt_ki"], self._pid_config["tilt_kd"],
                self._pid_config["deadzone_y"], integral_limit,
                max_jog, max_jog)
        except ValueError as exc:
            self._gimbal_status = "PID 参数异常：%s" % exc
            self._stop_auto_control()
            if self.follow_button.isChecked():
                self.follow_button.setChecked(False)
            return
        pan_delta = self._safe_signed_jog(
            "PAN", pan_sample.jog, PAN_SIGN, self._pan_angle,
            PAN_SAFE_MIN, PAN_SAFE_MAX, self._pan_pid)
        tilt_delta = self._safe_signed_jog(
            "TILT", tilt_sample.jog, TILT_SIGN, self._tilt_angle,
            TILT_SAFE_MIN, TILT_SAFE_MAX, self._tilt_pid)
        if pan_delta:
            self._pending_auto["PAN"] = (
                self._pan_pid, pan_delta * PAN_SIGN)
        if tilt_delta:
            self._pending_auto["TILT"] = (
                self._tilt_pid, tilt_delta * TILT_SIGN)
        if pan_delta == 0 and tilt_delta == 0:
            return
        worker.set_auto_enabled(True)
        worker.submit_auto_jog(pan_delta, tilt_delta)

    def _safe_signed_jog(self, axis_name, logical_jog, direction_sign,
                         current_angle, safe_min, safe_max, pid_axis):
        requested = int(logical_jog) * int(direction_sign)
        safe = safe_jog_for_angle(
            current_angle, requested, safe_min, safe_max)
        if requested and safe != requested:
            self._gimbal_status = (
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

    def _on_gimbal_response(self, response):
        angles = parse_gimbal_angles(response)
        if angles is None:
            self._pan_angle = None
            self._tilt_angle = None
            self._gimbal_status = "云台回复缺少真实角度，自动控制已停止"
            self._stop_auto_control()
            if self.follow_button.isChecked():
                self.follow_button.setChecked(False)
            return
        self._pan_angle, self._tilt_angle = angles
        self._gimbal_status = "云台角度：PAN %.1f° · TILT %.1f°" % angles

    def _reset_pid_state(self):
        self._pan_pid.reset()
        self._tilt_pid.reset()
        self._pending_auto.clear()
        self._last_pid_time = None

    def _stop_auto_control(self):
        if self._gimbal_worker is not None:
            self._gimbal_worker.set_auto_enabled(False)
        self._reset_pid_state()

    def _load_pid_parameters(self):
        self._stop_auto_control()
        result = load_pid_config(self._config_path)
        self._pid_config = result.values
        if result.error:
            self._config_status = result.error
        elif result.source == "saved":
            self._config_status = "PID 参数：已保存配置"
        else:
            self._config_status = "PID 参数：默认配置"
        return result

    def on_activated(self):
        if self._active:
            return
        self._load_pid_parameters()
        self._active = True
        self._start_gimbal_worker()

    def on_deactivated(self):
        self._active = False
        self.clear_target()
        if self.follow_button.isChecked():
            self.follow_button.setChecked(False)
        self._shutdown_gimbal()

    def _shutdown_gimbal(self):
        self._stop_auto_control()
        worker = self._gimbal_worker
        if worker is not None:
            worker.stop()
            if worker.isRunning():
                worker.wait()
            if worker is self._gimbal_worker:
                self._gimbal_worker = None
        self._gimbal_connected = False
        self._gimbal_status = "云台未连接"
        self._pan_angle = None
        self._tilt_angle = None

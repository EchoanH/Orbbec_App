"""人脸录入、实时识别与人员库管理页面。"""

import logging
import math
import time
import traceback

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen
from PyQt5.QtWidgets import (QAbstractItemView, QApplication, QFrame,
                             QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                             QMessageBox, QPushButton, QSpinBox, QTableWidget,
                             QTableWidgetItem, QVBoxLayout)

from gimbal.controller import GimbalWorker
from gimbal.pid import PIDAxis, parse_gimbal_angles, safe_jog_for_angle
from gimbal.pid_config import get_default_pid_config, load_pid_config
from inference.enroll_worker import EnrollWorker
from inference.face_db_adapter import delete_person, list_enrolled
from inference.face_tracking import FaceTargetLock, normalized_face_center
from perf_logging import get_perf_logger

from .base_page import BasePage


SMOOTHING_TAU_MS = 30.0
NAME_STABLE_FRAMES = 3
PAN_SIGN = -1
TILT_SIGN = -1
PAN_SAFE_MIN = 60.0
PAN_SAFE_MAX = 120.0
TILT_SAFE_MIN = 60.0
TILT_SAFE_MAX = 120.0
LOGGER = logging.getLogger(__name__)
PERF = get_perf_logger()

# 推理节流：与 face14_page 相同的闸门参数，限制向推理 worker 提交帧的频率。
INFERENCE_FPS_LIMIT = 10.0
# 容差系数：闸门间隔取目标间隔的 85%。帧到达时刻是离散的（33.3ms 一格），
# 若阈值卡得刚好，抖动会让每隔一次都差几毫秒不达标，实际频率直接腰斩
# （实测 10Hz 设定跑出 5.2Hz）。留 15% 容差可吸收抖动。
_SUBMIT_TOLERANCE = 0.85
_MIN_SUBMIT_INTERVAL_NS = (int(1e9 / INFERENCE_FPS_LIMIT * _SUBMIT_TOLERANCE)
                           if INFERENCE_FPS_LIMIT > 0 else 0)


def _log_page_exception(stage, exc):
    stack = traceback.format_exc()
    message = "%s: %s\n%s" % (stage, exc, stack)
    LOGGER.error(message)
    PERF.event("EnrollPage exception", detail=message, level="WARNING")


class EnrollPage(BasePage):
    page_title = "人脸录入"
    page_hint = "YuNet + SFace · 实时识别与人员库"

    def __init__(self, yunet_session, parent=None,
                 gimbal_worker_factory=None, config_path=None):
        super(EnrollPage, self).__init__(parent)
        self._yunet_session = yunet_session
        self._gimbal_worker_factory = (
            gimbal_worker_factory if gimbal_worker_factory is not None
            else GimbalWorker)
        self._config_path = config_path
        self._worker = None
        self._active = False
        self._stop_requested = False
        self._restart_when_finished = False
        self._status_text = "人脸模型加载中..."
        self._latest_box = None
        self._display_box = None
        self._smoothing_tick_ns = None
        self._latest_similarity = 0.0
        self._latest_detection_score = 0.0
        self._pending_name = None
        self._pending_count = 0
        self._stable_name = ""
        self._enrolling = False
        self._last_submit_ns = 0
        self._submit_skipped = 0
        self._face_tracking_enabled = False
        self._face_target_lock = FaceTargetLock()
        self._tracked_face_box = None
        self._pan_pid = PIDAxis()
        self._tilt_pid = PIDAxis()
        self._pid_config = get_default_pid_config()
        self._gimbal_worker = None
        self._gimbal_connected = False
        self._gimbal_status = "未连接"
        self._last_gimbal_error = None
        self._pan_angle = None
        self._tilt_angle = None
        self._pending_auto = {}
        self._last_pid_time = None
        self._build_controls()
        self._refresh_people()
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._shutdown_worker)
            app.aboutToQuit.connect(self._shutdown_gimbal)

    def _build_controls(self):
        panel = QFrame()
        panel.setObjectName("formPanel")
        panel.setMaximumHeight(330)
        outer = QHBoxLayout(panel)
        outer.setContentsMargins(18, 16, 18, 16)
        outer.setSpacing(24)

        form = QVBoxLayout()
        form.setSpacing(8)
        title = QLabel("登记新成员")
        title.setObjectName("formTitle")
        form.addWidget(title)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("请输入姓名")
        form.addWidget(self.name_edit)
        count_row = QHBoxLayout()
        count_row.addWidget(QLabel("采集张数"))
        self.sample_spin = QSpinBox()
        self.sample_spin.setRange(3, 5)
        self.sample_spin.setValue(3)
        count_row.addWidget(self.sample_spin)
        count_row.addStretch(1)
        form.addLayout(count_row)
        self.register_button = QPushButton("录入")
        self.register_button.setObjectName("primaryButton")
        self.register_button.clicked.connect(self._start_enrollment)
        form.addWidget(self.register_button)
        self.face_track_button = QPushButton("人脸跟踪：关闭")
        self.face_track_button.setObjectName("primaryButton")
        self.face_track_button.setCheckable(True)
        self.face_track_button.toggled.connect(
            self._on_face_tracking_toggled)
        form.addWidget(self.face_track_button)
        self.face_tracking_label = QLabel(
            "跟踪目标：未检测到人脸\n云台：未连接")
        self.face_tracking_label.setObjectName("resultLabel")
        self.face_tracking_label.setWordWrap(True)
        form.addWidget(self.face_tracking_label)
        self.result_label = QLabel("识别结果：等待人脸\n相似度：--")
        self.result_label.setObjectName("resultLabel")
        self.result_label.setAlignment(Qt.AlignTop)
        self.result_label.setWordWrap(True)
        self.result_label.setMinimumHeight(62)
        form.addWidget(self.result_label)
        outer.addLayout(form, 2)

        people = QVBoxLayout()
        people.setSpacing(8)
        self.count_label = QLabel("已录入人员：0")
        self.count_label.setObjectName("formTitle")
        people.addWidget(self.count_label)
        self.people_table = QTableWidget(0, 3)
        self.people_table.setHorizontalHeaderLabels(["姓名", "样本数", "录入时间"])
        self.people_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.people_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.people_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.people_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.people_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.people_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        people.addWidget(self.people_table, 1)
        self.delete_button = QPushButton("删除选中人员")
        self.delete_button.clicked.connect(self._delete_selected)
        people.addWidget(self.delete_button)
        outer.addLayout(people, 3)
        self.layout().addWidget(panel)

    def _should_submit(self):
        """推理节流闸门：距上次提交不足最小间隔时跳过，不占用推理算力。

        以"提交时刻"而非"结果返回时刻"为基准，且带 15% 容差，避免与
        离散的帧到达时刻发生拍频。
        """
        if _MIN_SUBMIT_INTERVAL_NS <= 0:
            return True
        now_ns = time.perf_counter_ns()
        if now_ns - self._last_submit_ns < _MIN_SUBMIT_INTERVAL_NS:
            self._submit_skipped += 1
            return False
        self._last_submit_ns = now_ns
        return True

    def process_frame(self, bgr_frame, bgr_display=None, depth_frame=None):
        if self._worker is not None and self._should_submit():
            # 推理永远送原始 1280x720 帧，显示放大不参与推理链路。
            self._worker.submit_frame(bgr_frame)
        # 绘制底图用采集线程预放大的显示帧；为 None 时回退原始帧。
        display_frame = (bgr_display if bgr_display is not None
                         else bgr_frame)
        # 两级缩放：display_scale（原始帧 -> 显示帧，宽高独立）与
        # target_scale（显示帧 -> 画布，等比，仅画布小于显示帧时才 <1）。
        scale_x, scale_y = self.compute_display_scale(
            bgr_frame.shape, display_frame.shape)
        display_height, display_width = display_frame.shape[:2]
        target_width, target_height, target_scale = self.compute_target_size(
            display_width, display_height)
        if target_scale >= 1.0:
            # 画布不小于显示帧：跳过 cv2.resize，直接用显示帧。
            # 修复原有 bug：此前无条件 resize 会在 scale=1.0 时对每帧做
            # 一次同尺寸全帧拷贝。
            small_frame = display_frame
        else:
            small_frame = cv2.resize(
                display_frame, (target_width, target_height),
                interpolation=cv2.INTER_LINEAR)
        box = self._update_display_box() if self._latest_box is not None else None
        box_scaled = None
        if box is not None:
            box_scaled = box.copy()
            # box 为 [x1, y1, x2, y2]：x 项乘 scale_x，y 项乘 scale_y，
            # 再统一乘 target_scale。
            box_scaled[0] *= scale_x * target_scale
            box_scaled[2] *= scale_x * target_scale
            box_scaled[1] *= scale_y * target_scale
            box_scaled[3] *= scale_y * target_scale
        label = None
        if box is not None:
            label = self._stable_name or "识别中"
            if self._stable_name:
                label = "%s · 相似度 %.2f · 置信度 %.3f" % (
                    self._stable_name, self._latest_similarity,
                    self._latest_detection_score)
            else:
                label = "识别中 · 置信度 %.3f" % self._latest_detection_score
        return self._draw_match(small_frame, box_scaled, label), self._status_text

    def on_activated(self):
        started_ns = time.perf_counter_ns()
        PERF.event("EnrollPage activate start", detail="active=True")
        self._active = True
        self._last_submit_ns = 0
        try:
            self._refresh_people()
            if self._worker is None:
                self._start_worker()
            elif self._stop_requested:
                self._restart_when_finished = True
        except Exception as exc:
            _log_page_exception("EnrollPage activate failed", exc)
            raise
        PERF.event("EnrollPage activate complete",
                   (time.perf_counter_ns() - started_ns) / 1e6)

    def on_deactivated(self):
        started_ns = time.perf_counter_ns()
        PERF.event("EnrollPage deactivate start", detail="begin worker stop/release")
        self._active = False
        try:
            if self.face_track_button.isChecked():
                self.face_track_button.setChecked(False)
            else:
                self._shutdown_gimbal()
            self._reset_match()
            self._set_enrolling(False)
            self._restart_when_finished = False
            if self._worker is not None:
                self._worker.cancel_enrollment()
                self._stop_requested = True
                self._worker.stop()
                if self._worker.isRunning():
                    self._worker.wait()
        except Exception as exc:
            _log_page_exception("EnrollPage deactivate failed", exc)
            raise
        PERF.event("EnrollPage deactivate complete",
                   (time.perf_counter_ns() - started_ns) / 1e6)

    def _start_worker(self):
        started_ns = time.perf_counter_ns()
        PERF.event("EnrollPage model loading start",
                   detail="start EnrollWorker; YuNet singleton id=%s + SFace" %
                   id(self._yunet_session))
        self._status_text = "人脸模型加载中..."
        self._stop_requested = False
        try:
            worker = EnrollWorker(self._yunet_session, self)
            worker.set_face_tracking_enabled(self._face_tracking_enabled)
            worker.status_ready.connect(self._on_status)
            worker.match_ready.connect(self._on_match)
            worker.face_candidates_ready.connect(self._on_face_candidates)
            worker.enroll_progress.connect(self._on_enroll_progress)
            worker.enroll_complete.connect(self._on_enroll_complete)
            worker.error_occurred.connect(self._on_error)
            worker.finished.connect(self._on_finished)
            self._worker = worker
            worker.start()
        except Exception as exc:
            _log_page_exception("EnrollPage model loading start failed", exc)
            raise
        PERF.event("EnrollPage model loading dispatched",
                   (time.perf_counter_ns() - started_ns) / 1e6)

    def _start_enrollment(self):
        clicked_ns = time.perf_counter_ns()
        PERF.event("EnrollPage enroll button clicked", detail="before validation")
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "无法录入", "请输入姓名。")
            return
        try:
            existing = {row[0] for row in list_enrolled()}
        except Exception as exc:
            _log_page_exception("EnrollPage enroll validation failed", exc)
            raise
        if name in existing:
            answer = QMessageBox.question(
                self, "确认覆盖", "该姓名已录入，是否覆盖？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                return
        if self._worker is None or not self._worker.isRunning():
            QMessageBox.warning(self, "模型未就绪", "人脸模型尚未启动。")
            return
        if self._face_tracking_enabled:
            self._stop_face_auto_control(reset_lock=False)
            self._set_face_tracking_status("录入期间暂停")
        target = self.sample_spin.value()
        self._set_enrolling(True)
        self.result_label.setText("开始录入 %s：0/%d\n请正对摄像头并保持光线充足" % (name, target))
        PERF.event("EnrollPage start_enrollment dispatch",
                   (time.perf_counter_ns() - clicked_ns) / 1e6,
                   "name=%s target=%d; worker will call extract_feature_for_enroll" %
                   (name, target))
        try:
            self._worker.start_enrollment(name, target)
        except Exception as exc:
            _log_page_exception("EnrollPage start_enrollment failed", exc)
            raise
        PERF.event("EnrollPage start_enrollment dispatched",
                   (time.perf_counter_ns() - clicked_ns) / 1e6)

    def _on_status(self, text):
        PERF.event("EnrollPage worker status", detail=str(text))
        self._status_text = text

    def _on_match(self, box, name, similarity, detection_score, reason):
        if not self._active or self._stop_requested or self._enrolling:
            return
        if box is None:
            self._reset_match()
            self.result_label.setText("识别结果：等待人脸\n相似度：--\n%s" % reason)
            self._status_text = reason
            return
        self._latest_box = np.asarray(box, dtype=np.float32)
        if self._display_box is None:
            self._display_box = self._latest_box.copy()
            self._smoothing_tick_ns = time.perf_counter_ns()
        candidate = name or "未知"
        if candidate == self._pending_name:
            self._pending_count += 1
        else:
            self._pending_name = candidate
            self._pending_count = 1
        if self._pending_count >= NAME_STABLE_FRAMES:
            self._stable_name = candidate
        self._latest_similarity = similarity
        self._latest_detection_score = detection_score
        display_name = self._stable_name or "识别中"
        self.result_label.setText(
            "识别结果：%s\n相似度：%.3f\n检测置信度：%.3f" % (
                display_name, similarity, detection_score))
        self._status_text = "%s · 相似度 %.3f · 检测 %.3f" % (
            display_name, similarity, detection_score)

    def _on_face_candidates(self, candidates, frame_shape):
        if (not self._active or not self._face_tracking_enabled or
                self._stop_requested or self._enrolling):
            return
        target = self._face_target_lock.update(candidates or [])
        if target is None:
            self._stop_face_auto_control(reset_lock=False)
            target_text = (
                "等待原锁定人脸" if self._face_target_lock.locked_box is not None
                else "未检测到人脸")
            self._set_face_tracking_status(target_text)
            return
        self._tracked_face_box = target
        normalized = normalized_face_center(target, frame_shape)
        self._set_face_tracking_status("人脸")
        self._maybe_control_face(normalized)

    def _on_face_tracking_toggled(self, enabled):
        self.face_track_button.setText(
            "人脸跟踪：开启" if enabled else "人脸跟踪：关闭")
        if not enabled:
            self._face_tracking_enabled = False
            if self._worker is not None:
                self._worker.set_face_tracking_enabled(False)
            self._stop_face_auto_control(reset_lock=True)
            self._shutdown_gimbal()
            self._set_face_tracking_status("未检测到人脸")
            return
        if not self._active:
            self.face_track_button.blockSignals(True)
            self.face_track_button.setChecked(False)
            self.face_track_button.blockSignals(False)
            self.face_track_button.setText("人脸跟踪：关闭")
            return
        result = load_pid_config(self._config_path)
        if result.error:
            self._face_tracking_enabled = False
            self.face_track_button.blockSignals(True)
            self.face_track_button.setChecked(False)
            self.face_track_button.blockSignals(False)
            self.face_track_button.setText("人脸跟踪：关闭")
            self._gimbal_status = result.error
            self._set_face_tracking_status("未检测到人脸")
            return
        self._pid_config = result.values
        self._face_tracking_enabled = True
        if self._worker is not None:
            self._worker.set_face_tracking_enabled(True)
        self._last_gimbal_error = None
        self._gimbal_status = "连接中"
        self._stop_face_auto_control(reset_lock=True)
        self._start_gimbal_worker()
        self._set_face_tracking_status("未检测到人脸")

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
        worker.command_completed.connect(self._on_gimbal_command_completed)
        worker.error_occurred.connect(self._on_gimbal_error)
        worker.finished.connect(self._on_gimbal_finished)
        self._gimbal_worker = worker
        worker.start()

    def _on_gimbal_status(self, text):
        if text == "云台已连接":
            if self._pan_angle is None or self._tilt_angle is None:
                self._gimbal_status = "已连接，等待当前角度"
            else:
                self._gimbal_status = "PAN %.1f° · TILT %.1f°" % (
                    self._pan_angle, self._tilt_angle)
        elif self._last_gimbal_error is None:
            self._gimbal_status = text
        self._set_face_tracking_status()

    def _on_gimbal_connection(self, connected):
        self._gimbal_connected = bool(connected)
        if connected:
            self._gimbal_status = "已连接，等待当前角度"
        else:
            self._pan_angle = None
            self._tilt_angle = None
            self._stop_face_auto_control(reset_lock=False)
            self._gimbal_status = self._last_gimbal_error or "未连接"
        self._set_face_tracking_status()

    def _on_gimbal_response(self, response):
        angles = parse_gimbal_angles(response)
        if angles is None:
            self._pan_angle = None
            self._tilt_angle = None
            self._gimbal_status = "回复缺少当前角度，自动控制已停止"
            self._stop_face_auto_control(reset_lock=False)
        else:
            self._pan_angle, self._tilt_angle = angles
            self._gimbal_status = "PAN %.1f° · TILT %.1f°" % angles
        self._set_face_tracking_status()

    def _on_gimbal_error(self, text):
        self._last_gimbal_error = text
        self._gimbal_status = text
        self._gimbal_connected = False
        self._pan_angle = None
        self._tilt_angle = None
        self._face_tracking_enabled = False
        if self._worker is not None:
            self._worker.set_face_tracking_enabled(False)
        self.face_track_button.blockSignals(True)
        self.face_track_button.setChecked(False)
        self.face_track_button.blockSignals(False)
        self.face_track_button.setText("人脸跟踪：关闭")
        self._stop_face_auto_control(reset_lock=True)
        self._shutdown_gimbal(final_status=text)
        self._set_face_tracking_status("未检测到人脸")

    def _on_gimbal_finished(self):
        worker = self.sender()
        if worker is self._gimbal_worker:
            self._gimbal_worker = None
        worker.deleteLater()
        self._gimbal_connected = False
        self._pan_angle = None
        self._tilt_angle = None
        self._stop_face_auto_control(reset_lock=False)
        if self._last_gimbal_error is None:
            self._gimbal_status = "未连接"
        self._set_face_tracking_status()

    def _maybe_control_face(self, normalized_center):
        worker = self._gimbal_worker
        if (worker is None or not self._face_tracking_enabled or
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
            self._stop_face_auto_control(reset_lock=False)
            self._set_face_tracking_status()
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
            self._set_face_tracking_status()
        return safe

    def _on_gimbal_command_completed(self, command, _response):
        parts = command.split()
        if len(parts) != 4 or parts[:2] != ["GIMBAL", "JOG"]:
            return
        pending = self._pending_auto.pop(parts[2], None)
        if pending is None:
            return
        pid_axis, logical_jog = pending
        pid_axis.consume_jog(logical_jog)

    def _reset_face_pid(self):
        self._pan_pid.reset()
        self._tilt_pid.reset()
        self._pending_auto.clear()
        self._last_pid_time = None

    def _stop_face_auto_control(self, reset_lock=False):
        worker = self._gimbal_worker
        if worker is not None:
            worker.set_auto_enabled(False)
        self._reset_face_pid()
        self._tracked_face_box = None
        if reset_lock:
            self._face_target_lock.reset()

    def _set_face_tracking_status(self, target_text=None):
        if target_text is None:
            target_text = (
                "人脸" if self._tracked_face_box is not None
                else "未检测到人脸")
        self.face_tracking_label.setText(
            "跟踪目标：%s\n云台：%s" % (target_text, self._gimbal_status))

    def _shutdown_gimbal(self, final_status="未连接"):
        self._stop_face_auto_control(reset_lock=True)
        worker = self._gimbal_worker
        if worker is not None:
            worker.stop()
            if worker.isRunning():
                worker.wait()
            if worker is self._gimbal_worker:
                self._gimbal_worker = None
        self._gimbal_connected = False
        self._pan_angle = None
        self._tilt_angle = None
        self._gimbal_status = final_status

    def _on_enroll_progress(self, count, target, score, reason):
        PERF.event("EnrollPage enroll progress",
                   detail="count=%d/%d score=%.3f reason=%s" %
                   (count, target, score, reason))
        if not self._enrolling:
            return
        self.result_label.setText(
            "录入进度：%d/%d\n本次置信度：%.3f\n%s" % (count, target, score, reason))
        self._status_text = "录入中 %d/%d · %s" % (count, target, reason)

    def _on_enroll_complete(self, name, record):
        PERF.event("EnrollPage enroll complete",
                   detail="name=%s sample_count=%s" %
                   (name, record.get("sample_count", 0)))
        self._set_enrolling(False)
        self._refresh_people()
        self.result_label.setText(
            "录入成功：%s\n样本数：%d\n录入时间：%s" % (
                name, record.get("sample_count", 0), record.get("enrolled_at", "?")))
        self._status_text = "录入成功：%s" % name
        QMessageBox.information(self, "录入完成", "%s 已成功录入。" % name)

    def _on_error(self, text):
        PERF.event("EnrollPage worker error", detail=str(text), level="WARNING")
        self._reset_match()
        self._set_enrolling(False)
        self._stop_face_auto_control(reset_lock=False)
        self._set_face_tracking_status("未检测到人脸")
        self._status_text = text
        self.result_label.setText(text)

    def _on_finished(self):
        worker = self.sender()
        PERF.event("EnrollPage worker finished",
                   detail="same_worker=%s" % (worker is self._worker))
        if worker is self._worker:
            self._worker = None
        worker.deleteLater()
        should_restart = self._active and self._restart_when_finished
        self._restart_when_finished = False
        self._stop_requested = False
        if self._face_tracking_enabled:
            self._stop_face_auto_control(reset_lock=False)
            self._set_face_tracking_status("人脸模型已停止")
        if should_restart and self._worker is None:
            self._start_worker()

    def _delete_selected(self):
        row = self.people_table.currentRow()
        if row < 0:
            QMessageBox.warning(self, "无法删除", "请先选择一名已录入人员。")
            return
        name = self.people_table.item(row, 0).text()
        answer = QMessageBox.question(
            self, "确认删除", "确定删除“%s”的人脸记录吗？" % name,
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return
        if delete_person(name):
            self._refresh_people()
            self.result_label.setText("已删除：%s" % name)
        else:
            QMessageBox.warning(self, "删除失败", "该姓名已不存在。")

    def _refresh_people(self):
        started_ns = time.perf_counter_ns()
        try:
            records = list_enrolled()
        except Exception as exc:
            _log_page_exception("EnrollPage list_enrolled failed", exc)
            raise
        self.people_table.setRowCount(len(records))
        for row, (name, sample_count, enrolled_at) in enumerate(records):
            self.people_table.setItem(row, 0, QTableWidgetItem(str(name)))
            self.people_table.setItem(row, 1, QTableWidgetItem(str(sample_count)))
            self.people_table.setItem(row, 2, QTableWidgetItem(str(enrolled_at)))
        self.count_label.setText("已录入人员：%d" % len(records))

        PERF.event("EnrollPage list_enrolled returned",
                   (time.perf_counter_ns() - started_ns) / 1e6,
                   "count=%d" % len(records))

    def _set_enrolling(self, active):
        self._enrolling = active
        self.name_edit.setEnabled(not active)
        self.sample_spin.setEnabled(not active)
        self.register_button.setEnabled(not active)
        self.delete_button.setEnabled(not active)

    def _update_display_box(self):
        target = self._latest_box
        if self._display_box is None:
            self._display_box = target.copy()
            self._smoothing_tick_ns = time.perf_counter_ns()
            return self._display_box
        now_ns = time.perf_counter_ns()
        elapsed_ms = max(0.0, (now_ns - self._smoothing_tick_ns) / 1e6)
        self._smoothing_tick_ns = now_ns
        alpha = 1.0 - math.exp(-elapsed_ms / SMOOTHING_TAU_MS)
        self._display_box = self._display_box + alpha * (target - self._display_box)
        return self._display_box

    def _reset_match(self):
        self._latest_box = None
        self._display_box = None
        self._smoothing_tick_ns = None
        self._latest_similarity = 0.0
        self._latest_detection_score = 0.0
        self._pending_name = None
        self._pending_count = 0
        self._stable_name = ""

    @staticmethod
    def _draw_match(bgr_frame, box, label):
        # 引导椭圆已取消（用户确认不再需要）：draw_face_guide 内部含
        # canvas.copy + overlay.copy + addWeighted 三次全帧运算，直接使用
        # 传入帧，视觉输出仅少了引导椭圆，其余 QPainter 绘制不变。
        canvas = bgr_frame
        if box is None or label is None:
            return canvas
        canvas = np.ascontiguousarray(canvas)
        height, width = canvas.shape[:2]
        rgb = canvas[:, :, ::-1].copy()
        image = QImage(rgb.data, width, height, int(rgb.strides[0]),
                       QImage.Format_RGB888).copy()
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor(74, 158, 255), 3))
        x1, y1, x2, y2 = [float(v) for v in box]
        painter.drawRect(QRectF(x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)))
        painter.setFont(QFont("Droid Sans Fallback", 16, QFont.Bold))
        metrics = painter.fontMetrics()
        text_width = metrics.horizontalAdvance(label) + 14
        text_height = metrics.height() + 8
        text_y = max(0.0, y1 - text_height)
        painter.fillRect(QRectF(x1, text_y, text_width, text_height), QColor(74, 158, 255))
        painter.setPen(QColor(8, 19, 31))
        painter.drawText(QRectF(x1 + 7, text_y, text_width - 7, text_height),
                         Qt.AlignVCenter | Qt.AlignLeft, label)
        painter.end()
        rendered = image.convertToFormat(QImage.Format_RGB888)
        ptr = rendered.bits()
        ptr.setsize(rendered.byteCount())
        rows = np.frombuffer(ptr, dtype=np.uint8).reshape(
            height, rendered.bytesPerLine())
        drawn_rgb = rows[:, :width * 3].reshape(height, width, 3)
        return drawn_rgb[:, :, ::-1].copy()

    def _shutdown_worker(self):
        started_ns = time.perf_counter_ns()
        PERF.event("EnrollPage shutdown start")
        self._active = False
        worker = self._worker
        try:
            if worker is not None and worker.isRunning():
                worker.cancel_enrollment()
                worker.stop()
                worker.wait()
            self._worker = None
        except Exception as exc:
            _log_page_exception("EnrollPage shutdown failed", exc)
            raise
        PERF.event("EnrollPage shutdown complete",
                   (time.perf_counter_ns() - started_ns) / 1e6)

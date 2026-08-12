"""人脸 14 关键点页面：管理专用推理线程并叠加最近一次检测结果。"""

import logging
import math
import time

import cv2
from PyQt5.QtWidgets import QApplication

from inference.face14_infer import draw_face14
from inference.face14_worker import Face14Worker
from ui.draw_utils import draw_text_box_bgr

from .base_page import BasePage


LOGGER = logging.getLogger(__name__)

# 现场对比开关：False 时直接绘制 worker 输出的原始坐标。
SMOOTHING_ENABLED = True
SMOOTHING_TAU_MS = 30.0
# 任一点位移超过新结果 14 点包围框最大边长的 20% 时整组直接吸附。
SMOOTHING_SNAP_RATIO = 0.20


class Face14Page(BasePage):
    page_title = "人脸关键点检测"
    page_hint = "14 关键点 · NPU 实时推理"

    def __init__(self, yunet_session, parent=None):
        super(Face14Page, self).__init__(parent)
        self._yunet_session = yunet_session
        self._worker = None
        self._active = False
        self._latest_face14 = None
        self._latest_face_box = None
        self._latest_score = 0.0
        self._latest_distance_cm = None
        self._display_face14 = None
        self._smoothing_tick_ns = None
        self._status_text = "模型加载中..."
        self._last_draw_error = None
        self._worker_failed = False
        self._worker_stop_requested = False
        self._restart_when_finished = False
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._shutdown_worker)

    def showEvent(self, event):
        super(Face14Page, self).showEvent(event)
        self.on_activated()

    def process_frame(self, bgr_frame, depth_frame=None):
        if self._worker is not None:
            self._worker.submit_frame(bgr_frame, depth_frame)
        # P0 UI 渲染性能优化：绘制前先降采样，坐标同步缩放。
        height, width = bgr_frame.shape[:2]
        target_width, target_height, scale = self.compute_target_size(
            width, height)
        small_frame = cv2.resize(
            bgr_frame, (target_width, target_height),
            interpolation=cv2.INTER_LINEAR)
        try:
            display_face14 = (self._update_display_face14()
                              if self._latest_face14 is not None else None)
            face14_scaled = None
            if display_face14 is not None:
                face14_scaled = {
                    name: (point[0] * scale, point[1] * scale)
                    for name, point in display_face14.items()
                }
            face_box_scaled = None
            if self._latest_face_box is not None:
                face_box_scaled = [value * scale
                                   for value in self._latest_face_box]
            rendered = draw_face14(small_frame, face14_scaled,
                                   face_box_scaled, self._latest_score)
            if (self._latest_distance_cm is not None
                    and self._latest_distance_cm > 0.0
                    and face_box_scaled is not None):
                x1, y1 = face_box_scaled[:2]
                rendered = draw_text_box_bgr(
                    rendered, "距离 %.1f cm" % self._latest_distance_cm,
                    x1, y1 - 62, font_size=15,
                    text_color=(8, 19, 31),
                    background_color=(74, 158, 255))
            return rendered, self._status_text
        except Exception as exc:
            message = "关键点绘制异常：%s" % exc
            if message != self._last_draw_error:
                LOGGER.exception(message)
                self._last_draw_error = message
            return small_frame, message

    def on_activated(self):
        self._active = True
        if self._worker is None:
            self._start_worker()
        elif self._worker_stop_requested:
            self._restart_when_finished = True

    def on_deactivated(self):
        self._active = False
        self._latest_face14 = None
        self._latest_face_box = None
        self._latest_score = 0.0
        self._latest_distance_cm = None
        self._display_face14 = None
        self._smoothing_tick_ns = None
        self._restart_when_finished = False
        if self._worker is not None:
            self._worker_stop_requested = True
            self._worker.stop()
            if self._worker.isRunning():
                self._worker.wait()

    def _start_worker(self):
        self._status_text = "模型加载中..."
        self._worker_failed = False
        self._worker_stop_requested = False
        self._restart_when_finished = False
        worker = Face14Worker(self._yunet_session, self)
        worker.status_ready.connect(self._on_status)
        worker.result_ready.connect(self._on_result)
        worker.error_occurred.connect(self._on_error)
        worker.finished.connect(self._on_worker_finished)
        self._worker = worker
        worker.start()

    def _on_status(self, text):
        self._status_text = text

    def _on_result(self, face14, face_box, score, distance_cm, elapsed_ms,
                   text):
        if not self._active or self._worker_stop_requested:
            return
        self._latest_face14 = face14
        self._latest_face_box = face_box
        self._latest_score = score
        self._latest_distance_cm = distance_cm
        if face14 is None:
            self._display_face14 = None
            self._smoothing_tick_ns = None
        elif (not SMOOTHING_ENABLED or self._display_face14 is None
              or self._should_snap_to_result(face14)):
            self._display_face14 = self._copy_face14(face14)
            self._smoothing_tick_ns = time.perf_counter_ns()
        self._status_text = text

    def _on_error(self, text):
        self._worker_failed = True
        self._latest_face14 = None
        self._latest_face_box = None
        self._latest_score = 0.0
        self._latest_distance_cm = None
        self._display_face14 = None
        self._smoothing_tick_ns = None
        self._status_text = text

    def _update_display_face14(self):
        """仅更新显示坐标；不修改 worker 传入的原始推理结果。"""
        target = self._latest_face14
        if target is None or not SMOOTHING_ENABLED:
            return target
        if self._display_face14 is None:
            self._display_face14 = self._copy_face14(target)
            self._smoothing_tick_ns = time.perf_counter_ns()
            return self._display_face14

        now_ns = time.perf_counter_ns()
        if self._smoothing_tick_ns is None:
            self._smoothing_tick_ns = now_ns
            return self._display_face14
        elapsed_ms = max(0.0, (now_ns - self._smoothing_tick_ns) / 1e6)
        self._smoothing_tick_ns = now_ns
        alpha = 1.0 - math.exp(-elapsed_ms / SMOOTHING_TAU_MS)
        smoothed = {}
        for name, target_point in target.items():
            current_point = self._display_face14[name]
            smoothed[name] = (
                current_point[0] + alpha * (target_point[0] - current_point[0]),
                current_point[1] + alpha * (target_point[1] - current_point[1]),
            )
        self._display_face14 = smoothed
        return smoothed

    def _should_snap_to_result(self, target):
        if set(target) != set(self._display_face14):
            return True
        xs = [point[0] for point in target.values()]
        ys = [point[1] for point in target.values()]
        face_size = max(max(xs) - min(xs), max(ys) - min(ys))
        if face_size <= 0.0:
            return True
        threshold = face_size * SMOOTHING_SNAP_RATIO
        for name, target_point in target.items():
            current_point = self._display_face14[name]
            if math.hypot(target_point[0] - current_point[0],
                          target_point[1] - current_point[1]) > threshold:
                return True
        return False

    @staticmethod
    def _copy_face14(face14):
        return {name: (float(point[0]), float(point[1]))
                for name, point in face14.items()}

    def _on_worker_finished(self):
        worker = self.sender()
        if worker is self._worker:
            self._worker = None
        worker.deleteLater()
        should_restart = (self._active and self._restart_when_finished
                          and not self._worker_failed)
        self._worker_stop_requested = False
        self._restart_when_finished = False
        if should_restart and self._worker is None:
            self._start_worker()

    def _shutdown_worker(self):
        self._active = False
        worker = self._worker
        if worker is not None and worker.isRunning():
            worker.stop()
            worker.wait()
        self._worker = None

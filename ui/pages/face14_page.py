"""人脸 14 关键点页面：管理专用推理线程并叠加最近一次检测结果。"""

import logging
import math
import time

import cv2
from PyQt5.QtWidgets import QApplication

from inference.face14_worker import Face14Worker
from perf_logging import get_perf_logger
from ui.draw_utils import draw_text_box_bgr
from ui.fast_draw import draw_face14_fast

from .base_page import BasePage


LOGGER = logging.getLogger(__name__)
PERF = get_perf_logger()

# 现场对比开关：False 时直接绘制 worker 输出的原始坐标。
SMOOTHING_ENABLED = True
SMOOTHING_TAU_MS = 30.0
# 任一点位移超过新结果 14 点包围框最大边长的 20% 时整组直接吸附。
SMOOTHING_SNAP_RATIO = 0.20

INFERENCE_FPS_LIMIT = 10.0
# 容差系数：闸门间隔取目标间隔的 85%。帧到达时刻是离散的（33.3ms 一格），
# 若阈值卡得刚好，抖动会让每隔一次都差几毫秒不达标，实际频率直接腰斩
# （实测 10Hz 设定跑出 5.2Hz）。留 15% 容差可吸收抖动。
_SUBMIT_TOLERANCE = 0.85
_MIN_SUBMIT_INTERVAL_NS = (int(1e9 / INFERENCE_FPS_LIMIT * _SUBMIT_TOLERANCE)
                           if INFERENCE_FPS_LIMIT > 0 else 0)


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
        self._last_submit_ns = 0
        self._submit_skipped = 0
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._shutdown_worker)

    def showEvent(self, event):
        super(Face14Page, self).showEvent(event)
        self.on_activated()

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

    def process_frame(self, bgr_frame, depth_frame=None):
        if self._worker is not None and self._should_submit():
            self._worker.submit_frame(bgr_frame, depth_frame)
        # P0 UI 渲染性能优化：只在画布小于源帧时才降采样；scale==1.0 时
        # 完全跳过 cv2.resize，直接用原帧绘制，放大交给 Qt。
        height, width = bgr_frame.shape[:2]
        target_width, target_height, scale = self.compute_target_size(
            width, height)
        if scale >= 1.0:
            small_frame = bgr_frame
        else:
            small_frame = cv2.resize(
                bgr_frame, (target_width, target_height),
                interpolation=cv2.INTER_LINEAR)
        try:
            display_face14 = (self._update_display_face14()
                              if self._latest_face14 is not None else None)
            face14_scaled = None
            if display_face14 is not None:
                if scale >= 1.0:
                    face14_scaled = display_face14
                else:
                    face14_scaled = {
                        name: (point[0] * scale, point[1] * scale)
                        for name, point in display_face14.items()
                    }
            face_box_scaled = None
            if self._latest_face_box is not None:
                if scale >= 1.0:
                    face_box_scaled = self._latest_face_box
                else:
                    face_box_scaled = [value * scale
                                       for value in self._latest_face_box]
            draw_started_ns = time.perf_counter_ns()
            rendered = draw_face14_fast(small_frame, face14_scaled,
                                        face_box_scaled, self._latest_score)
            PERF.event("诊断process_frame绘制耗时",
                       (time.perf_counter_ns() - draw_started_ns) / 1e6)
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
        LOGGER.warning("on_activated 被调用，当前 self._worker=%s, tid=%s",
                       self._worker, id(self))
        self._active = True
        self._last_submit_ns = 0
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
                   text, emit_wall_time=None):
        # 诊断埋点：信号被主线程真正处理的时刻，减去 worker 发出信号的时刻，
        # 得到跨线程排队延迟。两端都用 time.time()（墙钟），量级一致可直接
        # 相减；不要混用 perf_counter_ns()（起点不确定）与 time.time()。
        if emit_wall_time is not None:
            queue_delay_ms = (time.time() - emit_wall_time) * 1000.0
            PERF.event("诊断信号跨线程延迟", queue_delay_ms)
            if queue_delay_ms > 100.0:
                PERF.event("诊断信号跨线程延迟异常", queue_delay_ms,
                           "超过100ms", level="WARNING")
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
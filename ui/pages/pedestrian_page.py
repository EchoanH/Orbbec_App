"""行人检测演示页。"""

import logging
import time

import cv2
from PyQt5.QtWidgets import QApplication

from inference.pedestrian_worker import PedestrianWorker
from ui.draw_utils import draw_text_box_bgr
from ui.fast_draw import draw_pedestrians_fast

from .base_page import BasePage


LOGGER = logging.getLogger(__name__)

# 推理节流：与 face14_page 相同的闸门参数，限制向推理 worker 提交帧的频率。
INFERENCE_FPS_LIMIT = 10.0
# 容差系数：闸门间隔取目标间隔的 85%。帧到达时刻是离散的（33.3ms 一格），
# 若阈值卡得刚好，抖动会让每隔一次都差几毫秒不达标，实际频率直接腰斩
# （实测 10Hz 设定跑出 5.2Hz）。留 15% 容差可吸收抖动。
_SUBMIT_TOLERANCE = 0.85
_MIN_SUBMIT_INTERVAL_NS = (int(1e9 / INFERENCE_FPS_LIMIT * _SUBMIT_TOLERANCE)
                           if INFERENCE_FPS_LIMIT > 0 else 0)


class PedestrianPage(BasePage):
    page_title = "行人检测"
    page_hint = "YOLOv5s · COCO person · NPU 实时推理"

    def __init__(self, parent=None):
        super(PedestrianPage, self).__init__(parent)
        self._worker = None
        self._active = False
        self._latest_dets = []
        self._status_text = "行人模型加载中..."
        self._restart_when_finished = False
        self._stop_requested = False
        self._last_error = None
        self._last_submit_ns = 0
        self._submit_skipped = 0
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._shutdown_worker)

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
        if not self._latest_dets:
            return small_frame, self._status_text
        # dets 坐标为 [x1, y1, x2, y2]：x 项乘 scale_x，y 项乘 scale_y，
        # 再统一乘 target_scale。
        total_sx = scale_x * target_scale
        total_sy = scale_y * target_scale
        scaled_dets = [
            (name, score,
             [box[0] * total_sx, box[1] * total_sy,
              box[2] * total_sx, box[3] * total_sy])
            for name, score, box in self._latest_dets
        ]
        rendered = draw_pedestrians_fast(small_frame, scaled_dets)
        for index, (_, score, box) in enumerate(scaled_dets, 1):
            rendered = draw_text_box_bgr(
                rendered, "行人%d · 置信度 %.2f" % (index, score),
                box[0], box[1] - 26, font_size=12,
                text_color=(0, 255, 0), padding=2)
        rendered = draw_text_box_bgr(
            rendered, "人数: %d" % len(scaled_dets), 18, 8,
            font_size=16, text_color=(255, 255, 0))
        return rendered, self._status_text

    def on_activated(self):
        self._active = True
        self._last_submit_ns = 0
        if self._worker is None:
            self._start_worker()
        elif self._stop_requested:
            self._restart_when_finished = True

    def on_deactivated(self):
        self._active = False
        self._latest_dets = []
        self._restart_when_finished = False
        if self._worker is not None:
            self._stop_requested = True
            self._worker.stop()
            if self._worker.isRunning():
                self._worker.wait()

    def _start_worker(self):
        self._status_text = "行人模型加载中..."
        self._stop_requested = False
        worker = PedestrianWorker(self)
        worker.status_ready.connect(self._on_status)
        worker.result_ready.connect(self._on_result)
        worker.error_occurred.connect(self._on_error)
        worker.finished.connect(self._on_finished)
        self._worker = worker
        worker.start()

    def _on_status(self, text):
        self._status_text = text

    def _on_result(self, dets, elapsed_ms, text):
        if not self._active or self._stop_requested:
            return
        self._latest_dets = dets
        self._status_text = text

    def _on_error(self, text):
        self._latest_dets = []
        self._status_text = text
        self._last_error = text

    def _on_finished(self):
        worker = self.sender()
        if worker is self._worker:
            self._worker = None
        worker.deleteLater()
        should_restart = self._active and self._restart_when_finished
        self._restart_when_finished = False
        self._stop_requested = False
        if should_restart and self._worker is None:
            self._start_worker()

    def _shutdown_worker(self):
        self._active = False
        worker = self._worker
        if worker is not None and worker.isRunning():
            worker.stop()
            worker.wait()
        self._worker = None

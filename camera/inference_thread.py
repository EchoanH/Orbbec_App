"""推理线程占位：当前直通帧，为后续 NPU 模型接入保留独立执行边界。"""

import logging
import time
from queue import Empty, Full, Queue

from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot

from perf_logging import get_perf_logger

LOGGER = logging.getLogger(__name__)
PERF = get_perf_logger()


class InferenceThread(QThread):
    """独立推理位置；目前不加载模型，只转发最新帧避免积压。"""

    frame_ready = pyqtSignal(object, object)

    def __init__(self, parent=None):
        super(InferenceThread, self).__init__(parent)
        self._queue = Queue(maxsize=1)
        self._running = False

    @pyqtSlot(object, object)
    def submit_frame(self, bgr_frame, depth_frame):
        """接收采集帧；队列满时丢弃旧帧，保证演示画面保持实时。"""
        if not self._running:
            return
        item = (bgr_frame, depth_frame, time.perf_counter_ns())
        dropped = False
        try:
            self._queue.put_nowait(item)
        except Full:
            while True:
                try:
                    self._queue.get_nowait()
                    dropped = True
                    break
                except Empty:
                    break
            while True:
                try:
                    self._queue.put_nowait(item)
                    break
                except Full:
                    try:
                        self._queue.get_nowait()
                    except Empty:
                        continue
            if dropped:
                PERF.increment("pipeline_queue_dropped")
                PERF.event("主动跳帧（正常）", 0.0,
                           "通用容量1队列覆盖旧帧")

    def stop(self):
        self._running = False

    def run(self):
        self._running = True
        while self._running:
            try:
                bgr_frame, depth_frame, submitted_ns = self._queue.get(timeout=0.1)
            except Empty:
                continue
            PERF.set_gauge("pipeline_queue_depth", self._queue.qsize())
            PERF.event("通用推理帧出队",
                       (time.perf_counter_ns() - submitted_ns) / 1e6)
            # 后续在此处调用 NPU 推理，当前保持直通以验证完整 UI 链路。
            self.frame_ready.emit(bgr_frame, depth_frame)

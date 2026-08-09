"""行人检测后台 worker：只处理容量为 1 队列中的最新帧。"""

import logging
import time
from collections import deque
from queue import Empty, Full, Queue

from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot

from perf_logging import get_perf_logger
from .pedestrian_infer import PedestrianEngine


LOGGER = logging.getLogger(__name__)
PERF = get_perf_logger()


class PedestrianWorker(QThread):
    status_ready = pyqtSignal(str)
    result_ready = pyqtSignal(object, float, str)
    error_occurred = pyqtSignal(str)

    def __init__(self, parent=None):
        super(PedestrianWorker, self).__init__(parent)
        self._queue = Queue(maxsize=1)
        self._running = False
        self._engine = PedestrianEngine()
        self._last_error = None
        self._inference_timestamps = deque(maxlen=10)

    @pyqtSlot(object)
    def submit_frame(self, bgr_frame):
        if not self._running:
            return
        item = (bgr_frame, time.perf_counter_ns())
        try:
            self._queue.put_nowait(item)
        except Full:
            try:
                self._queue.get_nowait()
            except Empty:
                pass
            try:
                self._queue.put_nowait(item)
            except Full:
                return
            PERF.increment("pedestrian_queue_dropped")
            PERF.event("主动跳帧（正常）", 0.0,
                       "Pedestrian容量1队列覆盖旧帧")

    def stop(self):
        self._running = False

    def run(self):
        self._running = True
        try:
            self.status_ready.emit("行人模型加载中...")
            self._engine.load()
            self.status_ready.emit("行人模型已就绪")
            while self._running:
                try:
                    bgr_frame, submitted_ns = self._queue.get(timeout=0.1)
                except Empty:
                    continue
                started_ns = time.perf_counter_ns()
                try:
                    dets = self._engine.infer(bgr_frame)
                    elapsed_ms = (time.perf_counter_ns() - started_ns) / 1e6
                    self._inference_timestamps.append(time.perf_counter_ns())
                    update_rate = "检测更新 -- Hz"
                    if len(self._inference_timestamps) >= 2:
                        rate = (len(self._inference_timestamps) - 1) / (
                            (self._inference_timestamps[-1] - self._inference_timestamps[0]) / 1e9)
                        update_rate = "检测更新 %.1f Hz" % rate
                    PERF.event("行人单帧推理完成", elapsed_ms)
                    self._last_error = None
                    self.result_ready.emit(dets, elapsed_ms,
                                           "检出 %d 名行人 · %s" % (len(dets), update_rate))
                except Exception as exc:
                    message = "行人推理异常：%s" % exc
                    if message != self._last_error:
                        LOGGER.exception(message)
                        self._last_error = message
                    self.error_occurred.emit(message)
        except Exception as exc:
            message = "行人模型加载失败：%s" % exc
            LOGGER.exception(message)
            self.error_occurred.emit(message)
        finally:
            self._engine.release()


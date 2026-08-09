"""人脸 14 关键点后台线程：异步加载模型，并只处理最新摄像头帧。"""

import logging
import time
from collections import deque
from queue import Empty, Full, Queue

from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot

from perf_logging import get_perf_logger
from .face14_infer import Face14Engine


LOGGER = logging.getLogger(__name__)
PERF = get_perf_logger()


class Face14Worker(QThread):
    """容量为 1 的跳帧 worker，采集和 GUI 不等待推理完成。"""

    status_ready = pyqtSignal(str)
    result_ready = pyqtSignal(object, object, float, float, str)
    error_occurred = pyqtSignal(str)

    def __init__(self, yunet_session, parent=None):
        super(Face14Worker, self).__init__(parent)
        self._queue = Queue(maxsize=1)
        self._running = False
        self._engine = Face14Engine(yunet_session)
        self._last_error = None
        self._dropped_count = 0
        self._inference_timestamps = deque(maxlen=10)

    @pyqtSlot(object)
    def submit_frame(self, bgr_frame):
        if not self._running:
            return
        item = (bgr_frame, time.perf_counter_ns())
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
                self._dropped_count += 1
                PERF.increment("face14_queue_dropped")
                PERF.event("主动跳帧（正常）", 0.0,
                           "Face14容量1队列覆盖旧帧 累计=%d" % self._dropped_count)

    def stop(self):
        self._running = False

    def run(self):
        self._running = True
        import threading
        threading.current_thread().name = "Face14Worker"
        load_started_ns = time.perf_counter_ns()
        self.status_ready.emit("模型加载中...")
        try:
            self._engine.load()
        except Exception as exc:
            message = "模型加载失败：%s" % exc
            LOGGER.exception(message)
            self.error_occurred.emit(message)
            self._engine.release()
            return

        if not self._running:
            self._engine.release()
            return
        PERF.event("模型加载完成",
                   (time.perf_counter_ns() - load_started_ns) / 1e6)
        self.status_ready.emit("模型已就绪")

        try:
            while self._running:
                try:
                    bgr_frame, submitted_ns = self._queue.get(timeout=0.1)
                except Empty:
                    continue
                frame_started_ns = time.perf_counter_ns()
                PERF.set_gauge("face14_queue_depth", self._queue.qsize())
                PERF.event("推理帧开始",
                           (frame_started_ns - submitted_ns) / 1e6,
                           "最新帧排队延迟")
                started = time.perf_counter()
                try:
                    face14, face_box, score = self._engine.infer(bgr_frame)
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    PERF.increment("inference_frames")
                    self._inference_timestamps.append(time.perf_counter_ns())
                    update_rate_text = "检测更新 -- Hz"
                    if len(self._inference_timestamps) >= 2:
                        inference_fps = (len(self._inference_timestamps) - 1) / (
                            (self._inference_timestamps[-1] - self._inference_timestamps[0]) / 1e9)
                        PERF.set_gauge("inference_fps", inference_fps)
                        update_rate_text = "检测更新 %.1f Hz" % inference_fps
                    PERF.event("推理完成", elapsed_ms)
                    total_ms = (time.perf_counter_ns() - frame_started_ns) / 1e6
                    PERF.event("Face14单帧总耗时", total_ms,
                               "从队列取出到结果准备完成")
                    self._last_error = None
                    if face14 is None:
                        text = "未检测到人脸 · %s" % update_rate_text
                        self.result_ready.emit(None, None, 0.0, elapsed_ms, text)
                    else:
                        text = "已检测 · 置信度 %.3f · %s" % (score, update_rate_text)
                        self.result_ready.emit(face14, face_box, score, elapsed_ms, text)
                except Exception as exc:
                    total_ms = (time.perf_counter_ns() - frame_started_ns) / 1e6
                    PERF.event("Face14单帧总耗时", total_ms,
                               "从队列取出到异常结束", level="WARNING")
                    message = "推理异常：%s" % exc
                    if message != self._last_error:
                        LOGGER.exception(message)
                        self._last_error = message
                    self.error_occurred.emit(message)
        finally:
            self._engine.release()

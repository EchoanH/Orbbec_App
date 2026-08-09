"""摄像头采集线程：只做取帧与信号发送，不操作任何 QWidget。"""

import logging
import time
from collections import deque

from PyQt5.QtCore import QThread, pyqtSignal

from perf_logging import get_perf_logger

LOGGER = logging.getLogger(__name__)
PERF = get_perf_logger()


class CaptureThread(QThread):
    """常驻采集线程，为后续推理线程保留独立边界。"""

    frame_ready = pyqtSignal(object, object)
    status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    fps_changed = pyqtSignal(float)

    def __init__(self, source, parent=None):
        super(CaptureThread, self).__init__(parent)
        self._source = source
        self._running = False

    def stop(self):
        self._running = False

    def run(self):
        import threading
        threading.current_thread().name = "CaptureThread"
        self._running = True
        try:
            self._source.start()
            self.status_changed.emit("摄像头已连接")
        except Exception as exc:
            LOGGER.warning("摄像头启动失败：%s", exc)
            self._source.stop()
            self.error_occurred.emit(str(exc))
            self.status_changed.emit("摄像头未连接")
            return

        frame_count = 0
        last_tick = time.monotonic()
        capture_timestamps = deque(maxlen=10)
        try:
            while self._running:
                packet = self._source.read()
                if packet is None:
                    PERF.event("采集帧缺失", 0.0, level="WARNING")
                    continue
                bgr, depth = packet
                captured_ns = time.perf_counter_ns()
                capture_timestamps.append(captured_ns)
                PERF.increment("capture_frames")
                PERF.event("采集到新帧", 0.0,
                           "frame=%d" % (frame_count + 1))
                if len(capture_timestamps) == 10:
                    actual_fps = 9.0 / ((capture_timestamps[-1] - capture_timestamps[0]) / 1e9)
                    PERF.set_gauge("capture_fps", actual_fps)
                    PERF.event("采集FPS统计", 0.0, "最近10帧=%.1f" % actual_fps)
                self.frame_ready.emit(bgr, depth)
                frame_count += 1
                now = time.monotonic()
                if now - last_tick >= 1.0:
                    self.fps_changed.emit(frame_count / (now - last_tick))
                    frame_count = 0
                    last_tick = now
        except Exception as exc:
            LOGGER.exception("采集线程异常")
            self.error_occurred.emit("采集异常：%s" % exc)
        finally:
            self._source.stop()
            self.status_changed.emit("摄像头已停止")

"""摄像头采集线程：只做取帧与信号发送，不操作任何 QWidget。"""

import logging
import time
from collections import deque

import cv2
from PyQt5.QtCore import QThread, pyqtSignal

from perf_logging import get_perf_logger

from .orbbec_source import SKIPPED

LOGGER = logging.getLogger(__name__)
PERF = get_perf_logger()

# 解码节流：设为 0 表示完全不节流（推荐）。
# 曾设为 20/30 做节流，但因闸门以"解码结束时刻"为基准、而帧只在 33.3ms 的
# 整数倍到达，差几毫秒就要多等整整一帧，实测锁死在 10fps（拍频效应）。
# 现改为不节流；CPU 余量已由推理节流与深度帧优化腾出。
TARGET_DECODE_FPS = 0.0
MIN_DECODE_INTERVAL_NS = (int(1e9 / TARGET_DECODE_FPS)
                          if TARGET_DECODE_FPS > 0 else 0)

# 显示放大：源帧 1280x720 小于 video_label 1644x960，直接显示会四周留边。
# 主线程放大已排除（QPixmap.scaled 9.7ms/帧），改在采集线程预放大，
# 主线程 paintEvent 耗时不变。
# 高度取 924 而非 925：可被 4 整除，内存对齐更友好；宽高比偏差 0.08%，
# 累积不到 1.4px，不可见。
# 推理链路永远使用原始帧（1280x720），放大帧只用于显示：
# face14_infer.yunet_get_facebox 首行即 resize 到 640x640，输入变大会
# 抬高前处理成本；且 color_to_depth_point 按传入帧 shape 换算，彩色帧
# 尺寸变了而深度帧没变会导致距离计算错位。
UPSCALE_ENABLED = True
DISPLAY_TARGET = (1644, 924)
UPSCALE_INTERP = cv2.INTER_LINEAR


class CaptureThread(QThread):
    """常驻采集线程，为后续推理线程保留独立边界。"""

    # 三参信号：(bgr 原始帧, bgr_display 放大显示帧, depth)。
    # 推理 worker 引用 bgr 原始帧，display 只用于 UI 绘制。
    frame_ready = pyqtSignal(object, object, object)
    status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    fps_changed = pyqtSignal(float)

    def __init__(self, source, parent=None):
        super(CaptureThread, self).__init__(parent)
        self._source = source
        self._running = False

    def stop(self):
        self._running = False

    def _prepare_display_frame(self, bgr):
        """生成显示帧：在采集线程预放大，主线程 paintEvent 耗时不变。

        放大开关关闭、或源帧尺寸已等于目标尺寸时直接复用 bgr 本身
        （同一对象，零拷贝）。放大失败时降级为原帧，保证采集线程不崩。
        """
        if not UPSCALE_ENABLED:
            return bgr
        src_height, src_width = bgr.shape[:2]
        if (src_width, src_height) == DISPLAY_TARGET:
            return bgr
        started_ns = time.perf_counter_ns()
        try:
            bgr_display = cv2.resize(bgr, DISPLAY_TARGET,
                                     interpolation=UPSCALE_INTERP)
            PERF.event("显示帧放大完成",
                       (time.perf_counter_ns() - started_ns) / 1e6,
                       "采集线程预放大 %dx%d -> %s" % (
                           src_width, src_height, DISPLAY_TARGET))
            return bgr_display
        except Exception as exc:
            LOGGER.warning("显示帧放大失败，降级为原始帧：%s", exc)
            return bgr

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
        last_decode_ns = 0
        capture_timestamps = deque(maxlen=10)
        try:
            while self._running:
                now_ns = time.perf_counter_ns()
                skip_decode = (MIN_DECODE_INTERVAL_NS > 0 and
                               (now_ns - last_decode_ns) < MIN_DECODE_INTERVAL_NS)
                packet = self._source.read(skip_decode=skip_decode)
                if packet is SKIPPED:
                    continue
                if packet is None:
                    PERF.event("采集帧缺失", 0.0, level="WARNING")
                    continue
                last_decode_ns = time.perf_counter_ns()
                bgr, depth = packet
                captured_ns = last_decode_ns
                capture_timestamps.append(captured_ns)
                PERF.increment("capture_frames")
                if len(capture_timestamps) == 10:
                    actual_fps = 9.0 / ((capture_timestamps[-1] - capture_timestamps[0]) / 1e9)
                    PERF.set_gauge("capture_fps", actual_fps)
                # bgr 保持原始帧引用（不拷贝不修改），推理 worker 会引用它；
                # bgr_display 为放大显示帧，UI 绘制底图用它。
                bgr_display = self._prepare_display_frame(bgr)
                self.frame_ready.emit(bgr, bgr_display, depth)
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
"""所有演示页共享的接口与视频画布。"""

import time
from collections import deque
from typing import Optional, Tuple

import numpy as np
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from perf_logging import get_perf_logger


PERF = get_perf_logger()
UI_TIMESTAMPS = deque(maxlen=10)


class PerfVideoLabel(QLabel):
    """仅包装 QLabel 的绘制回调，记录 setPixmap 到 paintEvent 的间隔。"""

    def __init__(self, parent=None):
        super(PerfVideoLabel, self).__init__(parent)
        self._setpixmap_ns = None

    def mark_setpixmap(self, timestamp_ns):
        self._setpixmap_ns = timestamp_ns

    def paintEvent(self, event):
        started_ns = time.perf_counter_ns()
        if self._setpixmap_ns is not None:
            PERF.event("UI屏幕重绘开始",
                       (started_ns - self._setpixmap_ns) / 1e6)
            self._setpixmap_ns = None
        super(PerfVideoLabel, self).paintEvent(event)
        PERF.event("UI paintEvent完成",
                   (time.perf_counter_ns() - started_ns) / 1e6)


class BasePage(QWidget):
    """提供统一帧处理接口，后续仅替换 process_frame 内部推理实现。"""

    page_title = "视觉演示"
    page_hint = "实时画面"

    def __init__(self, parent=None):
        super(BasePage, self).__init__(parent)
        self._last_pixmap = None
        self.video_label = PerfVideoLabel()
        self.video_label.setObjectName("videoLabel")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setText("等待摄像头画面")
        self.video_label.setScaledContents(False)
        panel = QFrame()
        panel.setObjectName("videoPanel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(10, 10, 10, 10)
        panel_layout.addWidget(self.video_label)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.addWidget(panel, 1)

    def process_frame(self, bgr_frame, bgr_display=None, depth_frame=None) -> Tuple[object, str]:
        """统一帧处理接口。

        bgr_frame：原始帧（1280x720），推理链路永远用它。
        bgr_display：采集线程预放大的显示帧（1644x924），只用于绘制底图；
        为 None 时页面回退使用 bgr_frame（行为与改造前一致）。
        depth_frame：深度帧，按页需求使用。
        """
        return bgr_frame, ""

    def compute_display_scale(self, source_shape, display_shape):
        """返回 (scale_x, scale_y)。宽高独立计算，避免 0.08% 比例偏差累积。

        display_shape 为 None 或与 source 相同时返回 (1.0, 1.0)。
        worker 返回的坐标位于 source（原始帧）坐标系，绘制到显示帧上时
        按此系数换算：x 乘 scale_x，y 乘 scale_y。
        """
        if display_shape is None:
            return 1.0, 1.0
        source_height, source_width = source_shape[:2]
        display_height, display_width = display_shape[:2]
        if (source_width, source_height) == (display_width, display_height):
            return 1.0, 1.0
        return (display_width / float(source_width),
                display_height / float(source_height))

    def on_activated(self):
        pass

    def on_deactivated(self):
        pass

# P0 UI 渲染性能优化：绘制前先降采样，坐标同步缩放。
    def compute_target_size(self, source_width, source_height):
        """按视频画布等比计算目标尺寸，并返回宽度缩放比例。

        重要：只缩小，绝不放大。
        显示帧已由采集线程预放大到 1644x924，接近 video_label 1644x960；
        只有画布小于显示帧（如窗口被缩小）时才降采样，画布更大时返回
        原始尺寸与 scale=1.0。放大的昂贵路径（主线程 QPixmap.scaled
        实测 9.7ms/帧）已在采集线程解决，此处绝不执行放大。
        """
        source_width = int(source_width)
        source_height = int(source_height)
        if source_width <= 0 or source_height <= 0:
            return max(1, source_width), max(1, source_height), 1.0
        label_size = self.video_label.size()
        if label_size.width() <= 0 or label_size.height() <= 0:
            return source_width, source_height, 1.0
        fitted_size = QSize(source_width, source_height).scaled(
            label_size, Qt.KeepAspectRatio)
        target_width = max(1, fitted_size.width())
        target_height = max(1, fitted_size.height())
        if target_width >= source_width or target_height >= source_height:
            return source_width, source_height, 1.0
        return target_width, target_height, target_width / float(source_width)

    def show_frame(self, bgr_frame: Optional[np.ndarray]):
        if bgr_frame is None or getattr(bgr_frame, "size", 0) == 0:
            return
        frame = bgr_frame
        copied = False
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)
            copied = True
        height, width = frame.shape[:2]
        image_started_ns = time.perf_counter_ns()
        if hasattr(QImage, "Format_BGR888"):
            image = QImage(frame.data, width, height, int(frame.strides[0]),
                           QImage.Format_BGR888)
            PERF.event("UI BGR转RGB", 0.0,
                       "跳过颜色转换 Format_BGR888 copy=%s" % copied)
        else:
            rgb = frame[:, :, ::-1].copy()
            image = QImage(rgb.data, width, height, int(rgb.strides[0]),
                           QImage.Format_RGB888)
            PERF.event("UI BGR转RGB",
                       (time.perf_counter_ns() - image_started_ns) / 1e6,
                       "Qt无Format_BGR888，执行兼容拷贝")
        image_finished_ns = time.perf_counter_ns()
        PERF.event("UI numpy转QImage",
                   (image_finished_ns - image_started_ns) / 1e6,
                   "bytesPerLine=%d copy=%s" % (int(frame.strides[0]), copied))
        pixmap_started_ns = time.perf_counter_ns()
        self._last_pixmap = QPixmap.fromImage(image)
        pixmap_finished_ns = time.perf_counter_ns()
        PERF.event("UI QImage转QPixmap",
                   (pixmap_finished_ns - pixmap_started_ns) / 1e6)
        self._refresh_pixmap()

    def resizeEvent(self, event):
        super(BasePage, self).resizeEvent(event)
        self._refresh_pixmap(False)

    def _refresh_pixmap(self, count_frame=True):
        """P0 优化：不再做 QPixmap.scaled（实测 9.7ms/帧）。

        显示帧已由采集线程预放大到 1644x924，与 video_label 1644x960 几乎
        一致，直接 setPixmap 即可；保持 setScaledContents(False) 零缩放，
        paint 阶段无任何拉伸开销。
        """
        if self._last_pixmap is None:
            return
        set_started_ns = time.perf_counter_ns()
        self.video_label.setPixmap(self._last_pixmap)
        set_finished_ns = time.perf_counter_ns()
        self.video_label.mark_setpixmap(set_finished_ns)
        if count_frame:
            UI_TIMESTAMPS.append(set_finished_ns)
            PERF.increment("ui_frames")
            if len(UI_TIMESTAMPS) >= 2:
                ui_fps = (len(UI_TIMESTAMPS) - 1) / (
                    (UI_TIMESTAMPS[-1] - UI_TIMESTAMPS[0]) / 1e9)
                PERF.set_gauge("ui_fps", ui_fps)
            PERF.event("UI setPixmap完成",
                       (set_finished_ns - set_started_ns) / 1e6,
                       "帧到setPixmap完成=%0.1fms" % (
                           (set_finished_ns - getattr(
                               self, "_perf_frame_received_ns", set_finished_ns)) / 1e6))
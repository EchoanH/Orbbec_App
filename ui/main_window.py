"""全屏主窗口：固定导航、常驻采集线程和当前页帧路由。"""

import logging
import time
from typing import List, Tuple

from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtWidgets import (QFrame, QHBoxLayout, QLabel, QMainWindow,
                             QPushButton, QStackedWidget, QVBoxLayout,
                             QWidget)

from camera.capture_thread import CaptureThread
from camera.orbbec_source import OrbbecSource
from perf_logging import get_perf_logger
from ui.pages.enroll_page import EnrollPage
from ui.pages.face14_page import Face14Page
from ui.pages.gesture_page import GesturePage
from ui.pages.pedestrian_page import PedestrianPage


LOGGER = logging.getLogger(__name__)
PERF = get_perf_logger()

# 渲染节拍：主线程按此间隔取最新帧渲染。
# 20ms(50Hz) 快于相机的 33.3ms 出帧间隔，确保每帧到达后都能在下一拍被取走；
# QTimer 是"最少间隔"而非精确调度，设成 33 会因抖动错过整拍，实测掉到 22fps。
# 没有新帧时 _render_latest 直接返回，空转成本可忽略。
RENDER_INTERVAL_MS = 16

class MainWindow(QMainWindow):
    """应用壳层；所有页面共享一个采集线程，切页不重启摄像头。"""

    NAV_ITEMS: List[Tuple[str, str]] = [
        ("人脸关键点检测", "face14"),
        ("行人检测", "pedestrian"),
        ("手势识别", "gesture"),
        ("人脸录入", "enroll"),
    ]

    def __init__(self, yunet_session):
        super(MainWindow, self).__init__()
        self.setWindowTitle("AI 综合实验箱")
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.source = OrbbecSource()
        self.capture_thread = CaptureThread(self.source, self)
        self._perf_last_ui_frame_ns = None
        # 最新帧槽位：采集线程只写，主线程只读，写入为原子引用赋值。
        # 取代原 InferenceThread 直通层的"容量1队列丢旧帧"语义。
        self._latest_bgr = None
        # 显示帧槽位：采集线程预放大的 1644x924 帧，只用于绘制底图；
        # 推理 worker 引用的始终是 _latest_bgr 原始帧。
        self._latest_bgr_display = None
        self._latest_depth = None
        self._latest_stamp_ns = None
        self._rendered_stamp_ns = None
        self._dropped_frames = 0
        self.yunet_session = yunet_session
        self.pages = [Face14Page(self.yunet_session), PedestrianPage(),
                      GesturePage(), EnrollPage(self.yunet_session)]
        self.nav_buttons = []
        self._build_ui()
        self.capture_thread.frame_ready.connect(self._on_capture_frame)
        self.capture_thread.status_changed.connect(self._on_status)
        self.capture_thread.error_occurred.connect(self._on_error)
        self.capture_thread.fps_changed.connect(self._on_fps)
        self.render_timer = QTimer(self)
        self.render_timer.setTimerType(Qt.PreciseTimer)
        self.render_timer.timeout.connect(self._render_latest)
        self.render_timer.start(RENDER_INTERVAL_MS)
        self.capture_thread.start()

    def _build_ui(self):
        root = QWidget()
        root.setObjectName("content")
        outer = QHBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_sidebar())
        outer.addWidget(self._build_content(), 1)
        self.setCentralWidget(root)

    def _build_sidebar(self):
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(208)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 28, 0, 20)
        layout.setSpacing(0)
        brand = QLabel("AI 综合实验箱")
        brand.setObjectName("brand")
        brand.setContentsMargins(18, 0, 12, 0)
        layout.addWidget(brand)
        sub = QLabel("ATLAS 200I DK A2")
        sub.setObjectName("brandSub")
        sub.setContentsMargins(18, 5, 12, 28)
        layout.addWidget(sub)
        section = QLabel("演示模块")
        section.setObjectName("sectionLabel")
        section.setContentsMargins(18, 0, 12, 9)
        layout.addWidget(section)
        for index, (label, _) in enumerate(self.NAV_ITEMS):
            button = QPushButton(label)
            button.setObjectName("navButton")
            button.setCheckable(True)
            button.clicked.connect(lambda checked, i=index: self._switch_page(i))
            layout.addWidget(button)
            self.nav_buttons.append(button)
        layout.addStretch(1)
        foot = QLabel("CAMERA / READY\nESC 退出全屏")
        foot.setObjectName("brandSub")
        foot.setContentsMargins(18, 0, 12, 0)
        layout.addWidget(foot)
        self.nav_buttons[0].setChecked(True)
        return sidebar

    def _build_content(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        topbar = QFrame()
        topbar.setObjectName("topBar")
        bar = QHBoxLayout(topbar)
        bar.setContentsMargins(24, 16, 16, 16)
        self.title_label = QLabel()
        self.title_label.setObjectName("pageTitle")
        self.hint_label = QLabel()
        self.hint_label.setObjectName("pageHint")
        bar.addWidget(self.title_label)
        bar.addSpacing(14)
        bar.addWidget(self.hint_label)
        bar.addStretch(1)
        self.fps_label = QLabel("FPS --")
        self.fps_label.setObjectName("fpsLabel")
        bar.addWidget(self.fps_label)
        self.status_label = QLabel("摄像头初始化中")
        self.status_label.setObjectName("statusPill")
        bar.addWidget(self.status_label)
        close_button = QPushButton()
        close_button.setObjectName("closeButton")
        close_button.setFixedSize(34, 28)
        close_button.setToolTip("退出")
        close_button.clicked.connect(self.close)
        bar.addWidget(close_button)
        layout.addWidget(topbar)
        self.stack = QStackedWidget()
        for page in self.pages:
            self.stack.addWidget(page)
        layout.addWidget(self.stack, 1)
        self._update_header(0)
        return container

    def _update_header(self, index):
        page = self.pages[index]
        self.title_label.setText(page.page_title)
        self.hint_label.setText(page.page_hint)

    def _switch_page(self, index):
        if index == self.stack.currentIndex():
            return
        self.pages[self.stack.currentIndex()].on_deactivated()
        self.stack.setCurrentIndex(index)
        self.pages[index].on_activated()
        for i, button in enumerate(self.nav_buttons):
            button.setChecked(i == index)
        self._update_header(index)

    @pyqtSlot(object, object, object)
    def _on_capture_frame(self, bgr_frame, bgr_display, depth_frame):
        """采集线程回调：只更新最新帧槽位，不做任何绘制。

        若上一帧尚未被渲染就被覆盖，等同于原 InferenceThread 的主动跳帧。
        """
        if self._latest_stamp_ns is not None and self._rendered_stamp_ns != self._latest_stamp_ns:
            self._dropped_frames += 1
            PERF.increment("pipeline_queue_dropped")
        self._latest_bgr = bgr_frame
        self._latest_bgr_display = bgr_display
        self._latest_depth = depth_frame
        self._latest_stamp_ns = time.perf_counter_ns()

    def _render_latest(self):
        """渲染节拍：取最新帧交给当前页处理与显示。"""
        stamp_ns = self._latest_stamp_ns
        if stamp_ns is None or stamp_ns == self._rendered_stamp_ns:
            return
        bgr_frame = self._latest_bgr
        # 显示帧：采集线程预放大的帧，仅用于绘制；推理仍走 bgr_frame。
        bgr_display = self._latest_bgr_display
        depth_frame = self._latest_depth
        self._rendered_stamp_ns = stamp_ns
        if bgr_frame is None:
            return
        received_ns = time.perf_counter_ns()
        if self._perf_last_ui_frame_ns is not None:
            interval_ms = (received_ns - self._perf_last_ui_frame_ns) / 1e6
            if interval_ms > 50.0:
                PERF.event("UI线程阻塞", interval_ms,
                           "帧间隔超过50ms", level="WARNING")
        self._perf_last_ui_frame_ns = received_ns
        PERF.event("UI收到新帧", (received_ns - stamp_ns) / 1e6,
                   "采集到渲染的等待时间")
        page = self.pages[self.stack.currentIndex()]
        page._perf_frame_received_ns = received_ns
        rendered, result = page.process_frame(bgr_frame, bgr_display,
                                              depth_frame)
        page.show_frame(rendered)
        self.status_label.setText(result or "摄像头已连接")
        finished_ns = time.perf_counter_ns()
        elapsed_ms = (finished_ns - received_ns) / 1e6
        PERF.event("UI帧处理完成", elapsed_ms)
        if elapsed_ms > 50.0:
            PERF.event("UI线程阻塞", elapsed_ms,
                       "单帧处理超过50ms", level="WARNING")

    @pyqtSlot(str)
    def _on_status(self, text):
        self.status_label.setText(text)

    @pyqtSlot(str)
    def _on_error(self, text):
        LOGGER.warning(text)
        self.status_label.setText("摄像头未连接")
        for page in self.pages:
            page.video_label.setText("摄像头未连接\n\n%s" % text)

    @pyqtSlot(float)
    def _on_fps(self, value):
        self.fps_label.setText("FPS %0.1f" % value)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Escape:
            self.close()
            return
        super(MainWindow, self).keyPressEvent(event)

    def closeEvent(self, event):
        if self.render_timer.isActive():
            self.render_timer.stop()
        if self.capture_thread.isRunning():
            self.capture_thread.stop()
            self.capture_thread.wait()
        event.accept()
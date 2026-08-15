#!/usr/bin/env python3
"""独立启动壳层：复用主 GUI 的云台 PID 调试页面。"""

import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow

from camera.capture_thread import CaptureThread
from camera.orbbec_source import OrbbecSource
from style.theme import APP_STYLE
from ui.pages.gimbal_pid_tuner_page import GimbalPIDTunerPage


class GimbalPIDTuner(QMainWindow):
    """独立模式仅拥有相机壳层；PID UI/控制逻辑由共享页面提供。"""

    def __init__(self):
        super(GimbalPIDTuner, self).__init__()
        self.setWindowTitle("独立 PID 云台跟踪调试器")
        self.resize(1680, 980)
        self._shutting_down = False
        self._source = OrbbecSource()
        self._capture_thread = CaptureThread(self._source, parent=self)
        self.page = GimbalPIDTunerPage(parent=self)
        self.setCentralWidget(self.page)
        mode_warning = QLabel("独立模式运行前必须关闭主程序")
        mode_warning.setStyleSheet("color: #ffd58a; font-weight: 700;")
        self.statusBar().addPermanentWidget(mode_warning)
        self.statusBar().showMessage("自动控制默认关闭")
        self._capture_thread.frame_ready.connect(self._on_frame)
        self._capture_thread.status_changed.connect(self._on_camera_status)
        self._capture_thread.error_occurred.connect(self._on_camera_error)
        self._capture_thread.fps_changed.connect(self._on_fps)
        self.page.on_activated()
        self._capture_thread.start()

    @pyqtSlot(object, object, object)
    def _on_frame(self, bgr_frame, bgr_display, depth_frame):
        rendered, result = self.page.process_frame(
            bgr_frame, bgr_display, depth_frame)
        self.page.show_frame(rendered)
        self.statusBar().showMessage(result)

    @pyqtSlot(str)
    def _on_camera_status(self, text):
        self.statusBar().showMessage(text)

    @pyqtSlot(str)
    def _on_camera_error(self, text):
        self.page.on_camera_error(text)
        self.statusBar().showMessage("摄像头未连接：%s" % text)

    @pyqtSlot(float)
    def _on_fps(self, fps):
        self.statusBar().showMessage("采集帧率 %.1f" % float(fps))

    def closeEvent(self, event):
        self._shutdown()
        super(GimbalPIDTuner, self).closeEvent(event)

    def _shutdown(self):
        if self._shutting_down:
            return
        self._shutting_down = True
        self.page.on_deactivated()
        self._capture_thread.stop()
        if self._capture_thread.isRunning():
            self._capture_thread.wait()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    app = QApplication(sys.argv)
    app.setApplicationName("独立 PID 云台跟踪调试器")
    app.setFont(QFont("Droid Sans Fallback", 11))
    app.setStyleSheet(APP_STYLE)
    window = GimbalPIDTuner()
    app.aboutToQuit.connect(window._shutdown)
    window.showMaximized()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())

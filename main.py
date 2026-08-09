"""应用入口：设置中文字体、日志与全屏主窗口。"""

import logging
import os
import sys

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication

from inference.yunet_session import YuNetSession
from perf_logging import get_perf_logger
from style.theme import APP_STYLE
from ui.main_window import MainWindow


def configure_logging():
    """统一日志格式，便于现场演示时定位设备或线程问题。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main():
    configure_logging()
    get_perf_logger().event("应用启动", detail="DISPLAY=%s" % os.environ.get("DISPLAY", "<未设置>"))
    app = QApplication(sys.argv)
    app.setApplicationName("AI 综合实验箱")
    app.setFont(QFont("Droid Sans Fallback", 11))
    app.setStyleSheet(APP_STYLE)
    yunet_session = YuNetSession()
    window = MainWindow(yunet_session)
    app.aboutToQuit.connect(yunet_session.release)
    window.showFullScreen()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())

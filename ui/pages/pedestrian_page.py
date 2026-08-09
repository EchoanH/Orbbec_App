"""行人检测演示页。"""

import logging

from PyQt5.QtWidgets import QApplication

from inference.pedestrian_infer import draw_pedestrians
from inference.pedestrian_worker import PedestrianWorker

from .base_page import BasePage


LOGGER = logging.getLogger(__name__)


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
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._shutdown_worker)

    def process_frame(self, bgr_frame, depth_frame=None):
        if self._worker is not None:
            self._worker.submit_frame(bgr_frame)
        if not self._latest_dets:
            return bgr_frame, self._status_text
        return draw_pedestrians(bgr_frame, self._latest_dets), self._status_text

    def on_activated(self):
        self._active = True
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

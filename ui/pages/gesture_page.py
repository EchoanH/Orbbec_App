"""手势识别演示页。"""

from PyQt5.QtWidgets import QApplication

from inference.gesture_infer import draw
from inference.gesture_worker import GestureWorker

from .base_page import BasePage


class GesturePage(BasePage):
    page_title = "手势识别"
    page_hint = "palm + handpose · 四类手势 · NPU 实时推理"

    def __init__(self, parent=None):
        super(GesturePage, self).__init__(parent)
        self._worker = None
        self._active = False
        self._latest_lm = None
        self._latest_gesture = ""
        self._status_text = "手势模型加载中..."
        self._stop_requested = False
        self._restart_when_finished = False
        self._pending_gesture = None
        self._pending_count = 0
        self._stable_gesture = ""
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._shutdown_worker)

    def process_frame(self, bgr_frame, depth_frame=None):
        if self._worker is not None:
            self._worker.submit_frame(bgr_frame)
        if self._latest_lm is None or not self._stable_gesture:
            return bgr_frame, self._status_text
        return draw(bgr_frame.copy(), self._latest_lm, self._stable_gesture), self._status_text

    def on_activated(self):
        self._active = True
        if self._worker is None:
            self._start_worker()
        elif self._stop_requested:
            self._restart_when_finished = True

    def on_deactivated(self):
        self._active = False
        self._latest_lm = None
        self._latest_gesture = ""
        self._stable_gesture = ""
        self._pending_gesture = None
        self._pending_count = 0
        self._restart_when_finished = False
        if self._worker is not None:
            self._stop_requested = True
            self._worker.stop()
            if self._worker.isRunning():
                self._worker.wait()

    def _start_worker(self):
        self._status_text = "手势模型加载中..."
        self._stop_requested = False
        worker = GestureWorker(self)
        worker.status_ready.connect(self._on_status)
        worker.result_ready.connect(self._on_result)
        worker.error_occurred.connect(self._on_error)
        worker.finished.connect(self._on_finished)
        self._worker = worker
        worker.start()

    def _on_status(self, text):
        self._status_text = text

    def _on_result(self, lm, gesture, conf, status):
        if not self._active or self._stop_requested:
            return
        self._latest_lm = lm
        if gesture:
            if gesture == self._pending_gesture:
                self._pending_count += 1
            else:
                self._pending_gesture = gesture
                self._pending_count = 1
            if self._pending_count >= 3:
                self._stable_gesture = gesture
            self._latest_gesture = gesture
        else:
            self._latest_gesture = ""
            self._stable_gesture = ""
            self._pending_gesture = None
            self._pending_count = 0
        self._status_text = "手势：%s · 置信度 %.2f · %s" % (
            self._stable_gesture or "识别中", conf, status)

    def _on_error(self, text):
        self._latest_lm = None
        self._stable_gesture = ""
        self._status_text = text

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

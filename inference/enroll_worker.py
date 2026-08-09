"""人脸录入/识别后台 worker，共享进程级 YuNet，会话调用由单例加锁。"""

import gc
import logging
import os
import threading
import time
from queue import Empty, Full, Queue

from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot

from perf_logging import get_perf_logger
from .face_db_adapter import (DEVICE_ID, MATCH_CONF_TH, SFACE, build_index,
                               detect_face, enroll_person,
                               extract_feature_for_enroll,
                               extract_feature_for_match, match_feature)


LOGGER = logging.getLogger(__name__)
PERF = get_perf_logger()


class EnrollWorker(QThread):
    status_ready = pyqtSignal(str)
    match_ready = pyqtSignal(object, object, float, str)
    enroll_progress = pyqtSignal(int, int, float, str)
    enroll_complete = pyqtSignal(str, object)
    error_occurred = pyqtSignal(str)

    def __init__(self, yunet_session, parent=None):
        super(EnrollWorker, self).__init__(parent)
        self._yunet_session = yunet_session
        self._queue = Queue(maxsize=1)
        self._running = False
        self._sface_session = None
        self._yunet_idx = None
        self._mode_lock = threading.Lock()
        self._mode = "match"
        self._enroll_name = ""
        self._enroll_target = 3
        self._enroll_feats = []
        self._last_error = None

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
            PERF.increment("enroll_queue_dropped")
            PERF.event("主动跳帧（正常）", 0.0,
                       "Enroll容量1队列覆盖旧帧")

    def start_enrollment(self, name, target):
        with self._mode_lock:
            self._mode = "enroll"
            self._enroll_name = name
            self._enroll_target = target
            self._enroll_feats = []

    def cancel_enrollment(self):
        with self._mode_lock:
            self._mode = "match"
            self._enroll_name = ""
            self._enroll_feats = []

    def stop(self):
        self._running = False

    def run(self):
        self._running = True
        try:
            self.status_ready.emit("人脸模型加载中...")
            self._yunet_session.load()
            self._yunet_idx = build_index(self._yunet_session)
            from ais_bench.infer.interface import InferSession
            if not os.path.isfile(SFACE):
                raise RuntimeError("SFace 模型不存在：%s" % SFACE)
            self._sface_session = InferSession(DEVICE_ID, SFACE)
            self.status_ready.emit("人脸模型已就绪")
            while self._running:
                try:
                    bgr_frame, submitted_ns = self._queue.get(timeout=0.1)
                except Empty:
                    continue
                try:
                    with self._mode_lock:
                        mode = self._mode
                    if mode == "enroll":
                        self._process_enroll(bgr_frame)
                    else:
                        self._process_match(bgr_frame)
                except Exception as exc:
                    message = "人脸处理异常：%s" % exc
                    if message != self._last_error:
                        LOGGER.exception(message)
                        self._last_error = message
                    self.error_occurred.emit(message)
        except Exception as exc:
            message = "人脸模型加载失败：%s" % exc
            LOGGER.exception(message)
            self.error_occurred.emit(message)
        finally:
            if self._sface_session is not None:
                del self._sface_session
                self._sface_session = None
                gc.collect()
            self._yunet_idx = None

    def _process_enroll(self, bgr_frame):
        feat, score, reason = extract_feature_for_enroll(
            self._yunet_session, self._yunet_idx, self._sface_session, bgr_frame)
        with self._mode_lock:
            if self._mode != "enroll":
                return
            if feat is not None:
                self._enroll_feats.append(feat)
            count = len(self._enroll_feats)
            target = self._enroll_target
            name = self._enroll_name
        self.enroll_progress.emit(count, target, float(score or 0.0), reason or "采集成功")
        if count >= target:
            record = enroll_person(name, self._enroll_feats)
            with self._mode_lock:
                self._mode = "match"
                self._enroll_name = ""
                self._enroll_feats = []
            self.enroll_complete.emit(name, record)

    def _process_match(self, bgr_frame):
        feat, score, reason = extract_feature_for_match(
            self._yunet_session, self._yunet_idx, self._sface_session, bgr_frame)
        box = None
        if feat is not None:
            face = detect_face(self._yunet_session, self._yunet_idx, bgr_frame, MATCH_CONF_TH)
            if face is not None:
                box = face[1].tolist()
            name, similarity = match_feature(feat)
            text = reason or ("已匹配" if name else "未知")
            self.match_ready.emit(box, name, float(similarity), text)
        else:
            self.match_ready.emit(None, None, float(score or 0.0), reason or "未检测到人脸")

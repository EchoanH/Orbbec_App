"""人脸录入/识别后台 worker，共享进程级 YuNet，会话调用由单例加锁。"""

import gc
import logging
import os
import threading
import time
import traceback
from queue import Empty, Full, Queue

from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot

from perf_logging import get_perf_logger
from .face_tracking import decode_face_boxes
from .face_db_adapter import (DEVICE_ID, MATCH_CONF_TH, SFACE, build_index,
                               detect_face, enroll_person,
                               extract_feature_for_enroll,
                               extract_feature_for_match, match_feature)


LOGGER = logging.getLogger(__name__)
PERF = get_perf_logger()


def _log_worker_exception(stage, exc, elapsed_ms=0.0):
    stack = traceback.format_exc()
    message = "%s: %s\n%s" % (stage, exc, stack)
    LOGGER.error(message)
    PERF.event("EnrollWorker exception", elapsed_ms, message, level="WARNING")


class EnrollWorker(QThread):
    status_ready = pyqtSignal(str)
    match_ready = pyqtSignal(object, object, float, float, str)
    face_candidates_ready = pyqtSignal(object, object)
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
        self._face_tracking_enabled = threading.Event()

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
            except Empty as exc:
                _log_worker_exception(
                    "EnrollWorker queue replacement found no old frame", exc)
            try:
                self._queue.put_nowait(item)
            except Full as exc:
                _log_worker_exception(
                    "EnrollWorker queue replacement still full", exc)
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

    def set_face_tracking_enabled(self, enabled):
        if enabled:
            self._face_tracking_enabled.set()
        else:
            self._face_tracking_enabled.clear()

    def stop(self):
        self._running = False

    def run(self):
        self._running = True
        try:
            self.status_ready.emit("人脸模型加载中...")
            yunet_started_ns = time.perf_counter_ns()
            PERF.event("EnrollWorker YuNet load start", detail="shared singleton")
            try:
                self._yunet_session.load()
                self._yunet_idx = build_index(self._yunet_session)
            except Exception as exc:
                _log_worker_exception(
                    "EnrollWorker YuNet load failed", exc,
                    (time.perf_counter_ns() - yunet_started_ns) / 1e6)
                raise
            PERF.event("EnrollWorker YuNet load complete",
                       (time.perf_counter_ns() - yunet_started_ns) / 1e6,
                       "shared singleton")
            from ais_bench.infer.interface import InferSession
            sface_started_ns = time.perf_counter_ns()
            PERF.event("EnrollWorker SFace load start", detail=SFACE)
            try:
                if not os.path.isfile(SFACE):
                    raise RuntimeError("SFace 模型不存在：%s" % SFACE)
                self._sface_session = InferSession(DEVICE_ID, SFACE)
            except Exception as exc:
                _log_worker_exception(
                    "EnrollWorker SFace load failed", exc,
                    (time.perf_counter_ns() - sface_started_ns) / 1e6)
                raise
            PERF.event("EnrollWorker SFace load complete",
                       (time.perf_counter_ns() - sface_started_ns) / 1e6,
                       SFACE)
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
                    _log_worker_exception("EnrollWorker frame processing failed", exc)
                    self._last_error = message
                    self.error_occurred.emit(message)
        except Exception as exc:
            message = "人脸模型加载失败：%s" % exc
            _log_worker_exception("EnrollWorker run failed", exc)
            self.error_occurred.emit(message)
        finally:
            release_started_ns = time.perf_counter_ns()
            PERF.event("EnrollWorker model release start",
                       detail="SFace release; YuNet singleton retained")
            if self._sface_session is not None:
                del self._sface_session
                self._sface_session = None
                gc.collect()
            self._yunet_idx = None
            PERF.event("EnrollWorker model release complete",
                       (time.perf_counter_ns() - release_started_ns) / 1e6,
                       "SFace released; YuNet singleton retained")

    def _process_enroll(self, bgr_frame):
        started_ns = time.perf_counter_ns()
        PERF.event("EnrollWorker extract_feature_for_enroll start")
        try:
            feat, score, reason = extract_feature_for_enroll(
                self._yunet_session, self._yunet_idx, self._sface_session, bgr_frame)
        except Exception as exc:
            _log_worker_exception(
                "EnrollWorker extract_feature_for_enroll failed", exc,
                (time.perf_counter_ns() - started_ns) / 1e6)
            raise
        PERF.event("EnrollWorker extract_feature_for_enroll returned",
                   (time.perf_counter_ns() - started_ns) / 1e6,
                   "success=%s score=%s reason=%s" %
                   (feat is not None, score, reason))
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
            started_ns = time.perf_counter_ns()
            PERF.event("EnrollWorker enroll_person start",
                       detail="name=%s samples=%d" % (name, count))
            try:
                record = enroll_person(name, self._enroll_feats)
            except Exception as exc:
                _log_worker_exception(
                    "EnrollWorker enroll_person failed", exc,
                    (time.perf_counter_ns() - started_ns) / 1e6)
                raise
            PERF.event("EnrollWorker enroll_person returned",
                       (time.perf_counter_ns() - started_ns) / 1e6,
                       "success=True name=%s samples=%d" % (name, count))
            with self._mode_lock:
                self._mode = "match"
                self._enroll_name = ""
                self._enroll_feats = []
            self.enroll_complete.emit(name, record)

    def _process_match(self, bgr_frame):
        feat, score, reason = extract_feature_for_match(
            self._yunet_session, self._yunet_idx, self._sface_session, bgr_frame)
        box = None
        detection_score = float(score or 0.0)
        if feat is not None:
            face = detect_face(self._yunet_session, self._yunet_idx, bgr_frame, MATCH_CONF_TH)
            if face is not None:
                detection_score = float(face[0])
                box = face[1].tolist()
            name, similarity = match_feature(feat)
            text = reason or ("已匹配" if name else "未知")
            self.match_ready.emit(box, name, float(similarity), detection_score, text)
        else:
            self.match_ready.emit(None, None, 0.0, detection_score,
                                  reason or "未检测到人脸")
        if self._face_tracking_enabled.is_set():
            self.face_candidates_ready.emit(
                self._decode_current_face_boxes(bgr_frame), bgr_frame.shape)

    def _decode_current_face_boxes(self, bgr_frame):
        get_outputs = getattr(
            self._yunet_session, "last_inference_outputs", None)
        if get_outputs is None:
            return []
        try:
            return decode_face_boxes(
                get_outputs(), self._yunet_idx, bgr_frame.shape,
                MATCH_CONF_TH)
        except Exception as exc:
            _log_worker_exception(
                "EnrollWorker face candidate decode failed", exc)
            return []

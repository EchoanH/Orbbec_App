"""人脸录入、实时识别与人员库管理页面。"""

import logging
import math
import time
import traceback

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen
from PyQt5.QtWidgets import (QAbstractItemView, QApplication, QFrame,
                             QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                             QMessageBox, QPushButton, QSpinBox, QTableWidget,
                             QTableWidgetItem, QVBoxLayout)

from inference.enroll_worker import EnrollWorker
from inference.face_db_adapter import delete_person, list_enrolled
from inference.face14_infer import draw_face_guide
from perf_logging import get_perf_logger

from .base_page import BasePage


SMOOTHING_TAU_MS = 30.0
NAME_STABLE_FRAMES = 3
LOGGER = logging.getLogger(__name__)
PERF = get_perf_logger()


def _log_page_exception(stage, exc):
    stack = traceback.format_exc()
    message = "%s: %s\n%s" % (stage, exc, stack)
    LOGGER.error(message)
    PERF.event("EnrollPage exception", detail=message, level="WARNING")


class EnrollPage(BasePage):
    page_title = "人脸录入"
    page_hint = "YuNet + SFace · 实时识别与人员库"

    def __init__(self, yunet_session, parent=None):
        super(EnrollPage, self).__init__(parent)
        self._yunet_session = yunet_session
        self._worker = None
        self._active = False
        self._stop_requested = False
        self._restart_when_finished = False
        self._status_text = "人脸模型加载中..."
        self._latest_box = None
        self._display_box = None
        self._smoothing_tick_ns = None
        self._latest_similarity = 0.0
        self._latest_detection_score = 0.0
        self._pending_name = None
        self._pending_count = 0
        self._stable_name = ""
        self._enrolling = False
        self._build_controls()
        self._refresh_people()
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._shutdown_worker)

    def _build_controls(self):
        panel = QFrame()
        panel.setObjectName("formPanel")
        panel.setMaximumHeight(250)
        outer = QHBoxLayout(panel)
        outer.setContentsMargins(18, 16, 18, 16)
        outer.setSpacing(24)

        form = QVBoxLayout()
        form.setSpacing(8)
        title = QLabel("登记新成员")
        title.setObjectName("formTitle")
        form.addWidget(title)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("请输入姓名")
        form.addWidget(self.name_edit)
        count_row = QHBoxLayout()
        count_row.addWidget(QLabel("采集张数"))
        self.sample_spin = QSpinBox()
        self.sample_spin.setRange(3, 5)
        self.sample_spin.setValue(3)
        count_row.addWidget(self.sample_spin)
        count_row.addStretch(1)
        form.addLayout(count_row)
        self.register_button = QPushButton("录入")
        self.register_button.setObjectName("primaryButton")
        self.register_button.clicked.connect(self._start_enrollment)
        form.addWidget(self.register_button)
        self.result_label = QLabel("识别结果：等待人脸\n相似度：--")
        self.result_label.setObjectName("resultLabel")
        self.result_label.setAlignment(Qt.AlignTop)
        self.result_label.setWordWrap(True)
        self.result_label.setMinimumHeight(62)
        form.addWidget(self.result_label)
        outer.addLayout(form, 2)

        people = QVBoxLayout()
        people.setSpacing(8)
        self.count_label = QLabel("已录入人员：0")
        self.count_label.setObjectName("formTitle")
        people.addWidget(self.count_label)
        self.people_table = QTableWidget(0, 3)
        self.people_table.setHorizontalHeaderLabels(["姓名", "样本数", "录入时间"])
        self.people_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.people_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.people_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.people_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.people_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.people_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        people.addWidget(self.people_table, 1)
        self.delete_button = QPushButton("删除选中人员")
        self.delete_button.clicked.connect(self._delete_selected)
        people.addWidget(self.delete_button)
        outer.addLayout(people, 3)
        self.layout().addWidget(panel)

    def process_frame(self, bgr_frame, depth_frame=None):
        if self._worker is not None:
            self._worker.submit_frame(bgr_frame)
        # P0 UI 渲染性能优化：绘制前先降采样，坐标同步缩放。
        height, width = bgr_frame.shape[:2]
        target_width, target_height, scale = self.compute_target_size(
            width, height)
        small_frame = cv2.resize(
            bgr_frame, (target_width, target_height),
            interpolation=cv2.INTER_LINEAR)
        box = self._update_display_box() if self._latest_box is not None else None
        box_scaled = None
        if box is not None:
            box_scaled = box.copy()
            box_scaled *= scale
        label = None
        if box is not None:
            label = self._stable_name or "识别中"
            if self._stable_name:
                label = "%s %.2f · conf %.3f" % (
                    self._stable_name, self._latest_similarity,
                    self._latest_detection_score)
            else:
                label = "识别中 · conf %.3f" % self._latest_detection_score
        return self._draw_match(small_frame, box_scaled, label), self._status_text

    def on_activated(self):
        started_ns = time.perf_counter_ns()
        PERF.event("EnrollPage activate start", detail="active=True")
        self._active = True
        try:
            self._refresh_people()
            if self._worker is None:
                self._start_worker()
            elif self._stop_requested:
                self._restart_when_finished = True
        except Exception as exc:
            _log_page_exception("EnrollPage activate failed", exc)
            raise
        PERF.event("EnrollPage activate complete",
                   (time.perf_counter_ns() - started_ns) / 1e6)

    def on_deactivated(self):
        started_ns = time.perf_counter_ns()
        PERF.event("EnrollPage deactivate start", detail="begin worker stop/release")
        self._active = False
        try:
            self._reset_match()
            self._set_enrolling(False)
            self._restart_when_finished = False
            if self._worker is not None:
                self._worker.cancel_enrollment()
                self._stop_requested = True
                self._worker.stop()
                if self._worker.isRunning():
                    self._worker.wait()
        except Exception as exc:
            _log_page_exception("EnrollPage deactivate failed", exc)
            raise
        PERF.event("EnrollPage deactivate complete",
                   (time.perf_counter_ns() - started_ns) / 1e6)

    def _start_worker(self):
        started_ns = time.perf_counter_ns()
        PERF.event("EnrollPage model loading start",
                   detail="start EnrollWorker; YuNet singleton id=%s + SFace" %
                   id(self._yunet_session))
        self._status_text = "人脸模型加载中..."
        self._stop_requested = False
        try:
            worker = EnrollWorker(self._yunet_session, self)
            worker.status_ready.connect(self._on_status)
            worker.match_ready.connect(self._on_match)
            worker.enroll_progress.connect(self._on_enroll_progress)
            worker.enroll_complete.connect(self._on_enroll_complete)
            worker.error_occurred.connect(self._on_error)
            worker.finished.connect(self._on_finished)
            self._worker = worker
            worker.start()
        except Exception as exc:
            _log_page_exception("EnrollPage model loading start failed", exc)
            raise
        PERF.event("EnrollPage model loading dispatched",
                   (time.perf_counter_ns() - started_ns) / 1e6)

    def _start_enrollment(self):
        clicked_ns = time.perf_counter_ns()
        PERF.event("EnrollPage enroll button clicked", detail="before validation")
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "无法录入", "请输入姓名。")
            return
        try:
            existing = {row[0] for row in list_enrolled()}
        except Exception as exc:
            _log_page_exception("EnrollPage enroll validation failed", exc)
            raise
        if name in existing:
            answer = QMessageBox.question(
                self, "确认覆盖", "该姓名已录入，是否覆盖？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                return
        if self._worker is None or not self._worker.isRunning():
            QMessageBox.warning(self, "模型未就绪", "人脸模型尚未启动。")
            return
        target = self.sample_spin.value()
        self._set_enrolling(True)
        self.result_label.setText("开始录入 %s：0/%d\n请正对摄像头并保持光线充足" % (name, target))
        PERF.event("EnrollPage start_enrollment dispatch",
                   (time.perf_counter_ns() - clicked_ns) / 1e6,
                   "name=%s target=%d; worker will call extract_feature_for_enroll" %
                   (name, target))
        try:
            self._worker.start_enrollment(name, target)
        except Exception as exc:
            _log_page_exception("EnrollPage start_enrollment failed", exc)
            raise
        PERF.event("EnrollPage start_enrollment dispatched",
                   (time.perf_counter_ns() - clicked_ns) / 1e6)

    def _on_status(self, text):
        PERF.event("EnrollPage worker status", detail=str(text))
        self._status_text = text

    def _on_match(self, box, name, similarity, detection_score, reason):
        if not self._active or self._stop_requested or self._enrolling:
            return
        if box is None:
            self._reset_match()
            self.result_label.setText("识别结果：等待人脸\n相似度：--\n%s" % reason)
            self._status_text = reason
            return
        self._latest_box = np.asarray(box, dtype=np.float32)
        if self._display_box is None:
            self._display_box = self._latest_box.copy()
            self._smoothing_tick_ns = time.perf_counter_ns()
        candidate = name or "未知"
        if candidate == self._pending_name:
            self._pending_count += 1
        else:
            self._pending_name = candidate
            self._pending_count = 1
        if self._pending_count >= NAME_STABLE_FRAMES:
            self._stable_name = candidate
        self._latest_similarity = similarity
        self._latest_detection_score = detection_score
        display_name = self._stable_name or "识别中"
        self.result_label.setText(
            "识别结果：%s\n相似度：%.3f\n检测置信度：%.3f" % (
                display_name, similarity, detection_score))
        self._status_text = "%s · 相似度 %.3f · 检测 %.3f" % (
            display_name, similarity, detection_score)

    def _on_enroll_progress(self, count, target, score, reason):
        PERF.event("EnrollPage enroll progress",
                   detail="count=%d/%d score=%.3f reason=%s" %
                   (count, target, score, reason))
        if not self._enrolling:
            return
        self.result_label.setText(
            "录入进度：%d/%d\n本次置信度：%.3f\n%s" % (count, target, score, reason))
        self._status_text = "录入中 %d/%d · %s" % (count, target, reason)

    def _on_enroll_complete(self, name, record):
        PERF.event("EnrollPage enroll complete",
                   detail="name=%s sample_count=%s" %
                   (name, record.get("sample_count", 0)))
        self._set_enrolling(False)
        self._refresh_people()
        self.result_label.setText(
            "录入成功：%s\n样本数：%d\n录入时间：%s" % (
                name, record.get("sample_count", 0), record.get("enrolled_at", "?")))
        self._status_text = "录入成功：%s" % name
        QMessageBox.information(self, "录入完成", "%s 已成功录入。" % name)

    def _on_error(self, text):
        PERF.event("EnrollPage worker error", detail=str(text), level="WARNING")
        self._reset_match()
        self._set_enrolling(False)
        self._status_text = text
        self.result_label.setText(text)

    def _on_finished(self):
        worker = self.sender()
        PERF.event("EnrollPage worker finished",
                   detail="same_worker=%s" % (worker is self._worker))
        if worker is self._worker:
            self._worker = None
        worker.deleteLater()
        should_restart = self._active and self._restart_when_finished
        self._restart_when_finished = False
        self._stop_requested = False
        if should_restart and self._worker is None:
            self._start_worker()

    def _delete_selected(self):
        row = self.people_table.currentRow()
        if row < 0:
            QMessageBox.warning(self, "无法删除", "请先选择一名已录入人员。")
            return
        name = self.people_table.item(row, 0).text()
        answer = QMessageBox.question(
            self, "确认删除", "确定删除“%s”的人脸记录吗？" % name,
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return
        if delete_person(name):
            self._refresh_people()
            self.result_label.setText("已删除：%s" % name)
        else:
            QMessageBox.warning(self, "删除失败", "该姓名已不存在。")

    def _refresh_people(self):
        started_ns = time.perf_counter_ns()
        try:
            records = list_enrolled()
        except Exception as exc:
            _log_page_exception("EnrollPage list_enrolled failed", exc)
            raise
        self.people_table.setRowCount(len(records))
        for row, (name, sample_count, enrolled_at) in enumerate(records):
            self.people_table.setItem(row, 0, QTableWidgetItem(str(name)))
            self.people_table.setItem(row, 1, QTableWidgetItem(str(sample_count)))
            self.people_table.setItem(row, 2, QTableWidgetItem(str(enrolled_at)))
        self.count_label.setText("已录入人员：%d" % len(records))

        PERF.event("EnrollPage list_enrolled returned",
                   (time.perf_counter_ns() - started_ns) / 1e6,
                   "count=%d" % len(records))

    def _set_enrolling(self, active):
        self._enrolling = active
        self.name_edit.setEnabled(not active)
        self.sample_spin.setEnabled(not active)
        self.register_button.setEnabled(not active)
        self.delete_button.setEnabled(not active)

    def _update_display_box(self):
        target = self._latest_box
        if self._display_box is None:
            self._display_box = target.copy()
            self._smoothing_tick_ns = time.perf_counter_ns()
            return self._display_box
        now_ns = time.perf_counter_ns()
        elapsed_ms = max(0.0, (now_ns - self._smoothing_tick_ns) / 1e6)
        self._smoothing_tick_ns = now_ns
        alpha = 1.0 - math.exp(-elapsed_ms / SMOOTHING_TAU_MS)
        self._display_box = self._display_box + alpha * (target - self._display_box)
        return self._display_box

    def _reset_match(self):
        self._latest_box = None
        self._display_box = None
        self._smoothing_tick_ns = None
        self._latest_similarity = 0.0
        self._latest_detection_score = 0.0
        self._pending_name = None
        self._pending_count = 0
        self._stable_name = ""

    @staticmethod
    def _draw_match(bgr_frame, box, label):
        canvas = draw_face_guide(bgr_frame)
        if box is None or label is None:
            return canvas
        canvas = np.ascontiguousarray(canvas)
        height, width = canvas.shape[:2]
        rgb = canvas[:, :, ::-1].copy()
        image = QImage(rgb.data, width, height, int(rgb.strides[0]),
                       QImage.Format_RGB888).copy()
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor(74, 158, 255), 3))
        x1, y1, x2, y2 = [float(v) for v in box]
        painter.drawRect(QRectF(x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)))
        painter.setFont(QFont("Droid Sans Fallback", 16, QFont.Bold))
        metrics = painter.fontMetrics()
        text_width = metrics.horizontalAdvance(label) + 14
        text_height = metrics.height() + 8
        text_y = max(0.0, y1 - text_height)
        painter.fillRect(QRectF(x1, text_y, text_width, text_height), QColor(74, 158, 255))
        painter.setPen(QColor(8, 19, 31))
        painter.drawText(QRectF(x1 + 7, text_y, text_width - 7, text_height),
                         Qt.AlignVCenter | Qt.AlignLeft, label)
        painter.end()
        rendered = image.convertToFormat(QImage.Format_RGB888)
        ptr = rendered.bits()
        ptr.setsize(rendered.byteCount())
        rows = np.frombuffer(ptr, dtype=np.uint8).reshape(
            height, rendered.bytesPerLine())
        drawn_rgb = rows[:, :width * 3].reshape(height, width, 3)
        return drawn_rgb[:, :, ::-1].copy()

    def _shutdown_worker(self):
        started_ns = time.perf_counter_ns()
        PERF.event("EnrollPage shutdown start")
        self._active = False
        worker = self._worker
        try:
            if worker is not None and worker.isRunning():
                worker.cancel_enrollment()
                worker.stop()
                worker.wait()
            self._worker = None
        except Exception as exc:
            _log_page_exception("EnrollPage shutdown failed", exc)
            raise
        PERF.event("EnrollPage shutdown complete",
                   (time.perf_counter_ns() - started_ns) / 1e6)

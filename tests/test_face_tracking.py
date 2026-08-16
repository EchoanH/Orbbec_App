import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtWidgets import QApplication

from gimbal.controller import GimbalWorker
from gimbal.pid_config import get_default_pid_config, save_pid_config
from inference.enroll_worker import EnrollWorker
from inference.face_tracking import (
    FACE_LOCK_MAX_MISSES,
    FaceTargetLock,
    decode_face_boxes,
    normalized_face_center,
)
from inference.yunet_session import YuNetSession
from ui.pages.enroll_page import EnrollPage
from ui.pages.target_tracking_page import TargetTrackingPage
import ui.pages.enroll_page as enroll_page_module


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SerialOwnership(object):
    def __init__(self):
        self.active = 0
        self.maximum_active = 0
        self.commands = []

    def open(self, **_kwargs):
        return CountingSerial(self)


class CountingSerial(object):
    def __init__(self, ownership):
        self._ownership = ownership
        self._pending = b""
        self._closed = False
        self.pan = 90
        self.tilt = 90
        ownership.active += 1
        ownership.maximum_active = max(
            ownership.maximum_active, ownership.active)

    def reset_input_buffer(self):
        pass

    def write(self, payload):
        command = payload.decode("ascii").strip()
        self._ownership.commands.append(command)
        parts = command.split()
        if parts[:3] == ["GIMBAL", "JOG", "PAN"]:
            self.pan += int(parts[3])
        elif parts[:3] == ["GIMBAL", "JOG", "TILT"]:
            self.tilt += int(parts[3])
        self._pending = (
            "GIMBAL OK pan=%d tilt=%d\n" % (self.pan, self.tilt)
        ).encode("ascii")
        return len(payload)

    def flush(self):
        pass

    def readline(self):
        response = self._pending
        self._pending = b""
        return response

    def close(self):
        if not self._closed:
            self._closed = True
            self._ownership.active -= 1


class RecordingAutoWorker(object):
    def __init__(self):
        self.auto_enabled = False
        self.jogs = []

    def set_auto_enabled(self, enabled):
        self.auto_enabled = bool(enabled)

    def submit_auto_jog(self, pan_delta, tilt_delta):
        self.jogs.append((int(pan_delta), int(tilt_delta)))


class FaceTargetLockTests(unittest.TestCase):
    def test_single_face_normalized_center(self):
        normalized = normalized_face_center(
            [50.0, 25.0, 100.0, 75.0], (100, 200, 3))
        self.assertAlmostEqual(normalized[0], -0.25)
        self.assertAlmostEqual(normalized[1], 0.0)

    def test_initial_target_is_largest_valid_face(self):
        target = FaceTargetLock().update([
            [10, 10, 30, 30],
            [80, 20, 150, 90],
            [0, 0, 0, 10],
        ])
        np.testing.assert_allclose(target, [80, 20, 150, 90])

    def test_locked_target_wins_over_new_larger_face(self):
        lock = FaceTargetLock()
        lock.update([[10, 10, 40, 40]])
        target = lock.update([
            [12, 11, 42, 41],
            [80, 5, 180, 95],
        ])
        np.testing.assert_allclose(target, [12, 11, 42, 41])
        self.assertEqual(lock.misses, 0)

    def test_temporary_miss_does_not_switch_to_other_face(self):
        lock = FaceTargetLock()
        lock.update([[10, 10, 40, 40]])
        other = [[150, 10, 190, 50]]
        self.assertIsNone(lock.update(other))
        self.assertIsNotNone(lock.locked_box)
        self.assertIsNone(lock.update(other))
        self.assertIsNotNone(lock.locked_box)
        target = lock.update(other)
        np.testing.assert_allclose(target, other[0])

    def test_sustained_missing_faces_unlocks_target(self):
        lock = FaceTargetLock()
        lock.update([[10, 10, 40, 40]])
        for _ in range(FACE_LOCK_MAX_MISSES):
            self.assertIsNone(lock.update([]))
        self.assertIsNone(lock.locked_box)
        self.assertEqual(lock.misses, 0)


class YuNetOutputReuseTests(unittest.TestCase):
    def test_reading_last_outputs_does_not_run_second_inference(self):
        class FakeSession(object):
            def __init__(self):
                self.calls = 0
                self.outputs = [np.array([1.0], dtype=np.float32)]

            def infer(self, _inputs):
                self.calls += 1
                return self.outputs

        yunet = YuNetSession(model_path="unused")
        fake = FakeSession()
        yunet._session = fake
        returned = yunet.infer([object()])
        snapshot = yunet.last_inference_outputs()
        self.assertEqual(fake.calls, 1)
        self.assertIs(returned[0], snapshot[0])

    def test_cached_output_decodes_all_nms_face_boxes(self):
        boxes = np.array([
            [10, 10, 40, 40],
            [80, 20, 150, 100],
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        keypoints = np.zeros((2, 5, 2), dtype=np.float32)
        with patch(
                "inference.face_tracking.yunet_decode",
                return_value=(boxes, scores, keypoints)):
            decoded = decode_face_boxes(
                [object()], {}, (120, 200, 3), 0.60)
        self.assertEqual(len(decoded), 2)
        np.testing.assert_allclose(decoded[0], boxes[0])
        np.testing.assert_allclose(decoded[1], boxes[1])

    def test_candidate_postprocess_only_runs_when_tracking_enabled(self):
        worker = EnrollWorker(object())
        decode_calls = []
        worker._decode_current_face_boxes = (
            lambda _frame: decode_calls.append(True) or [])
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        with patch(
                "inference.enroll_worker.extract_feature_for_match",
                return_value=(None, None, "未检测到人脸")):
            worker._process_match(frame)
            self.assertEqual(decode_calls, [])
            worker.set_face_tracking_enabled(True)
            worker._process_match(frame)
        self.assertEqual(decode_calls, [True])


class FaceGimbalPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _wait_until(self, condition, timeout=2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.app.processEvents()
            if condition():
                return True
            time.sleep(0.005)
        return False

    @staticmethod
    def _worker_builder(ownership):
        def build(parent=None):
            return GimbalWorker(
                serial_factory=ownership.open, parent=parent)
        return build

    def test_tracking_defaults_off_and_recognition_behavior_is_unchanged(self):
        yunet_singleton = object()
        page = EnrollPage(yunet_singleton)
        self.assertIs(page._yunet_session, yunet_singleton)
        self.assertFalse(page.face_track_button.isChecked())
        self.assertEqual(page.face_track_button.text(), "人脸跟踪：关闭")
        self.assertIsNone(page._gimbal_worker)

        page._start_worker = lambda: None
        page.on_activated()
        self.assertTrue(page._active)
        self.assertIsNone(page._gimbal_worker)
        for _ in range(3):
            page._on_match([10, 20, 70, 90], "张三", 0.82, 0.91, "已匹配")
        np.testing.assert_allclose(page._latest_box, [10, 20, 70, 90])
        self.assertEqual(page._stable_name, "张三")
        self.assertIn("识别结果：张三", page.result_label.text())
        self.assertIsNone(page._gimbal_worker)
        page.on_deactivated()
        page.deleteLater()
        self.app.processEvents()

    def test_lost_stops_auto_jog_and_resets_pid_without_unlocking_early(self):
        page = EnrollPage(object())
        worker = RecordingAutoWorker()
        page._active = True
        page._face_tracking_enabled = True
        page._gimbal_worker = worker
        page._face_target_lock.update([[10, 10, 40, 40]])
        page._pan_pid.integral = 0.4
        page._pan_pid.previous_error = 0.2
        page._pan_pid.accumulator = 0.8
        page._tilt_pid.integral = -0.3
        page._tilt_pid.previous_error = -0.1
        page._tilt_pid.accumulator = -0.7
        page._pending_auto["PAN"] = (page._pan_pid, 1)
        page._last_pid_time = 1.0
        worker.auto_enabled = True

        page._on_face_candidates([], (100, 200, 3))
        self.assertFalse(worker.auto_enabled)
        self.assertIsNotNone(page._face_target_lock.locked_box)
        for axis in (page._pan_pid, page._tilt_pid):
            self.assertEqual(axis.integral, 0.0)
            self.assertIsNone(axis.previous_error)
            self.assertEqual(axis.accumulator, 0.0)
        self.assertEqual(page._pending_auto, {})
        self.assertIsNone(page._last_pid_time)
        self.assertIn("等待原锁定人脸", page.face_tracking_label.text())
        page._gimbal_worker = None
        page.deleteLater()
        self.app.processEvents()

    def test_face_pid_uses_fractional_output_and_fixed_direction(self):
        page = EnrollPage(object())
        worker = RecordingAutoWorker()
        page._gimbal_worker = worker
        page._face_tracking_enabled = True
        page._gimbal_connected = True
        page._pan_angle = 90.0
        page._tilt_angle = 90.0
        page._pid_config.update({
            "pan_kp": 3.5,
            "pan_ki": 0.0,
            "pan_kd": 0.0,
            "tilt_kp": 0.0,
            "tilt_ki": 0.0,
            "tilt_kd": 0.0,
            "deadzone_x": 0.0,
            "deadzone_y": 0.0,
            "control_interval_s": 0.08,
            "max_jog_deg": 2.0,
        })
        for _ in range(3):
            page._last_pid_time = time.monotonic() - 0.1
            page._maybe_control_face((0.1, 0.0))
        self.assertEqual(worker.jogs, [(-1, 0)])
        page._on_gimbal_command_completed(
            "GIMBAL JOG PAN -1", "GIMBAL OK")
        self.assertAlmostEqual(page._pan_pid.accumulator, 0.05, places=5)
        self.assertEqual(enroll_page_module.PAN_SIGN, -1)
        self.assertEqual(enroll_page_module.TILT_SIGN, -1)
        page._gimbal_worker = None
        page.deleteLater()
        self.app.processEvents()

    def test_missing_real_angles_blocks_automatic_jog(self):
        page = EnrollPage(object())
        worker = RecordingAutoWorker()
        page._active = True
        page._face_tracking_enabled = True
        page._gimbal_worker = worker
        page._gimbal_connected = True
        page._gimbal_status = "已连接，等待当前角度"
        page._on_face_candidates(
            [[120, 100, 220, 220]], (360, 640, 3))
        page._last_pid_time = time.monotonic() - 1.0
        page._on_face_candidates(
            [[122, 100, 222, 220]], (360, 640, 3))
        self.assertEqual(worker.jogs, [])
        self.assertFalse(worker.auto_enabled)
        self.assertIn("等待当前角度", page.face_tracking_label.text())
        page._gimbal_worker = None
        page.deleteLater()
        self.app.processEvents()

    def test_toggle_loads_config_and_releases_serial_without_late_jog(self):
        ownership = SerialOwnership()
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "gimbal_pid.json"
            values = get_default_pid_config()
            values["pan_kp"] = 6.25
            save_pid_config(values, config_path)
            page = EnrollPage(
                object(),
                gimbal_worker_factory=self._worker_builder(ownership),
                config_path=config_path)
            page._active = True
            self.assertEqual(ownership.active, 0)
            page.face_track_button.setChecked(True)
            self.assertTrue(self._wait_until(
                lambda: ownership.active == 1 and
                page._pan_angle is not None))
            self.assertAlmostEqual(page._pid_config["pan_kp"], 6.25)
            self.assertIn("PAN 90.0°", page.face_tracking_label.text())

            page.face_track_button.setChecked(False)
            self.assertEqual(ownership.active, 0)
            self.assertIsNone(page._gimbal_worker)
            command_count = len(ownership.commands)
            page._on_face_candidates(
                [[120, 100, 220, 220]], (360, 640, 3))
            time.sleep(0.08)
            self.app.processEvents()
            self.assertEqual(len(ownership.commands), command_count)
            self.assertFalse(page._face_tracking_enabled)
            page.on_deactivated()
            page.deleteLater()
            self.app.processEvents()

    def test_serial_exception_disables_tracking_and_resets_pid(self):
        def build(parent=None):
            def fail_open(**_kwargs):
                raise OSError("测试串口不可用")
            return GimbalWorker(serial_factory=fail_open, parent=parent)

        page = EnrollPage(object(), gimbal_worker_factory=build)
        page._active = True
        page._pan_pid.integral = 0.4
        page._pan_pid.previous_error = 0.2
        page._pan_pid.accumulator = 0.8
        page.face_track_button.setChecked(True)
        self.assertTrue(self._wait_until(
            lambda: not page.face_track_button.isChecked() and
            page._gimbal_worker is None))
        self.assertFalse(page._face_tracking_enabled)
        self.assertEqual(page._pan_pid.integral, 0.0)
        self.assertIsNone(page._pan_pid.previous_error)
        self.assertEqual(page._pan_pid.accumulator, 0.0)
        self.assertIn("云台未连接", page.face_tracking_label.text())
        page.on_deactivated()
        page.deleteLater()
        self.app.processEvents()

    def test_enroll_to_target_page_switch_keeps_single_serial_owner(self):
        ownership = SerialOwnership()
        builder = self._worker_builder(ownership)
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "gimbal_pid.json"
            enroll_page = EnrollPage(
                object(), gimbal_worker_factory=builder,
                config_path=config_path)
            target_page = TargetTrackingPage(
                gimbal_worker_factory=builder, config_path=config_path)
            enroll_page._active = True
            enroll_page.face_track_button.setChecked(True)
            self.assertTrue(self._wait_until(lambda: ownership.active == 1))
            enroll_page.on_deactivated()
            self.assertEqual(ownership.active, 0)
            self.assertIsNone(enroll_page._gimbal_worker)

            target_page.on_activated()
            self.assertTrue(self._wait_until(lambda: ownership.active == 1))
            target_page.on_deactivated()
            self.assertEqual(ownership.active, 0)
            self.assertIsNone(target_page._gimbal_worker)
            self.assertEqual(ownership.maximum_active, 1)
            enroll_page.deleteLater()
            target_page.deleteLater()
            self.app.processEvents()

    def test_tracking_adds_no_yunet_session_or_infer_call(self):
        main_source = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")
        page_source = (PROJECT_ROOT / "ui" / "pages" /
                       "enroll_page.py").read_text(encoding="utf-8")
        tracking_source = (PROJECT_ROOT / "inference" /
                           "face_tracking.py").read_text(encoding="utf-8")
        worker_source = (PROJECT_ROOT / "inference" /
                         "enroll_worker.py").read_text(encoding="utf-8")
        self.assertEqual(main_source.count("YuNetSession()"), 1)
        self.assertNotIn("YuNetSession(", page_source)
        self.assertNotIn("YuNetSession(", tracking_source)
        self.assertNotIn(".infer(", tracking_source)
        self.assertNotIn("CaptureThread", page_source)
        self.assertNotIn("OrbbecSource", page_source)
        self.assertIn("last_inference_outputs", worker_source)


if __name__ == "__main__":
    unittest.main()

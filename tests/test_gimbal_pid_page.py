import ast
import os
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtWidgets import QApplication, QGroupBox, QLabel

from gimbal.controller import GimbalWorker
from inference.target_tracker import TargetTracker
from ui.pages.gimbal_pid_tuner_page import GimbalPIDTunerPage
from ui.pages.target_tracking_page import TargetTrackingPage
import ui.pages.target_tracking_page as target_page_module


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SerialOwnership(object):
    def __init__(self):
        self.active = 0
        self.maximum_active = 0
        self.commands = []

    def open(self, **kwargs):
        return CountingSerial(self, kwargs)


class CountingSerial(object):
    def __init__(self, ownership, kwargs):
        self._ownership = ownership
        self.kwargs = kwargs
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
        elif command == "GIMBAL CENTER":
            self.pan = self.tilt = 90
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


class GimbalPageLifecycleTests(unittest.TestCase):
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

    def test_page_create_track_and_destroy(self):
        ownership = SerialOwnership()
        page = GimbalPIDTunerPage(
            gimbal_worker_factory=self._worker_builder(ownership))
        self.assertIsNone(page.gimbal_worker)
        self.assertFalse(page.auto_enabled)
        page.on_activated()
        self.assertTrue(self._wait_until(lambda: ownership.active == 1))

        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        rng = np.random.default_rng(11)
        frame[120:220, 240:400] = rng.integers(
            0, 256, (100, 160, 3), dtype=np.uint8)
        self.assertTrue(page._tracker.initialize(
            frame, (240, 120, 160, 100)))
        rendered, _status = page.process_frame(frame, frame, None)
        page.show_frame(rendered)
        self.assertEqual(page._tracking_state, TargetTracker.TRACKING)
        self.assertIn("跟踪中", page.target_status.text())

        labels = {label.text() for label in page.findChildren(QLabel)}
        for expected in (
                "水平轴比例系数 Kp", "水平轴积分系数 Ki",
                "水平轴微分系数 Kd", "俯仰轴比例系数 Kp",
                "俯仰轴积分系数 Ki", "俯仰轴微分系数 Kd",
                "水平死区", "垂直死区", "控制周期（秒）",
                "最大单次转角（°）", "积分限幅", "当前误差",
                "比例项 P", "积分项 I", "微分项 D",
                "PID 原始输出", "小数累积量", "实际 JOG 指令"):
            self.assertIn(expected, labels)
        group_titles = {
            group.title() for group in page.findChildren(QGroupBox)}
        self.assertIn("水平轴 PAN PID 参数", group_titles)
        self.assertIn("俯仰轴 TILT PID 参数", group_titles)

        page.on_deactivated()
        self.assertEqual(ownership.active, 0)
        self.assertIsNone(page.gimbal_worker)
        self.assertFalse(page.auto_enabled)
        page.deleteLater()
        self.app.processEvents()

    def test_target_and_pid_pages_never_overlap_serial_ownership(self):
        ownership = SerialOwnership()
        worker_builder = self._worker_builder(ownership)
        original_worker = target_page_module.GimbalWorker
        target_page_module.GimbalWorker = worker_builder
        target_page = TargetTrackingPage()
        pid_page = GimbalPIDTunerPage(
            gimbal_worker_factory=worker_builder)
        try:
            target_page.on_activated()
            self.assertTrue(self._wait_until(lambda: ownership.active == 1))
            target_page.on_deactivated()
            self.assertEqual(ownership.active, 0)

            pid_page.on_activated()
            self.assertTrue(self._wait_until(lambda: ownership.active == 1))
            worker = pid_page.gimbal_worker
            worker.set_auto_enabled(True)
            worker.submit_auto_jog(-2, -2)
            pid_page.on_deactivated()
            command_count = len(ownership.commands)
            time.sleep(0.08)
            self.app.processEvents()
            self.assertEqual(len(ownership.commands), command_count)
            self.assertEqual(ownership.active, 0)
            self.assertEqual(ownership.maximum_active, 1)
            self.assertIsNone(pid_page.gimbal_worker)
        finally:
            target_page_module.GimbalWorker = original_worker
            target_page.on_deactivated()
            pid_page.on_deactivated()
            target_page.deleteLater()
            pid_page.deleteLater()
            self.app.processEvents()

    def test_main_window_is_only_main_gui_capture_owner(self):
        main_path = PROJECT_ROOT / "ui" / "main_window.py"
        page_path = PROJECT_ROOT / "ui" / "pages" / "gimbal_pid_tuner_page.py"
        main_tree = ast.parse(main_path.read_text(encoding="utf-8"))
        calls = [
            node for node in ast.walk(main_tree)
            if isinstance(node, ast.Call) and
            isinstance(node.func, ast.Name) and
            node.func.id == "CaptureThread"
        ]
        self.assertEqual(len(calls), 1)
        page_source = page_path.read_text(encoding="utf-8")
        self.assertNotIn("CaptureThread", page_source)
        self.assertNotIn("OrbbecSource", page_source)


if __name__ == "__main__":
    unittest.main()

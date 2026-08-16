import ast
import os
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtWidgets import QApplication, QGroupBox, QLabel

from gimbal.controller import GimbalWorker
from gimbal.pid_config import get_default_pid_config, save_pid_config
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


class RecordingAutoWorker(object):
    def __init__(self):
        self.auto_enabled = False
        self.jogs = []

    def set_auto_enabled(self, enabled):
        self.auto_enabled = bool(enabled)

    def submit_auto_jog(self, pan_delta, tilt_delta):
        self.jogs.append((int(pan_delta), int(tilt_delta)))


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

    def test_tuner_page_saves_and_restores_current_parameters(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "gimbal_pid.json"
            page = GimbalPIDTunerPage(config_path=config_path)
            page.parameters["pan_kp"].set_value(5.5)
            page.parameters["tilt_kd"].set_value(0.45)
            page.parameters["control_interval"].set_value(0.24)
            page.parameters["max_jog"].set_value(3.0)
            page.save_button.click()
            self.app.processEvents()
            self.assertTrue(config_path.exists())
            self.assertIn("参数保存成功", page.control_status.text())

            page.parameters["pan_kp"].set_value(1.0)
            page.parameters["tilt_kd"].set_value(0.0)
            result = page._load_saved_parameters(announce=True)
            self.assertEqual(result.source, "saved")
            self.assertAlmostEqual(page._value("pan_kp"), 5.5)
            self.assertAlmostEqual(page._value("tilt_kd"), 0.45)
            self.assertAlmostEqual(page._value("control_interval"), 0.24)
            self.assertAlmostEqual(page._value("max_jog"), 3.0)
            page.deleteLater()
            self.app.processEvents()

            reopened_page = GimbalPIDTunerPage(config_path=config_path)
            self.assertAlmostEqual(reopened_page._value("pan_kp"), 5.5)
            self.assertIn(
                "当前参数来源：已保存配置",
                reopened_page.config_source_label.text())
            reopened_page.deleteLater()
            self.app.processEvents()

            config_path.write_text("{broken", encoding="utf-8")
            broken_page = GimbalPIDTunerPage(config_path=config_path)
            self.assertIn(
                "参数加载失败，已使用默认值",
                broken_page.control_status.text())
            self.assertEqual(broken_page._value("pan_kp"), 4.0)
            broken_page.deleteLater()
            self.app.processEvents()

    def test_target_page_reloads_latest_saved_config_on_activation(self):
        ownership = SerialOwnership()
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "gimbal_pid.json"
            first = get_default_pid_config()
            first["pan_kp"] = 5.0
            save_pid_config(first, config_path)
            page = TargetTrackingPage(
                gimbal_worker_factory=self._worker_builder(ownership),
                config_path=config_path)
            self.assertEqual(page._pid_config["pan_kp"], 5.0)

            latest = dict(first)
            latest["pan_kp"] = 6.5
            latest["control_interval_s"] = 0.24
            save_pid_config(latest, config_path)
            page.on_activated()
            self.assertTrue(self._wait_until(lambda: ownership.active == 1))
            self.assertEqual(page._pid_config["pan_kp"], 6.5)
            self.assertEqual(page._pid_config["control_interval_s"], 0.24)
            self.assertEqual(page._config_status, "PID 参数：已保存配置")
            page.on_deactivated()
            self.assertEqual(ownership.active, 0)
            page.deleteLater()
            self.app.processEvents()

    def test_target_page_uses_fractional_pid_and_fixed_direction(self):
        page = TargetTrackingPage(
            config_path=Path(tempfile.gettempdir()) / "missing_pid_config.json")
        worker = RecordingAutoWorker()
        page._gimbal_worker = worker
        page._gimbal_connected = True
        page._pan_angle = 90.0
        page._tilt_angle = 90.0
        page.follow_button.blockSignals(True)
        page.follow_button.setChecked(True)
        page.follow_button.blockSignals(False)
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
            page._maybe_control_gimbal((0.1, 0.0))
        self.assertEqual(worker.jogs, [(-1, 0)])
        self.assertIn("PAN", page._pending_auto)
        page._on_command_completed("GIMBAL JOG PAN -1", "GIMBAL OK")
        self.assertNotIn("PAN", page._pending_auto)
        self.assertAlmostEqual(page._pan_pid.accumulator, 0.05, places=5)
        page._pan_pid.accumulator = 1.5
        limited = page._safe_signed_jog(
            "PAN", 1, target_page_module.PAN_SIGN, 60.0,
            target_page_module.PAN_SAFE_MIN,
            target_page_module.PAN_SAFE_MAX, page._pan_pid)
        self.assertEqual(limited, 0)
        self.assertEqual(page._pan_pid.accumulator, 0.0)
        page._gimbal_worker = None
        page.deleteLater()
        self.app.processEvents()

    def test_target_page_stop_paths_reset_all_pid_state(self):
        page = TargetTrackingPage(
            config_path=Path(tempfile.gettempdir()) / "missing_pid_config.json")

        def prime_pid_state():
            page._pan_pid.integral = 0.4
            page._pan_pid.previous_error = 0.2
            page._pan_pid.accumulator = 0.8
            page._tilt_pid.integral = -0.3
            page._tilt_pid.previous_error = -0.1
            page._tilt_pid.accumulator = -0.7
            page._pending_auto["PAN"] = (page._pan_pid, 1)
            page._last_pid_time = 1.0

        def assert_reset():
            for axis in (page._pan_pid, page._tilt_pid):
                self.assertEqual(axis.integral, 0.0)
                self.assertIsNone(axis.previous_error)
                self.assertEqual(axis.accumulator, 0.0)
            self.assertEqual(page._pending_auto, {})
            self.assertIsNone(page._last_pid_time)

        prime_pid_state()
        page.clear_target()
        assert_reset()

        prime_pid_state()
        page._on_follow_toggled(False)
        assert_reset()

        prime_pid_state()
        page._start_gimbal_worker = lambda: None
        page._center_gimbal()
        assert_reset()

        prime_pid_state()
        page._on_gimbal_error("通信异常：测试")
        assert_reset()

        class LostTracker(object):
            state = TargetTracker.TRACKING

            @staticmethod
            def update(_frame):
                return None

            @staticmethod
            def clear():
                pass

        prime_pid_state()
        page._tracker = LostTracker()
        page._active = True
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        page.process_frame(frame, frame, None)
        assert_reset()

        prime_pid_state()
        page.on_deactivated()
        assert_reset()
        page.deleteLater()
        self.app.processEvents()

    def test_target_page_no_longer_contains_segmented_jog_logic(self):
        source = (PROJECT_ROOT / "ui" / "pages" /
                  "target_tracking_page.py").read_text(encoding="utf-8")
        for old_name in (
                "_jog_for_error", "GIMBAL_SMALL_ERROR",
                "GIMBAL_MEDIUM_ERROR", "GIMBAL_SMALL_JOG_DEG",
                "GIMBAL_MEDIUM_JOG_DEG", "GIMBAL_LARGE_JOG_DEG"):
            self.assertNotIn(old_name, source)
        self.assertIn("PIDAxis", source)
        self.assertIn("safe_jog_for_angle", source)

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

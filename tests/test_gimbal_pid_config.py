import json
import tempfile
import unittest
from pathlib import Path

from gimbal.pid_config import (
    PIDConfigError,
    get_default_pid_config,
    load_pid_config,
    save_pid_config,
    validate_pid_config,
)


class PIDConfigTests(unittest.TestCase):
    def setUp(self):
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.config_path = (
            Path(self._temporary_directory.name) / "config" / "gimbal_pid.json")

    def tearDown(self):
        self._temporary_directory.cleanup()

    def test_missing_file_uses_safe_defaults(self):
        result = load_pid_config(self.config_path)
        self.assertEqual(result.source, "default")
        self.assertEqual(result.values, get_default_pid_config())
        self.assertEqual(result.error, "")

    def test_atomic_save_then_load_preserves_values(self):
        values = get_default_pid_config()
        values.update({
            "pan_kp": 5.25,
            "tilt_kd": 0.4,
            "control_interval_s": 0.22,
            "max_jog_deg": 3.0,
            "integral_limit": 0.5,
        })
        saved_path = save_pid_config(values, self.config_path)
        result = load_pid_config(self.config_path)
        self.assertEqual(saved_path, self.config_path)
        self.assertEqual(result.source, "saved")
        self.assertEqual(result.values, values)
        self.assertEqual(list(self.config_path.parent.glob("*.tmp")), [])

    def test_broken_json_uses_safe_defaults(self):
        self.config_path.parent.mkdir(parents=True)
        self.config_path.write_text("{broken", encoding="utf-8")
        result = load_pid_config(self.config_path)
        self.assertEqual(result.source, "default")
        self.assertEqual(result.values, get_default_pid_config())
        self.assertIn("参数加载失败，已使用默认值", result.error)

    def test_missing_field_uses_safe_defaults(self):
        values = get_default_pid_config()
        del values["tilt_kd"]
        self.config_path.parent.mkdir(parents=True)
        self.config_path.write_text(
            json.dumps(values), encoding="utf-8")
        result = load_pid_config(self.config_path)
        self.assertEqual(result.source, "default")
        self.assertEqual(result.values, get_default_pid_config())
        self.assertIn("配置缺少字段：tilt_kd", result.error)

    def test_invalid_values_are_rejected(self):
        cases = (
            ("pan_kp", 10.1),
            ("pan_ki", "0.1"),
            ("deadzone_x", -0.01),
            ("control_interval_s", 0.01),
            ("max_jog_deg", 2.5),
            ("integral_limit", True),
            ("pan_sign", 1),
            ("tilt_sign", 0),
        )
        for field, value in cases:
            with self.subTest(field=field, value=value):
                values = get_default_pid_config()
                values[field] = value
                with self.assertRaises(PIDConfigError):
                    validate_pid_config(values)


if __name__ == "__main__":
    unittest.main()

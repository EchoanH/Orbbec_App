"""云台 PID 长期参数的共享校验、加载与原子保存。"""

import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "gimbal_pid.json")

_DEFAULT_VALUES = {
    "pan_kp": 4.0,
    "pan_ki": 0.0,
    "pan_kd": 0.25,
    "tilt_kp": 4.0,
    "tilt_ki": 0.0,
    "tilt_kd": 0.25,
    "deadzone_x": 0.13,
    "deadzone_y": 0.13,
    "control_interval_s": 0.18,
    "max_jog_deg": 2.0,
    "integral_limit": 1.0,
    "pan_sign": -1,
    "tilt_sign": -1,
}

_VALUE_RANGES = {
    "pan_kp": (0.0, 10.0),
    "pan_ki": (0.0, 2.0),
    "pan_kd": (0.0, 2.0),
    "tilt_kp": (0.0, 10.0),
    "tilt_ki": (0.0, 2.0),
    "tilt_kd": (0.0, 2.0),
    "deadzone_x": (0.0, 0.30),
    "deadzone_y": (0.0, 0.30),
    "control_interval_s": (0.08, 0.50),
    "max_jog_deg": (1.0, 5.0),
    "integral_limit": (0.0, 5.0),
}


class PIDConfigError(ValueError):
    """PID 配置字段、类型或取值无效。"""


@dataclass(frozen=True)
class PIDConfigLoadResult:
    values: dict
    source: str
    path: Path
    error: str = ""


def get_default_pid_config():
    """返回安全默认参数的独立副本。"""
    return dict(_DEFAULT_VALUES)


def validate_pid_config(values):
    """严格校验并标准化一整套 PID 参数。"""
    if not isinstance(values, Mapping):
        raise PIDConfigError("配置根节点必须是 JSON 对象")

    missing = [key for key in _DEFAULT_VALUES if key not in values]
    if missing:
        raise PIDConfigError("配置缺少字段：%s" % ", ".join(missing))

    normalized = {}
    for key, (minimum, maximum) in _VALUE_RANGES.items():
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PIDConfigError("字段 %s 必须是数字" % key)
        value = float(value)
        if not math.isfinite(value):
            raise PIDConfigError("字段 %s 必须是有限数字" % key)
        if not minimum <= value <= maximum:
            raise PIDConfigError(
                "字段 %s 必须在 %.2f～%.2f 范围内" %
                (key, minimum, maximum))
        if key == "max_jog_deg" and not value.is_integer():
            raise PIDConfigError("字段 max_jog_deg 必须是整数角度")
        normalized[key] = value

    for key in ("pan_sign", "tilt_sign"):
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PIDConfigError("字段 %s 必须是数字 -1" % key)
        if float(value) != -1.0:
            raise PIDConfigError("字段 %s 必须保持为 -1" % key)
        normalized[key] = -1

    return normalized


def load_pid_config(config_path=None):
    """加载配置；缺失或无效时返回安全默认值与中文错误说明。"""
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    defaults = get_default_pid_config()
    try:
        with path.open("r", encoding="utf-8") as stream:
            values = json.load(stream)
        values = validate_pid_config(values)
    except FileNotFoundError:
        return PIDConfigLoadResult(defaults, "default", path)
    except (OSError, UnicodeError, json.JSONDecodeError,
            PIDConfigError) as exc:
        return PIDConfigLoadResult(
            defaults, "default", path,
            "参数加载失败，已使用默认值：%s" % exc)
    return PIDConfigLoadResult(values, "saved", path)


def save_pid_config(values, config_path=None):
    """校验后在目标目录内原子替换配置文件。"""
    normalized = validate_pid_config(values)
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    temporary_path = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n",
                dir=str(path.parent), prefix=".gimbal_pid_", suffix=".tmp",
                delete=False) as stream:
            temporary_path = Path(stream.name)
            json.dump(normalized, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary_path), str(path))
    except (OSError, PIDConfigError) as exc:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        if isinstance(exc, PIDConfigError):
            raise
        raise PIDConfigError("无法写入配置文件：%s" % exc) from exc
    return path

"""PID 云台调试器使用的纯控制算法，不依赖 Qt 或串口。"""

import math
import re
from dataclasses import dataclass


_ANGLE_RE = re.compile(
    r"\b(pan|tilt)\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PIDDiagnostics:
    error: float = 0.0
    p_term: float = 0.0
    i_term: float = 0.0
    d_term: float = 0.0
    raw_output: float = 0.0
    output: float = 0.0
    accumulator: float = 0.0
    jog: int = 0
    in_deadzone: bool = False


class PIDAxis(object):
    """单轴 PID、抗积分饱和和整数 JOG 小数累计器。"""

    def __init__(self):
        self.integral = 0.0
        self.previous_error = None
        self.accumulator = 0.0

    def reset(self):
        self.integral = 0.0
        self.previous_error = None
        self.accumulator = 0.0

    def clear_accumulator(self):
        self.accumulator = 0.0

    def update(self, error, dt, kp, ki, kd, deadzone,
               integral_limit, output_limit, max_jog_deg):
        error = float(error)
        dt = float(dt)
        integral_limit = max(0.0, float(integral_limit))
        output_limit = max(0.0, float(output_limit))
        max_jog = max(0, int(math.floor(float(max_jog_deg))))
        if not math.isfinite(error) or not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("error 必须有限且 dt 必须大于 0")

        if abs(error) <= float(deadzone):
            self.reset()
            return PIDDiagnostics(error=error, in_deadzone=True)

        derivative = 0.0
        if self.previous_error is not None:
            derivative = (error - self.previous_error) / dt

        candidate_integral = _clamp(
            self.integral + error * dt, -integral_limit, integral_limit)
        p_term = float(kp) * error
        i_term = float(ki) * candidate_integral
        d_term = float(kd) * derivative
        raw_output = p_term + i_term + d_term
        output = _clamp(raw_output, -output_limit, output_limit)

        # 条件积分：输出已饱和且误差仍在推动饱和时，不继续累积积分。
        if (float(ki) != 0.0 and raw_output != output and
                error * raw_output > 0.0):
            candidate_integral = self.integral
            i_term = float(ki) * candidate_integral
            raw_output = p_term + i_term + d_term
            output = _clamp(raw_output, -output_limit, output_limit)

        self.integral = candidate_integral
        self.previous_error = error
        self.accumulator += output

        jog = 0
        if max_jog > 0 and abs(self.accumulator) >= 1.0:
            jog = int(math.trunc(self.accumulator))
            jog = int(_clamp(jog, -max_jog, max_jog))

        return PIDDiagnostics(
            error=error,
            p_term=p_term,
            i_term=i_term,
            d_term=d_term,
            raw_output=raw_output,
            output=output,
            accumulator=self.accumulator,
            jog=jog,
            in_deadzone=False,
        )

    def consume_jog(self, jog):
        """仅扣除实际已提交发送的整数部分。"""
        jog = int(jog)
        if jog:
            self.accumulator -= jog


def safe_jog_for_angle(current_angle, jog, safe_min, safe_max):
    """将 JOG 限制在不会越过软件安全角度范围的整数值内。"""
    if current_angle is None:
        return 0
    current = float(current_angle)
    jog = int(jog)
    safe_min = float(safe_min)
    safe_max = float(safe_max)
    if not math.isfinite(current) or safe_min > safe_max:
        return 0
    if jog > 0:
        allowance = int(math.floor(max(0.0, safe_max - current)))
        return min(jog, allowance)
    if jog < 0:
        allowance = int(math.floor(max(0.0, current - safe_min)))
        return max(jog, -allowance)
    return 0


def parse_gimbal_angles(response):
    """从 STM32 回复中解析角度；任一轴缺失时返回 None。"""
    values = {}
    for name, value in _ANGLE_RE.findall(str(response)):
        values[name.lower()] = float(value)
    if "pan" not in values or "tilt" not in values:
        return None
    return values["pan"], values["tilt"]


def _clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))

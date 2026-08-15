import unittest

from gimbal.pid import PIDAxis, parse_gimbal_angles, safe_jog_for_angle


class PIDAxisTests(unittest.TestCase):
    def test_pid_math_and_first_derivative_is_zero(self):
        axis = PIDAxis()
        first = axis.update(0.5, 0.2, 4.0, 0.5, 0.25,
                            0.1, 2.0, 10.0, 5.0)
        self.assertAlmostEqual(first.p_term, 2.0)
        self.assertAlmostEqual(first.i_term, 0.05)
        self.assertEqual(first.d_term, 0.0)
        second = axis.update(0.3, 0.1, 4.0, 0.5, 0.25,
                             0.1, 2.0, 10.0, 5.0)
        self.assertAlmostEqual(second.d_term, -0.5)

    def test_integral_clamp_and_conditional_anti_windup(self):
        axis = PIDAxis()
        for _ in range(20):
            axis.update(1.0, 1.0, 0.0, 1.0, 0.0,
                        0.0, 0.5, 10.0, 5.0)
        self.assertAlmostEqual(axis.integral, 0.5)

        saturated = PIDAxis()
        for _ in range(5):
            sample = saturated.update(1.0, 1.0, 10.0, 1.0, 0.0,
                                      0.0, 5.0, 2.0, 2.0)
        self.assertEqual(sample.output, 2.0)
        self.assertEqual(saturated.integral, 0.0)

    def test_deadzone_clears_all_state(self):
        axis = PIDAxis()
        axis.update(0.5, 0.2, 1.0, 1.0, 0.0, 0.1, 2.0, 5.0, 5.0)
        sample = axis.update(0.05, 0.2, 1.0, 1.0, 0.0,
                             0.1, 2.0, 5.0, 5.0)
        self.assertTrue(sample.in_deadzone)
        self.assertEqual(axis.integral, 0.0)
        self.assertIsNone(axis.previous_error)
        self.assertEqual(axis.accumulator, 0.0)

    def test_fractional_accumulator_preserves_remainder(self):
        axis = PIDAxis()
        first = axis.update(0.1, 0.1, 3.5, 0.0, 0.0,
                            0.0, 2.0, 5.0, 2.0)
        second = axis.update(0.1, 0.1, 3.5, 0.0, 0.0,
                             0.0, 2.0, 5.0, 2.0)
        third = axis.update(0.1, 0.1, 3.5, 0.0, 0.0,
                            0.0, 2.0, 5.0, 2.0)
        self.assertEqual(first.jog, 0)
        self.assertEqual(second.jog, 0)
        self.assertEqual(third.jog, 1)
        axis.consume_jog(third.jog)
        self.assertAlmostEqual(axis.accumulator, 0.05)

    def test_output_and_jog_are_clamped(self):
        axis = PIDAxis()
        sample = axis.update(1.0, 0.1, 20.0, 0.0, 0.0,
                             0.0, 2.0, 2.0, 2.0)
        self.assertEqual(sample.output, 2.0)
        self.assertEqual(sample.jog, 2)

    def test_reset(self):
        axis = PIDAxis()
        axis.update(-0.8, 0.2, 1.0, 1.0, 0.0,
                    0.0, 2.0, 5.0, 5.0)
        axis.reset()
        self.assertEqual(axis.integral, 0.0)
        self.assertIsNone(axis.previous_error)
        self.assertEqual(axis.accumulator, 0.0)


class SafetyTests(unittest.TestCase):
    def test_safe_angle_limit(self):
        self.assertEqual(safe_jog_for_angle(119.0, 2, 60, 120), 1)
        self.assertEqual(safe_jog_for_angle(120.0, 2, 60, 120), 0)
        self.assertEqual(safe_jog_for_angle(60.4, -2, 60, 120), 0)
        self.assertEqual(safe_jog_for_angle(61.0, -2, 60, 120), -1)
        self.assertEqual(safe_jog_for_angle(None, 2, 60, 120), 0)
        self.assertEqual(safe_jog_for_angle(120.0, -2, 60, 120), -2)

    def test_parse_angles(self):
        self.assertEqual(
            parse_gimbal_angles("GIMBAL OK pan=91.5 tilt=88"),
            (91.5, 88.0),
        )
        self.assertEqual(
            parse_gimbal_angles("GIMBAL tilt = -2.5 pan = +10"),
            (10.0, -2.5),
        )
        self.assertIsNone(parse_gimbal_angles("GIMBAL OK"))


if __name__ == "__main__":
    unittest.main()

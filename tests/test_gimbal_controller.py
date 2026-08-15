import os
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from gimbal.controller import GimbalWorker


class FakeSerial(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.commands = []
        self._pending = b""
        self.closed = False
        self.pan = 90
        self.tilt = 90

    def reset_input_buffer(self):
        pass

    def write(self, payload):
        command = payload.decode("ascii").strip()
        self.commands.append(command)
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
        self.closed = True


class GimbalWorkerFakeSerialTests(unittest.TestCase):
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

    def test_single_owner_request_reply_and_auto_stop(self):
        serial_port = FakeSerial()
        factory_calls = []
        completed = []

        def factory(**kwargs):
            factory_calls.append(kwargs)
            return serial_port

        worker = GimbalWorker(serial_factory=factory)
        worker.command_completed.connect(
            lambda command, response: completed.append((command, response)))
        worker.start()
        self.assertTrue(self._wait_until(lambda: len(completed) >= 1))
        self.assertEqual(completed[0][0], "GIMBAL GET")
        self.assertEqual(factory_calls[0]["port"], "/dev/ttyUSB0")
        self.assertEqual(factory_calls[0]["baudrate"], 115200)
        self.assertEqual(factory_calls[0]["bytesize"], 8)
        self.assertEqual(factory_calls[0]["parity"], "N")
        self.assertEqual(factory_calls[0]["stopbits"], 1)

        worker.set_auto_enabled(True)
        worker.submit_auto_jog(-2, 1)
        self.assertTrue(self._wait_until(lambda: len(completed) >= 3))
        self.assertEqual(
            [item[0] for item in completed[1:3]],
            ["GIMBAL JOG PAN -2", "GIMBAL JOG TILT 1"],
        )

        worker.set_auto_enabled(False)
        command_count = len(serial_port.commands)
        worker.submit_auto_jog(5, 5)
        time.sleep(0.08)
        self.app.processEvents()
        self.assertEqual(len(serial_port.commands), command_count)

        worker.stop()
        self.assertTrue(worker.wait(2000))
        self.assertTrue(serial_port.closed)


if __name__ == "__main__":
    unittest.main()

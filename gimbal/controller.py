"""单串口所有者的异步云台命令 worker。"""

import threading
import time
from queue import Empty, Full, Queue

from PyQt5.QtCore import QThread, pyqtSignal


DEFAULT_PORT = "/dev/ttyUSB0"
BAUDRATE = 115200
SERIAL_TIMEOUT_S = 0.5
MIN_COMMAND_INTERVAL_S = 0.05
MANUAL_QUEUE_SIZE = 8


class GimbalProtocolError(RuntimeError):
    pass


class GimbalTimeoutError(RuntimeError):
    pass


class _StopRequested(RuntimeError):
    pass


class GimbalWorker(QThread):
    """唯一串口所有者；所有写入和 readline 都发生在本线程。"""

    status_changed = pyqtSignal(str)
    connection_changed = pyqtSignal(bool)
    response_received = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, port=DEFAULT_PORT, serial_factory=None, parent=None):
        super(GimbalWorker, self).__init__(parent)
        self._port = port
        self._serial_factory = serial_factory
        self._manual_queue = Queue(maxsize=MANUAL_QUEUE_SIZE)
        self._auto_lock = threading.Lock()
        self._auto_enabled = False
        self._latest_auto_jog = None
        self._auto_generation = 0
        self._stop_event = threading.Event()
        self._last_send_time = 0.0

    def request_get(self):
        self._enqueue_manual("GIMBAL GET")

    def request_center(self):
        self.clear_auto()
        self._enqueue_manual("GIMBAL CENTER")

    def set_auto_enabled(self, enabled):
        with self._auto_lock:
            self._auto_enabled = bool(enabled)
            if not self._auto_enabled:
                self._auto_generation += 1
                self._latest_auto_jog = None

    def submit_auto_jog(self, pan_delta, tilt_delta):
        """覆盖旧自动控制，只保留最新目标误差对应的一组 JOG。"""
        pan_delta = int(pan_delta)
        tilt_delta = int(tilt_delta)
        if pan_delta == 0 and tilt_delta == 0:
            return
        with self._auto_lock:
            if self._auto_enabled:
                self._auto_generation += 1
                self._latest_auto_jog = (
                    self._auto_generation, pan_delta, tilt_delta)

    def clear_auto(self):
        with self._auto_lock:
            self._auto_generation += 1
            self._latest_auto_jog = None

    def _is_auto_enabled(self):
        with self._auto_lock:
            return self._auto_enabled

    def _is_current_auto(self, generation):
        with self._auto_lock:
            return (self._auto_enabled and
                    generation == self._auto_generation)

    def stop(self):
        self.set_auto_enabled(False)
        self._stop_event.set()

    def _enqueue_manual(self, command):
        try:
            self._manual_queue.put_nowait(command)
        except Full:
            self.error_occurred.emit("通信异常：云台命令队列已满")

    def _take_next(self):
        try:
            return "manual", self._manual_queue.get_nowait()
        except Empty:
            pass
        with self._auto_lock:
            if not self._auto_enabled or self._latest_auto_jog is None:
                return None, None
            jog = self._latest_auto_jog
            self._latest_auto_jog = None
            return "auto", jog

    def _disable_auto_after_error(self, message):
        self.set_auto_enabled(False)
        self.error_occurred.emit(message)
        self.status_changed.emit(message)

    def _open_serial(self):
        factory = self._serial_factory
        if factory is None:
            import serial
            factory = serial.Serial
        return factory(
            port=self._port, baudrate=BAUDRATE, bytesize=8, parity="N",
            stopbits=1, timeout=SERIAL_TIMEOUT_S,
            write_timeout=SERIAL_TIMEOUT_S, xonxoff=False, rtscts=False,
            dsrdtr=False)

    def _wait_for_send_slot(self):
        remaining = (self._last_send_time + MIN_COMMAND_INTERVAL_S -
                     time.monotonic())
        if remaining > 0.0 and self._stop_event.wait(remaining):
            raise _StopRequested()
        if self._stop_event.is_set():
            raise _StopRequested()

    def _send_and_wait(self, serial_port, command):
        self._wait_for_send_slot()
        serial_port.write((command + "\n").encode("ascii"))
        serial_port.flush()
        self._last_send_time = time.monotonic()
        response_bytes = serial_port.readline()
        if not response_bytes:
            raise GimbalTimeoutError("等待云台回复超时")
        response = response_bytes.decode("ascii", errors="replace").strip()
        if response.startswith("ERR"):
            raise GimbalProtocolError(response)
        if not response.startswith("GIMBAL"):
            raise GimbalProtocolError("未知回复：%s" % response)
        self.response_received.emit(response)
        self.status_changed.emit("云台已连接")
        return response

    def _send_auto_pair(self, serial_port, jog):
        generation, pan_delta, tilt_delta = jog
        if pan_delta and self._is_current_auto(generation):
            self._send_and_wait(
                serial_port, "GIMBAL JOG PAN %d" % pan_delta)
        if tilt_delta and self._is_current_auto(generation):
            self._send_and_wait(
                serial_port, "GIMBAL JOG TILT %d" % tilt_delta)

    def run(self):
        self._stop_event.clear()
        self._last_send_time = 0.0
        serial_port = None
        try:
            serial_port = self._open_serial()
        except Exception as exc:
            message = "云台未连接：%s" % exc
            self.status_changed.emit(message)
            self.error_occurred.emit(message)
            self.connection_changed.emit(False)
            return

        try:
            if hasattr(serial_port, "reset_input_buffer"):
                serial_port.reset_input_buffer()
            self.connection_changed.emit(True)
            self.status_changed.emit("云台已连接")
            try:
                self._send_and_wait(serial_port, "GIMBAL GET")
            except GimbalProtocolError as exc:
                self._disable_auto_after_error("通信异常：%s" % exc)
            except GimbalTimeoutError as exc:
                self._disable_auto_after_error("通信异常：%s" % exc)
                return

            while not self._stop_event.is_set():
                command_type, payload = self._take_next()
                if payload is None:
                    self._stop_event.wait(0.02)
                    continue
                try:
                    if command_type == "manual":
                        self._send_and_wait(serial_port, payload)
                    else:
                        self._send_auto_pair(serial_port, payload)
                except GimbalProtocolError as exc:
                    self._disable_auto_after_error("通信异常：%s" % exc)
                except GimbalTimeoutError as exc:
                    self._disable_auto_after_error("通信异常：%s" % exc)
                    return
                except _StopRequested:
                    return
                except Exception as exc:
                    self._disable_auto_after_error("通信异常：%s" % exc)
                    return
        except _StopRequested:
            pass
        except Exception as exc:
            self._disable_auto_after_error("通信异常：%s" % exc)
        finally:
            self.set_auto_enabled(False)
            if serial_port is not None:
                try:
                    serial_port.close()
                except Exception:
                    pass
            self.connection_changed.emit(False)

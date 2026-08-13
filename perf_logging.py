"""异步性能日志：记录时间戳与统计，不让文件 I/O 阻塞业务线程。"""

import atexit
import datetime
import os
import threading
import time
from collections import defaultdict
from queue import Empty, Full, Queue

#VERBOSE = False

class _PerfStats(object):
    """跨线程共享的轻量计数器与仪表值。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._counters = defaultdict(int)
        self._gauges = {}
        self.started_ns = time.perf_counter_ns()

    def increment(self, name, amount=1):
        with self._lock:
            self._counters[name] += amount

    def set_gauge(self, name, value):
        with self._lock:
            self._gauges[name] = value

    def snapshot(self):
        with self._lock:
            return dict(self._counters), dict(self._gauges)


class PerfLogger(object):
    """每次进程启动创建一个新日志文件，并异步写入控制台和磁盘。"""

    def __init__(self):
        self.stats = _PerfStats()
        self.display = os.environ.get("DISPLAY", "<未设置>")
        root = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(root, "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        self.path = os.path.join(log_dir, "perf_%s.log" % stamp)
        try:
            self._file = open(self.path, "w", encoding="utf-8", buffering=8192)
        except Exception:
            self._file = None
        self._queue = Queue(maxsize=20000)
        self._stop = threading.Event()
        self._closed = False
        self._writer = threading.Thread(
            target=self._writer_loop, name="PerfLogWriter", daemon=True)
        self._writer.start()
        atexit.register(self.close)

    def event(self, name, elapsed_ms=0.0, detail="", level="INFO"):
        """非阻塞记录事件；队列满时丢弃日志而不影响业务线程。"""
        if self._closed:
            return
        item = (time.time(), threading.current_thread().name, name,
                float(elapsed_ms or 0.0), detail, level)
        try:
            self._queue.put_nowait(item)
        except Full:
            self.stats.increment("perf_log_dropped")

    def increment(self, name, amount=1):
        self.stats.increment(name, amount)

    def set_gauge(self, name, value):
        self.stats.set_gauge(name, value)

    def _format(self, item):
        stamp, thread_name, name, elapsed_ms, detail, level = item
        clock = datetime.datetime.fromtimestamp(stamp).strftime("%H:%M:%S.%f")[:-3]
        prefix = "WARNING " if level == "WARNING" else ""
        line = "[%s] [%s] %s%s 耗时=%.1fms" % (
            clock, thread_name, prefix, name, elapsed_ms)
        if detail:
            line += " " + str(detail)
        return line

    def _write_line(self, line):
        if self._file is not None:
            try:
                self._file.write(line + "\n")
            except Exception:
                self._file = None
        try:
            print(line, flush=True)
        except Exception:
            pass

    def _summary_line(self, title):
        counters, gauges = self.stats.snapshot()
        elapsed_s = max((time.perf_counter_ns() - self.stats.started_ns) / 1e9, 1e-6)
        capture_fps = gauges.get("capture_fps", counters.get("capture_frames", 0) / elapsed_s)
        infer_fps = gauges.get("inference_fps", counters.get("inference_frames", 0) / elapsed_s)
        ui_fps = gauges.get("ui_fps", counters.get("ui_frames", 0) / elapsed_s)
        face14_queue_depth = gauges.get("face14_queue_depth", 0)
        pipeline_queue_depth = gauges.get("pipeline_queue_depth", 0)
        queue_depth = face14_queue_depth + pipeline_queue_depth
        face14_dropped = counters.get("face14_queue_dropped", 0)
        pipeline_dropped = counters.get("pipeline_queue_dropped", 0)
        detail = (
            "DISPLAY=%s 采集FPS=%.1f 推理FPS=%.1f UI渲染FPS=%.1f "
            "队列积压=%d(face14=%d,通用=%d) 主动跳帧=%d(face14=%d,通用=%d) "
            "异常丢帧=0 日志丢弃=%d" % (
                self.display, capture_fps, infer_fps, ui_fps, queue_depth,
                face14_queue_depth, pipeline_queue_depth,
                face14_dropped + pipeline_dropped, face14_dropped, pipeline_dropped,
                counters.get("perf_log_dropped", 0)))
        return self._format((time.time(), threading.current_thread().name,
                             title, 0.0, detail, "INFO"))

    def _writer_loop(self):
        next_summary = time.monotonic() + 5.0
        while not self._stop.is_set() or not self._queue.empty():
            timeout = max(0.0, next_summary - time.monotonic())
            try:
                item = self._queue.get(timeout=timeout)
                if item is None:
                    continue
                self._write_line(self._format(item))
            except Empty:
                self._write_line(self._summary_line("五秒汇总"))
                self._flush_file()
                next_summary = time.monotonic() + 5.0
                continue
            if time.monotonic() >= next_summary:
                self._write_line(self._summary_line("五秒汇总"))
                self._flush_file()
                next_summary = time.monotonic() + 5.0

    def _flush_file(self):
        if self._file is not None:
            try:
                self._file.flush()
            except Exception:
                self._file = None

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put_nowait(None)
        except Full:
            pass
        self._stop.set()
        if self._writer.is_alive() and threading.current_thread() is not self._writer:
            self._writer.join(timeout=5.0)
        self._write_line(self._summary_line("程序退出摘要"))
        self._flush_file()
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass


_LOGGER = None
_LOGGER_LOCK = threading.Lock()


def get_perf_logger():
    """获取进程级性能日志器。"""
    global _LOGGER
    if _LOGGER is None:
        with _LOGGER_LOCK:
            if _LOGGER is None:
                _LOGGER = PerfLogger()
                _LOGGER.event("性能日志启动", detail="DISPLAY=%s 文件=%s" % (
                    _LOGGER.display, _LOGGER.path))
    return _LOGGER

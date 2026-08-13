"""Orbbec Gemini 2 取流封装：严格持有 SDK 对象并固定已验证 profile。"""

import logging
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from perf_logging import get_perf_logger

try:
    from pyorbbecsdk import (Config, Context, OBAlignMode, OBFormat,
                             OBSensorType, Pipeline)
except ImportError:
    Config = Context = Pipeline = None
    OBAlignMode = OBFormat = OBSensorType = None


LOGGER = logging.getLogger(__name__)
PERF = get_perf_logger()

# read() 的第三种返回值：帧取到了但按节流策略主动跳过解码。
# 与 None（取帧失败/超时）区分开，避免被记成异常丢帧。
SKIPPED = "SKIPPED"


class OrbbecSource(object):
    """管理设备、profile、pipeline 的完整生命周期。"""

    def __init__(self):
        self._ctx = None
        self._device_list = None
        self._device = None
        self._pipeline = None
        self._config = None
        self._started = False

    @property
    def available(self):
        return Context is not None

    def _pick_profile(self, sensor_type, fmt, width, height, fps=30):
        profiles = self._pipeline.get_stream_profile_list(sensor_type)
        for index in range(profiles.get_count()):
            profile = profiles.get_stream_profile_by_index(index)
            if (profile.get_format() == fmt and profile.get_fps() == fps
                    and profile.get_width() == width and profile.get_height() == height):
                return profile
        raise RuntimeError("未找到匹配的 Orbbec profile: %sx%s@%s" % (width, height, fps))

    def start(self):
        """初始化并启动彩色 + 深度流；任何失败都由上层降级为占位界面。"""
        if not self.available:
            raise RuntimeError("未找到 pyorbbecsdk，当前环境进入无摄像头模式")
        self._ctx = Context()
        self._device_list = self._ctx.query_devices()
        if self._device_list.get_count() <= 0:
            raise RuntimeError("未检测到 Orbbec 摄像头")
        self._device = self._device_list.get_device_by_index(0)
        self._pipeline = Pipeline(self._device)
        color_profile = self._pick_profile(
            OBSensorType.COLOR_SENSOR, OBFormat.MJPG, 1280, 720)
        depth_profile = self._pick_profile(
            OBSensorType.DEPTH_SENSOR, OBFormat.Y16, 1280, 800)
        self._config = Config()
        self._config.enable_stream(color_profile)
        self._config.enable_stream(depth_profile)
        self._config.set_align_mode(OBAlignMode.HW_MODE)
        self._pipeline.start(self._config)
        self._started = True
        LOGGER.info("Orbbec 取流已启动：彩色 1280x720 MJPG，深度 1280x800 Y16")

    def read(self, timeout_ms=300, skip_decode=False):
        """读取一帧并解码为 BGR 与对齐后的 float32 毫米深度数组。

        skip_decode=True 时仍然取帧（保持 SDK 缓冲区流转），但不做 MJPG 解码
        与深度转换，直接返回 SKIPPED。解码实测 43ms/帧、占 CaptureThread 九成
        开销，而下游只消费得动 15~17fps，多解出来的帧会被主动丢弃。
        """
        if not self._started:
            return None
        started_ns = time.perf_counter_ns()
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms)
        except Exception as exc:
            elapsed_ms = (time.perf_counter_ns() - started_ns) / 1e6
            PERF.event("wait_for_frames失败", elapsed_ms, str(exc), level="WARNING")
            raise
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1e6
        if not frames:
            PERF.event("wait_for_frames超时", elapsed_ms,
                       "timeout_ms=%s" % timeout_ms, level="WARNING")
            return None
        PERF.event("wait_for_frames完成", elapsed_ms,
                   "timeout_ms=%s" % timeout_ms)
        if skip_decode:
            PERF.increment("decode_skipped")
            return SKIPPED
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            PERF.event("采集帧不完整", 0.0, level="WARNING")
            return None
        # 诊断埋点：MJPG 软解码耗时（CaptureThread 的最大单项开销）
        decode_started_ns = time.perf_counter_ns()
        bgr = cv2.imdecode(
            np.frombuffer(color_frame.get_data(), np.uint8), cv2.IMREAD_COLOR)
        PERF.event("诊断MJPG解码耗时",
                   (time.perf_counter_ns() - decode_started_ns) / 1e6)
        # 诊断埋点：深度帧 reshape（不再转 float32）
        # 原先 .astype(np.float32) 会把 1280x800 uint16 整块拷成 4MB float32，
        # 而深度只被 get_distance 用于取单点邻域中值——切片、布尔筛选、median
        # 在 uint16 上行为一致，转换纯属浪费。实测 3.8ms/帧。
        depth_started_ns = time.perf_counter_ns()
        depth_mm = np.frombuffer(depth_frame.get_data(), np.uint16).reshape(
            depth_frame.get_height(), depth_frame.get_width())
        PERF.event("诊断深度帧转换耗时",
                   (time.perf_counter_ns() - depth_started_ns) / 1e6)
        if bgr is None:
            raise RuntimeError("彩色帧 MJPG 解码失败")
        return bgr, depth_mm

    def stop(self):
        """停止 pipeline，释放引用；重复调用保持幂等。"""
        if self._pipeline is not None and self._started:
            try:
                self._pipeline.stop()
            except Exception:
                LOGGER.exception("停止 Orbbec pipeline 失败")
        self._started = False
        self._config = None
        self._pipeline = None
        self._device = None
        self._device_list = None
        self._ctx = None
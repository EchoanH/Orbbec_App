"""进程级 YuNet 会话：Face14 与人脸录入共享同一 NPU 模型实例。"""

import gc
import os
import threading


YUNET_OM = "/root/echo/atc_work/yunet_640.om"
DEVICE_ID = 0


class YuNetSession(object):
    """延迟加载一次，并串行保护 ais_bench 会话调用。"""

    def __init__(self, model_path=YUNET_OM, device_id=DEVICE_ID):
        self.model_path = model_path
        self.device_id = device_id
        self._session = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    def load(self):
        with self._load_lock:
            if self._session is not None:
                return
            if not os.path.isfile(self.model_path):
                raise RuntimeError("YuNet 模型不存在：%s" % self.model_path)
            from ais_bench.infer.interface import InferSession
            self._session = InferSession(self.device_id, self.model_path)

    def infer(self, inputs):
        self.load()
        with self._infer_lock:
            return self._session.infer(inputs)

    def get_outputs(self):
        self.load()
        with self._infer_lock:
            return self._session.get_outputs()

    def release(self):
        with self._load_lock:
            with self._infer_lock:
                session = self._session
                self._session = None
            if session is not None:
                del session
                gc.collect()

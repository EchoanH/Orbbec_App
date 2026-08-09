"""人脸 14 关键点推理：算法函数原样来自已验证的 38_face14_demo.py。"""

import gc
import os
import time
from importlib.util import module_from_spec, spec_from_file_location

import cv2
import numpy as np

from perf_logging import get_perf_logger


YUNET_OM = "/root/echo/atc_work/yunet_640.om"
LANDMARK_OM = "/root/echo/atc_work/landmark111_112.om"
FACE14_MAP_PY = "/root/echo/37_face14_map.py"

YUNET_SIZE = 640
STRIDES = [8, 16, 32]
CONF_TH = 0.60
NMS_TH = 0.30

LM_INPUT_SIZE = 112
HEATMAP_SIZE = 14

PART_COLOR = {
    "眉": (255, 200, 0),
    "眼": (0, 255, 255),
    "鼻": (0, 255, 0),
    "口": (0, 0, 255),
    "脸颊/轮廓": (255, 0, 255),
}

torch = None
InferSession = None
extract_face14 = None
FACE14_BY_PART = None
PERF = get_perf_logger()


# 以下解码、前处理和后处理函数原样照抄自 38_face14_demo.py，不要修改。
def yunet_build_index(sess):
    keys = []
    for p in ("cls", "obj", "bbox", "kps"):
        for s in STRIDES:
            keys.append("%s_%d" % (p, s))
    try:
        names = [d.name for d in sess.get_outputs()]
    except Exception:
        names = []
    idx = {}
    for k in keys:
        for i, n in enumerate(names):
            if n.endswith(k):
                idx[k] = i
                break
    if len(idx) == len(keys):
        return idx, "按名字匹配"
    return {k: i for i, k in enumerate(keys)}, "按位置兜底(名字匹配失败)"


def yunet_decode(outs, idx, W, H):
    sx, sy = W / float(YUNET_SIZE), H / float(YUNET_SIZE)
    B, S, K = [], [], []
    for s in STRIDES:
        cls = np.array(outs[idx["cls_%d" % s]]).astype(np.float32).reshape(-1)
        obj = np.array(outs[idx["obj_%d" % s]]).astype(np.float32).reshape(-1)
        bb = np.array(outs[idx["bbox_%d" % s]]).astype(np.float32).reshape(-1, 4)
        kp = np.array(outs[idx["kps_%d" % s]]).astype(np.float32).reshape(-1, 10)

        cols = YUNET_SIZE // s
        n = bb.shape[0]
        ar = np.arange(n)
        c = (ar % cols).astype(np.float32)
        r = (ar // cols).astype(np.float32)

        score = np.sqrt(np.clip(cls, 0, 1) * np.clip(obj, 0, 1))

        cx = (c + bb[:, 0]) * s
        cy = (r + bb[:, 1]) * s
        w = np.exp(bb[:, 2]) * s
        h = np.exp(bb[:, 3]) * s

        x1 = (cx - w / 2) * sx
        y1 = (cy - h / 2) * sy
        x2 = (cx + w / 2) * sx
        y2 = (cy + h / 2) * sy

        k = np.empty((n, 5, 2), dtype=np.float32)
        for j in range(5):
            k[:, j, 0] = (c + kp[:, 2 * j]) * s * sx
            k[:, j, 1] = (r + kp[:, 2 * j + 1]) * s * sy

        B.append(np.stack([x1, y1, x2, y2], 1))
        S.append(score)
        K.append(k)
    return np.concatenate(B), np.concatenate(S), np.concatenate(K)


def yunet_nms(boxes, scores, th):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= th]
    return keep


def crop_square(img_bgr, face_box):
    height, width = img_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in face_box[:4]]
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    size = int(max(w, h) * 1.0)
    cx = x1 + w // 2
    cy = y1 + h // 2
    x1 = cx - size // 2
    x2 = x1 + size
    y1 = cy - size // 2
    y2 = y1 + size
    dx = max(0, -x1)
    dy = max(0, -y1)
    x1 = max(0, x1)
    y1 = max(0, y1)
    edx = max(0, x2 - width)
    edy = max(0, y2 - height)
    x2 = min(width, x2)
    y2 = min(height, y2)
    cropped_box = [x1, y1, x2, y2]
    cropped = img_bgr[y1:y2, x1:x2]
    if dx > 0 or dy > 0 or edx > 0 or edy > 0:
        cropped = cv2.copyMakeBorder(cropped, dy, edy, dx, edx, cv2.BORDER_CONSTANT, value=0)
    cropped = cv2.resize(cropped, (LM_INPUT_SIZE, LM_INPUT_SIZE))
    return cropped, cropped_box


def lm_preprocess_bgr(cropped_bgr):
    x_in = np.ascontiguousarray(cropped_bgr, dtype=np.float32).reshape(1, LM_INPUT_SIZE, LM_INPUT_SIZE, 3)
    x_in = x_in.transpose(0, 3, 1, 2) / 255.0
    return np.ascontiguousarray(x_in, dtype=np.float32)


def lm_postprocess(softmax_out, cropped_box):
    softmax = torch.from_numpy(softmax_out.reshape(1, -1, HEATMAP_SIZE, HEATMAP_SIZE))
    xx, yy = torch.meshgrid(list(map(torch.arange, [HEATMAP_SIZE, HEATMAP_SIZE])))
    approx_x = softmax.mul(xx.float()).view(1, -1, HEATMAP_SIZE * HEATMAP_SIZE).sum(2).unsqueeze(2)
    approx_y = softmax.mul(yy.float()).view(1, -1, HEATMAP_SIZE * HEATMAP_SIZE).sum(2).unsqueeze(2)
    landmarks = [approx_x * HEATMAP_SIZE * 1.5, approx_y * HEATMAP_SIZE * 1.5]
    landmarks = torch.cat(landmarks, 2).numpy()
    box_size = cropped_box[2] - cropped_box[0]
    landmarks = np.reshape(landmarks, (-1, 2)) * box_size / float(LM_INPUT_SIZE)
    landmarks[:, 0] += cropped_box[0]
    landmarks[:, 1] += cropped_box[1]
    return landmarks[0:105], landmarks[105:]


# 只按允许项调整：idx 预计算，HWC 到 CHW/float32 合并为一次必要拷贝。
def yunet_get_facebox(sess_yunet, idx, img_bgr):
    preprocess_started_ns = time.perf_counter_ns()
    H, W = img_bgr.shape[:2]
    im = cv2.resize(img_bgr, (YUNET_SIZE, YUNET_SIZE), interpolation=cv2.INTER_LINEAR)
    blob = np.ascontiguousarray(
        np.expand_dims(im.transpose(2, 0, 1), 0), dtype=np.float32)
    PERF.event("YuNet前处理完成",
               (time.perf_counter_ns() - preprocess_started_ns) / 1e6)
    infer_started_ns = time.perf_counter_ns()
    outs = sess_yunet.infer([blob])
    PERF.event("YuNet infer完成",
               (time.perf_counter_ns() - infer_started_ns) / 1e6)
    postprocess_started_ns = time.perf_counter_ns()
    boxes, scores, kps = yunet_decode(outs, idx, W, H)
    m = scores > CONF_TH
    if m.sum() == 0:
        PERF.event("YuNet后处理完成",
                   (time.perf_counter_ns() - postprocess_started_ns) / 1e6,
                   "未检测到人脸")
        return None, None
    b, s, k = boxes[m], scores[m], kps[m]
    keep = yunet_nms(b, s, NMS_TH)
    if not keep:
        PERF.event("YuNet后处理完成",
                   (time.perf_counter_ns() - postprocess_started_ns) / 1e6,
                   "NMS后无有效人脸")
        return None, None
    best = keep[0]
    PERF.event("YuNet后处理完成",
               (time.perf_counter_ns() - postprocess_started_ns) / 1e6,
               "检测到人脸")
    return b[best], s[best]


# 只按允许项删除文件写入和 out_path 参数，绘制逻辑保持原样。
def draw_face14(img_bgr, face14):
    canvas = img_bgr.copy()
    for part, names in FACE14_BY_PART.items():
        color = PART_COLOR[part]
        pts_this_part = [face14[n] for n in names]
        if len(pts_this_part) >= 2:
            for i in range(len(pts_this_part) - 1):
                p1 = tuple(int(v) for v in pts_this_part[i])
                p2 = tuple(int(v) for v in pts_this_part[i + 1])
                cv2.line(canvas, p1, p2, color, 1, cv2.LINE_AA)
        for name in names:
            x, y = face14[name]
            cv2.circle(canvas, (int(x), int(y)), 4, color, -1)
            cv2.circle(canvas, (int(x), int(y)), 4, (255, 255, 255), 1)
    return canvas


class Face14Engine(object):
    """持有两段模型会话；算法函数之外只负责加载、串联与释放。"""

    def __init__(self, yunet_session):
        self.sess_yunet = yunet_session
        self.sess_landmark = None
        self.yunet_idx = None
        self.yunet_index_mode = ""

    def load(self):
        """在后台线程导入运行库并加载模型，避免阻塞 GUI。"""
        global FACE14_BY_PART, InferSession, extract_face14, torch

        if not os.path.exists(FACE14_MAP_PY):
            raise RuntimeError("14 点映射文件不存在：%s" % FACE14_MAP_PY)
        if not os.path.exists(YUNET_OM):
            raise RuntimeError("YuNet 模型不存在：%s" % YUNET_OM)
        if not os.path.exists(LANDMARK_OM):
            raise RuntimeError("landmark111 模型不存在：%s" % LANDMARK_OM)

        import torch as torch_module
        from ais_bench.infer.interface import InferSession as InferSessionClass

        torch = torch_module
        InferSession = InferSessionClass

        _spec = spec_from_file_location("face14_map", FACE14_MAP_PY)
        _face14 = module_from_spec(_spec)
        _spec.loader.exec_module(_face14)
        extract_face14 = _face14.extract_face14
        FACE14_BY_PART = _face14.FACE14_BY_PART

        self.sess_yunet.load()
        self.sess_landmark = InferSession(0, LANDMARK_OM)
        self.yunet_idx, self.yunet_index_mode = yunet_build_index(self.sess_yunet)

    def infer(self, img_bgr):
        """执行 YuNet + landmark111 单帧完整链路。"""
        face_box, score = yunet_get_facebox(self.sess_yunet, self.yunet_idx, img_bgr)
        if face_box is None:
            return None, None
        crop_started_ns = time.perf_counter_ns()
        cropped, cropped_box = crop_square(img_bgr, face_box)
        PERF.event("landmark裁剪完成",
                   (time.perf_counter_ns() - crop_started_ns) / 1e6)
        preprocess_started_ns = time.perf_counter_ns()
        x_in = lm_preprocess_bgr(cropped)
        PERF.event("landmark前处理完成",
                   (time.perf_counter_ns() - preprocess_started_ns) / 1e6)
        infer_started_ns = time.perf_counter_ns()
        outs = self.sess_landmark.infer([x_in])
        PERF.event("landmark111 infer完成",
                   (time.perf_counter_ns() - infer_started_ns) / 1e6)
        postprocess_started_ns = time.perf_counter_ns()
        lmks105, _ = lm_postprocess(outs[0], cropped_box)
        PERF.event("landmark后处理完成",
                   (time.perf_counter_ns() - postprocess_started_ns) / 1e6,
                   "lm_postprocess(torch)")
        mapping_started_ns = time.perf_counter_ns()
        face14 = extract_face14(lmks105)
        PERF.event("14点映射完成",
                   (time.perf_counter_ns() - mapping_started_ns) / 1e6)
        return face14, float(score)

    def release(self):
        """当前 ais_bench 无 free_resource，按已验证方式删除会话引用。"""
        if self.sess_landmark is not None:
            del self.sess_landmark
            self.sess_landmark = None
            gc.collect()
        self.yunet_idx = None

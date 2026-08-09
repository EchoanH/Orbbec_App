# -*- coding: utf-8 -*-
"""行人检测：保留 39_pedestrian_detect.py 的已验证推理链路。"""

import gc
import os
import time

import cv2
import numpy as np


OM_PATH = "/root/echo/atc_work/yolov5s_640.om"
NAMES_PATH = "/home/HwHiAiUser/samples/notebooks/01-yolov5/coco_names.txt"
IMGSZ = 640
CONF_TH = 0.25
IOU_TH = 0.45
DEVICE_ID = 0
PERSON_CLASS_ID = 0
DIV255 = False


def load_names():
    if os.path.isfile(NAMES_PATH):
        with open(NAMES_PATH, "r", encoding="utf-8") as f:
            n = [l.strip() for l in f if l.strip()]
        if n:
            return n
    return ["cls_%d" % i for i in range(80)]


def letterbox(img, new_size=640, color=(114, 114, 114)):
    """等比缩放 + 灰边填充，返回 图, 缩放比, (左pad, 上pad)"""
    h, w = img.shape[:2]
    r = min(new_size / h, new_size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dh, dw = new_size - nh, new_size - nw
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2
    out = cv2.copyMakeBorder(resized, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return out, r, (left, top)


def preprocess(bgr, div255):
    """B5 约定：letterbox -> BGR2RGB -> HWC2CHW -> float32；div255 由参数控制"""
    lb, r, pad = letterbox(bgr, IMGSZ)
    x = lb[:, :, ::-1]                      # BGR -> RGB
    x = x.transpose(2, 0, 1)                # HWC -> CHW
    x = np.ascontiguousarray(x, dtype=np.float32)
    if div255:
        x = x / 255.0
    x = np.expand_dims(x, 0)                # (1,3,640,640)
    return np.ascontiguousarray(x, dtype=np.float32), r, pad


def pick_pred(outputs):
    """从多路输出里挑出 (1, N, 85) 那一路；挑不到返回 None"""
    cands = []
    for o in outputs:
        a = np.array(o)
        if a.ndim == 3 and a.shape[-1] >= 6:
            cands.append(a)
    if not cands:
        return None
    cands.sort(key=lambda a: a.shape[1], reverse=True)
    return cands[0].astype(np.float32)


def nms(boxes, scores, iou_th):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_th]
    return keep


def postprocess_all(pred, r, pad, orig_shape, names):
    """
    与 04_ab_infer_check.py 的 postprocess 完全一致，
    返回全部类别检出结果，不在此处过滤 person。
    pred: (1,N,85) -> [(cls_id_int, name, score, [x1,y1,x2,y2]), ...]
    """
    p = pred[0]
    obj = p[:, 4]
    m = obj > CONF_TH
    if m.sum() == 0:
        return []
    c = p[m]
    cls_scores = c[:, 5:] * c[:, 4:5]
    cls_id = cls_scores.argmax(1)
    score = cls_scores.max(1)
    m2 = score > CONF_TH
    if m2.sum() == 0:
        return []
    c, cls_id, score = c[m2], cls_id[m2], score[m2]

    cx, cy, bw, bh = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
    x1 = (cx - bw / 2 - pad[0]) / r
    y1 = (cy - bh / 2 - pad[1]) / r
    x2 = (cx + bw / 2 - pad[0]) / r
    y2 = (cy + bh / 2 - pad[1]) / r
    H, W = orig_shape[:2]
    x1 = np.clip(x1, 0, W); x2 = np.clip(x2, 0, W)
    y1 = np.clip(y1, 0, H); y2 = np.clip(y2, 0, H)
    boxes = np.stack([x1, y1, x2, y2], 1)

    res = []
    for cid in np.unique(cls_id):
        idx = np.where(cls_id == cid)[0]
        keep = nms(boxes[idx], score[idx], IOU_TH)
        for k in keep:
            j = idx[k]
            name = names[int(cid)] if int(cid) < len(names) else "cls_%d" % int(cid)
            res.append((int(cid), name, float(score[j]), boxes[j].tolist()))
    res.sort(key=lambda t: t[2], reverse=True)
    return res


def filter_person(all_dets):
    """从全类别检出结果里只保留 COCO class 0 (person)"""
    return [(name, score, box) for (cid, name, score, box) in all_dets
            if cid == PERSON_CLASS_ID]


def draw_pedestrians(bgr, dets):
    canvas = bgr.copy()
    for i, (name, sc, b) in enumerate(dets):
        x1, y1, x2, y2 = [int(v) for v in b]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = "person_%d %.2f" % (i + 1, sc)
        cv2.putText(canvas, label, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(canvas, "人数: %d" % len(dets), (18, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    return canvas


class PedestrianEngine(object):
    def __init__(self):
        self.sess = None
        self.names = None

    def load(self):
        if not os.path.isfile(OM_PATH):
            raise RuntimeError("行人模型不存在：%s" % OM_PATH)
        from ais_bench.infer.interface import InferSession
        self.sess = InferSession(DEVICE_ID, OM_PATH)
        self.names = load_names()

    def infer(self, bgr):
        blob, r, pad = preprocess(bgr, DIV255)
        pred = pick_pred(self.sess.infer([blob]))
        if pred is None:
            return []
        return filter_person(postprocess_all(pred, r, pad, bgr.shape, self.names))

    def release(self):
        if self.sess is not None:
            del self.sess
            self.sess = None
            gc.collect()

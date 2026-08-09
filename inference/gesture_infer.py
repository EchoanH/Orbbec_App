# -*- coding: utf-8 -*-
"""手势识别：保留 24_gesture.py 的已验证四类规则。"""

import ast
import gc
import os
import re

import cv2
import numpy as np

from .depth_utils import color_to_depth_point, get_distance


PALM_OM = "/root/echo/atc_work/palm_192_nr.om"
HAND_OM = "/root/echo/atc_work/handpose_224.om"
REF_PY = "/root/echo/ref/palm_detection_mp_palmdet.py"

PALM_INPUT = 192
HAND_INPUT = np.array([224, 224])
SCORE_TH, NMS_TH = 0.50, 0.30
HAND_CONF_TH = 0.50
THUMB_OPEN_TH = 0.85
EXT_TH = 0.0

PALM_BOX_PRE_SHIFT_VECTOR = np.array([0, 0], np.float32)
PALM_BOX_PRE_ENLARGE_FACTOR = 4
PALM_BOX_SHIFT_VECTOR = np.array([0, -0.4], np.float32)
PALM_BOX_ENLARGE_FACTOR = 3

FINGERS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8],
           [0, 9, 10, 11, 12], [0, 13, 14, 15, 16],
           [0, 17, 18, 19, 20]]
TIP = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
PIP = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}
FIVE_TIPS = [4, 8, 12, 16, 20]


def ground_truth(tag):
    if tag == "0retest": return "拳头"
    if tag.startswith("hand1_"):
        n = int(tag.split("_")[1])
        return "点赞" if n == 11 else "五指"
    if tag.startswith("hand2_"):
        n = int(tag.split("_")[1])
        return "V字" if n >= 1 else "待确认"
    if tag.startswith("hand_"): return "拳头"
    return "待确认"


def load_anchors(path):
    src = open(path, "r", encoding="utf-8").read()
    i = src.find("def _load_anchors"); j = src.find("np.array(", i)
    k = j + len("np.array("); depth = 1
    while k < len(src):
        if src[k] == "(": depth += 1
        elif src[k] == ")":
            depth -= 1
            if depth == 0: break
        k += 1
    body = re.sub(r",\s*dtype\s*=\s*np\.float32\s*$", "",
                  src[j+len("np.array("):k].strip())
    return np.array(ast.literal_eval(body), dtype=np.float32)


def nms(boxes, scores, th):
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = np.maximum(0, x2-x1)*np.maximum(0, y2-y1)
    order = scores.argsort()[::-1]; keep = []
    while order.size:
        i = order[0]; keep.append(i)
        if order.size == 1: break
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2-xx1)*np.maximum(0, yy2-yy1)
        order = order[1:][inter/(areas[i]+areas[order[1:]]-inter+1e-9) <= th]
    return keep


def palm_detect(sess, anchors, bgr):
    h, w = bgr.shape[:2]
    ratio = min(PALM_INPUT/float(h), PALM_INPUT/float(w))
    nh, nw = int(h*ratio), int(w*ratio)
    im = cv2.resize(bgr, (nw, nh))
    ph, pw = PALM_INPUT-nh, PALM_INPUT-nw
    left, top = pw//2, ph//2
    im = cv2.copyMakeBorder(im, top, ph-top, left, pw-left,
                            cv2.BORDER_CONSTANT, None, (0,0,0))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    blob = np.ascontiguousarray(im[np.newaxis,:,:,:])
    pad_bias = (np.array([left, top], np.float32)/ratio).astype(np.float32)
    outs = sess.infer([blob])
    o0 = np.array(outs[0]).astype(np.float32); o1 = np.array(outs[1]).astype(np.float32)
    if o0.shape[-1] != 18: o0, o1 = o1, o0
    score = 1.0/(1.0+np.exp(-np.clip(o1[0,:,0].astype(np.float64), -60, 60)))
    scale = float(max(w, h))
    cxy = o0[0,:,0:2]/PALM_INPUT; wh = o0[0,:,2:4]/PALM_INPUT
    xy1 = (cxy - wh/2 + anchors)*scale; xy2 = (cxy + wh/2 + anchors)*scale
    boxes = np.concatenate([xy1, xy2], 1) - np.array(
        [pad_bias[0], pad_bias[1], pad_bias[0], pad_bias[1]], np.float32)
    lms = (o0[0,:,4:].reshape(-1,7,2)/PALM_INPUT + anchors[:,None,:])*scale - pad_bias
    m = score > SCORE_TH
    if m.sum() == 0: return [], float(score.max())
    idx = np.where(m)[0]
    sel = idx[nms(boxes[idx], score[idx], NMS_TH)]
    return ([(np.concatenate([boxes[i], lms[i].reshape(-1)]).astype(np.float32),
              float(score[i])) for i in sel], float(score.max()))


def crop_and_pad(image, palm_bbox, for_rotation=False):
    wh = palm_bbox[1] - palm_bbox[0]
    shift = (PALM_BOX_PRE_SHIFT_VECTOR if for_rotation else PALM_BOX_SHIFT_VECTOR)*wh
    palm_bbox = palm_bbox + shift
    center = np.sum(palm_bbox, axis=0)/2
    wh = palm_bbox[1] - palm_bbox[0]
    sc = PALM_BOX_PRE_ENLARGE_FACTOR if for_rotation else PALM_BOX_ENLARGE_FACTOR
    half = wh*sc/2
    palm_bbox = np.array([center-half, center+half]).astype(np.int32)
    palm_bbox[:,0] = np.clip(palm_bbox[:,0], 0, image.shape[1])
    palm_bbox[:,1] = np.clip(palm_bbox[:,1], 0, image.shape[0])
    image = image[palm_bbox[0][1]:palm_bbox[1][1], palm_bbox[0][0]:palm_bbox[1][0], :]
    if image.size == 0: return None, palm_bbox, np.array([0,0], np.int32)
    side = int(np.linalg.norm(image.shape[:2]) if for_rotation else max(image.shape[:2]))
    ph, pw = side-image.shape[0], side-image.shape[1]
    l, t = pw//2, ph//2
    image = cv2.copyMakeBorder(image, t, ph-t, l, pw-l, cv2.BORDER_CONSTANT, None, (0,0,0))
    return image, palm_bbox, palm_bbox[0] - [l, t]


def preprocess(image, palm):
    pad_bias = np.array([0,0], dtype=np.int32)
    pb = palm[0:4].reshape(2,2)
    img, pb, bias = crop_and_pad(image, pb, True)
    if img is None: return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pad_bias = pad_bias + bias
    pb = pb - pad_bias
    plm = palm[4:18].reshape(7,2) - pad_bias
    p1, p2 = plm[0], plm[2]
    rad = np.pi/2 - np.arctan2(-(p2[1]-p1[1]), p2[0]-p1[0])
    rad = rad - 2*np.pi*np.floor((rad+np.pi)/(2*np.pi))
    ang = np.rad2deg(rad)
    c = np.sum(pb, axis=0)/2
    rm = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), ang, 1.0)
    rot = cv2.warpAffine(img, rm, (img.shape[1], img.shape[0]))
    homo = np.c_[plm, np.ones(plm.shape[0])]
    rlm = np.array([np.dot(homo, rm[0]), np.dot(homo, rm[1])])
    rb = np.array([np.amin(rlm, axis=1), np.amax(rlm, axis=1)])
    crop, rb, _ = crop_and_pad(rot, rb)
    if crop is None or crop.size == 0: return None
    blob = cv2.resize(crop, dsize=tuple(HAND_INPUT),
                      interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
    return np.ascontiguousarray(blob[np.newaxis,:,:,:]), rb, ang, rm, pad_bias


def postprocess(outs, rb, ang, rm, pad_bias):
    arrs = [np.array(o).astype(np.float32) for o in outs]
    lm_l = [a for a in arrs if a.size == 63]
    sc_l = [a for a in arrs if a.size == 1]
    if len(lm_l) < 2 or len(sc_l) < 2: return None
    lm = lm_l[0].reshape(-1,3).copy()
    conf = float(sc_l[0].reshape(-1)[0]); handed = float(sc_l[1].reshape(-1)[0])
    wh = rb[1]-rb[0]; sf = wh/HAND_INPUT
    lm[:,:2] = (lm[:,:2] - HAND_INPUT/2)*max(sf); lm[:,2] = lm[:,2]*max(sf)
    crm = cv2.getRotationMatrix2D((0,0), ang, 1.0)
    rl = np.dot(lm[:,:2], crm[:,:2])
    comp = np.array([[rm[0][0], rm[1][0]], [rm[0][1], rm[1][1]]])
    tr = np.array([rm[0][2], rm[1][2]])
    inv = np.c_[comp, np.array([-np.dot(comp[0],tr), -np.dot(comp[1],tr)])]
    c = np.append(np.sum(rb, axis=0)/2, 1)
    oc = np.array([np.dot(c, inv[0]), np.dot(c, inv[1])])
    lm[:,:2] = rl + oc + pad_bias
    return lm, conf, handed


def classify(lm):
    p = lm[:,:2]
    base = np.linalg.norm(p[0]-p[9]) + 1e-6
    def d(a,b): return np.linalg.norm(p[a]-p[b])/base
    ext = {f: (d(TIP[f],0)-d(PIP[f],0)) > EXT_TH for f in TIP}
    d49 = d(4,9)
    thumb_open = d49 > THUMB_OPEN_TH
    n = sum(ext.values())
    detail = "食%+.2f 中%+.2f 无名%+.2f 小%+.2f d49=%.2f" % (
        d(TIP["index"],0)-d(PIP["index"],0), d(TIP["middle"],0)-d(PIP["middle"],0),
        d(TIP["ring"],0)-d(PIP["ring"],0), d(TIP["pinky"],0)-d(PIP["pinky"],0), d49)
    if n == 4:
        return "五指", detail
    if ext["index"] and ext["middle"] and not ext["ring"] and not ext["pinky"]:
        return "V字", detail
    if n == 0:
        return ("点赞" if thumb_open else "拳头"), detail
    return "未知", detail


def draw(vis, lm, g):
    p = lm[:,:2].astype(np.int32)
    for f in FINGERS:
        for a, b in zip(f[:-1], f[1:]):
            cv2.line(vis, tuple(p[a]), tuple(p[b]), (0,255,0), 2)
    for i in range(21):
        cv2.circle(vis, tuple(p[i]), 3, (255,0,0), -1)
    for i in FIVE_TIPS:
        cv2.circle(vis, tuple(p[i]), 8, (0,0,255), 2)
        cv2.putText(vis, str(i), (p[i][0]+8, p[i][1]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    if g and g != "未知":
        cv2.putText(vis, g, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0,0,255), 4)
    return vis


class GestureEngine(object):
    def __init__(self):
        self.palm_session = None
        self.hand_session = None
        self.anchors = None

    def load(self):
        if not os.path.isfile(PALM_OM):
            raise RuntimeError("palm 模型不存在：%s" % PALM_OM)
        if not os.path.isfile(HAND_OM):
            raise RuntimeError("handpose 模型不存在：%s" % HAND_OM)
        if not os.path.isfile(REF_PY):
            raise RuntimeError("anchor 参考实现不存在：%s" % REF_PY)
        from ais_bench.infer.interface import InferSession
        self.anchors = load_anchors(REF_PY)
        self.palm_session = InferSession(0, PALM_OM)
        self.hand_session = InferSession(0, HAND_OM)

    def infer(self, bgr, depth_mm=None):
        palms, palm_score = palm_detect(self.palm_session, self.anchors, bgr)
        if not palms:
            return None, None, 0.0, None, "palm 未检出（最高 %.3f）" % palm_score
        pre = preprocess(bgr, palms[0][0])
        if pre is None:
            return None, None, 0.0, None, "手部裁剪失败"
        blob, rb, ang, rm, pad_bias = pre
        result = postprocess(self.hand_session.infer([blob]), rb, ang, rm, pad_bias)
        if result is None:
            return None, None, 0.0, None, "handpose 输出解析失败"
        lm, conf, handed = result
        if conf < HAND_CONF_TH:
            return None, None, conf, None, "handpose 置信度 %.3f" % conf
        gesture, detail = classify(lm)
        distance_cm = None
        if depth_mm is not None and getattr(depth_mm, "ndim", 0) >= 2:
            depth_x, depth_y = color_to_depth_point(
                lm[0][0], lm[0][1], bgr.shape, depth_mm.shape)
            distance_mm = get_distance(depth_mm, depth_x, depth_y)
            if distance_mm > 0.0:
                distance_cm = distance_mm / 10.0
        return lm, gesture, conf, distance_cm, detail

    def release(self):
        if self.hand_session is not None:
            del self.hand_session
            self.hand_session = None
        if self.palm_session is not None:
            del self.palm_session
            self.palm_session = None
        self.anchors = None
        gc.collect()

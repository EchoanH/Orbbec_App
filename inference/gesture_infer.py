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
DEBUG_GESTURE = os.environ.get("DEBUG_GESTURE", "0") == "1"
DEBUG_GESTURE_DIR = "debug_gesture"

PALM_INPUT = 192
PALM_CENTER_ROI_RATIO = 0.70
HAND_INPUT = np.array([224, 224])
SCORE_TH = 0.35
PALM_FALLBACK_SCORE_TH = 0.15
PALM_FALLBACK_MAX_ALTERNATIVES = 5
NMS_TH = 0.30
HAND_CONF_TH = 0.50
THUMB_OPEN_TH = 0.85
PIP_ANGLE_TH = 150.0
GESTURE_VALID_CLASSES = ("五指", "拳头", "点赞", "V字")

LM_SQUARE_RESCUE_SCALE = 3.0
BBOX_SQUARE_RESCUE_SCALE = 2.5
UNKNOWN_RESCUE_CONF_GAIN = 0.05
MULTI_PALM_MIN_SCORE = 0.60
MULTI_PALM_MIN_SCORE_RATIO = 0.87
MULTI_PALM_MIN_CONF_GAIN = 0.12
MULTI_PALM_MAX_IOU = 0.10

PALM_BOX_PRE_SHIFT_VECTOR = np.array([0, 0], np.float32)
PALM_BOX_PRE_ENLARGE_FACTOR = 4
PALM_BOX_SHIFT_VECTOR = np.array([0, -0.4], np.float32)
PALM_BOX_ENLARGE_FACTOR = 3

FINGERS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8],
           [0, 9, 10, 11, 12], [0, 13, 14, 15, 16],
           [0, 17, 18, 19, 20]]
PIP_ANGLE_POINTS = {
    "index": (5, 6, 7),
    "middle": (9, 10, 11),
    "ring": (13, 14, 15),
    "pinky": (17, 18, 19),
}
FIVE_TIPS = [4, 8, 12, 16, 20]


def _save_debug_image(name, bgr):
    os.makedirs(DEBUG_GESTURE_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(DEBUG_GESTURE_DIR, name), bgr)


def _save_palm_debug(image, palm):
    vis = image.copy()
    bbox = palm[0:4].reshape(2,2).astype(np.int32)
    landmarks = palm[4:18].reshape(7,2).astype(np.int32)
    cv2.rectangle(vis, tuple(bbox[0]), tuple(bbox[1]), (0,255,0), 2)
    for i, point in enumerate(landmarks):
        cv2.circle(vis, tuple(point), 4, (0,0,255), -1)
        cv2.putText(vis, str(i), (point[0]+5, point[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
    _save_debug_image("01_palm_bbox.jpg", vis)


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


def palm_candidate_score(box, score, image_shape):
    h, w = image_shape[:2]
    bw = max(0.0, float(box[2]-box[0]))
    bh = max(0.0, float(box[3]-box[1]))
    area_ratio = min(1.0, bw*bh/float(w*h))
    box_center = np.array([(box[0]+box[2])/2, (box[1]+box[3])/2])
    image_center = np.array([w/2.0, h/2.0])
    max_distance = np.linalg.norm(image_center) + 1e-6
    center_score = max(0.0, 1.0-np.linalg.norm(box_center-image_center)/max_distance)
    return float(score)*0.6 + area_ratio*0.2 + center_score*0.2


def _palm_detect_once(sess, anchors, bgr, save_input=False):
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
    if save_input and DEBUG_GESTURE:
        palm_input = np.clip(blob[0]*255.0, 0, 255).astype(np.uint8)
        _save_debug_image("palm_input.jpg",
                          cv2.cvtColor(palm_input, cv2.COLOR_RGB2BGR))
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
    return boxes, lms, score


def _collect_palm_candidates(sess, anchors, bgr, score_th):
    h, w = bgr.shape[:2]
    boxes, lms, score = _palm_detect_once(sess, anchors, bgr, True)

    roi_h = max(1, int(h*PALM_CENTER_ROI_RATIO))
    roi_w = max(1, int(w*PALM_CENTER_ROI_RATIO))
    roi_y = (h-roi_h)//2
    roi_x = (w-roi_w)//2
    roi = bgr[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    roi_boxes, roi_lms, roi_score = _palm_detect_once(sess, anchors, roi)
    roi_boxes += np.array([roi_x, roi_y, roi_x, roi_y], np.float32)
    roi_lms += np.array([roi_x, roi_y], np.float32)

    boxes = np.concatenate([boxes, roi_boxes], axis=0)
    lms = np.concatenate([lms, roi_lms], axis=0)
    score = np.concatenate([score, roi_score], axis=0)
    m = score > score_th
    if m.sum() == 0: return [], float(score.max())
    idx = np.where(m)[0]
    print("score candidates:", len(idx))
    if DEBUG_GESTURE:
        vis = bgr.copy()
        for i in idx:
            box = boxes[i].astype(np.int32)
            cv2.rectangle(vis, tuple(box[0:2]), tuple(box[2:4]), (0,255,0), 2)
            cv2.putText(vis, "idx=%d score=%.3f" % (i, score[i]),
                        (box[0], max(15, box[1]-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        _save_debug_image("palm_candidates.jpg", vis)
    sel = idx[nms(boxes[idx], score[idx], NMS_TH)]
    final_scores = {int(i): palm_candidate_score(boxes[i], score[i], bgr.shape)
                    for i in sel}
    sel = sorted(sel, key=lambda i: final_scores[int(i)], reverse=True)
    print("palm candidates:")
    for i in sel:
        print("idx=%d score=%.3f final=%.3f box=%s" % (
            i, score[i], final_scores[int(i)], boxes[i]))
    return ([(np.concatenate([boxes[i], lms[i].reshape(-1)]).astype(np.float32),
              float(score[i]), int(i)) for i in sel], float(score.max()))


def palm_detect(sess, anchors, bgr):
    candidates, max_score = _collect_palm_candidates(
        sess, anchors, bgr, SCORE_TH)
    return [(palm, score) for palm, score, _ in candidates], max_score


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
    if DEBUG_GESTURE:
        _save_debug_image("02_rotate.jpg", cv2.cvtColor(rot, cv2.COLOR_RGB2BGR))
    homo = np.c_[plm, np.ones(plm.shape[0])]
    rlm = np.array([np.dot(homo, rm[0]), np.dot(homo, rm[1])])
    rb = np.array([np.amin(rlm, axis=1), np.amax(rlm, axis=1)])
    crop, rb, _ = crop_and_pad(rot, rb)
    if crop is None or crop.size == 0: return None
    if DEBUG_GESTURE:
        _save_debug_image("03_final_crop.jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    blob = cv2.resize(crop, dsize=tuple(HAND_INPUT),
                      interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
    if DEBUG_GESTURE:
        handpose_input = np.clip(blob*255.0, 0, 255).astype(np.uint8)
        _save_debug_image("04_handpose_input.jpg",
                          cv2.cvtColor(handpose_input, cv2.COLOR_RGB2BGR))
    return np.ascontiguousarray(blob[np.newaxis,:,:,:]), rb, ang, rm, pad_bias


def preprocess_square(image, palm, strategy, scale):
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

    if strategy == "lm":
        lm_min = np.amin(rlm, axis=1)
        lm_max = np.amax(rlm, axis=1)
        lm_center = (lm_min + lm_max)/2
        lm_wh = lm_max - lm_min
        source_side = max(lm_wh[0], lm_wh[1])
        roi_center = lm_center + PALM_BOX_SHIFT_VECTOR*source_side
    else:
        bbox_wh = pb[1] - pb[0]
        source_side = max(bbox_wh[0], bbox_wh[1])
        roi_center = c + PALM_BOX_SHIFT_VECTOR*source_side

    roi_side = source_side*scale
    half = roi_side/2
    rb = np.round([roi_center-half, roi_center+half]).astype(np.int32)
    rb[:,0] = np.clip(rb[:,0], 0, rot.shape[1])
    rb[:,1] = np.clip(rb[:,1], 0, rot.shape[0])
    crop = rot[rb[0][1]:rb[1][1], rb[0][0]:rb[1][0], :]
    if crop.size == 0: return None
    side = max(crop.shape[:2])
    ph, pw = side-crop.shape[0], side-crop.shape[1]
    l, t = pw//2, ph//2
    crop = cv2.copyMakeBorder(crop, t, ph-t, l, pw-l,
                              cv2.BORDER_CONSTANT, None, (0,0,0))
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


def safe_angle_2d(a, b, c):
    ba = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    bc = np.asarray(c, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(ba)*np.linalg.norm(bc))
    if denom <= 1e-6:
        return 0.0
    cosine = float(np.dot(ba, bc))/denom
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def classify(lm):
    p = lm[:,:2]
    base = np.linalg.norm(p[0]-p[9]) + 1e-6
    def d(a,b): return np.linalg.norm(p[a]-p[b])/base
    angles = {f: safe_angle_2d(p[a], p[b], p[c])
              for f, (a, b, c) in PIP_ANGLE_POINTS.items()}
    ext = {f: angle >= PIP_ANGLE_TH for f, angle in angles.items()}
    d49 = d(4,9)
    thumb_open = d49 > THUMB_OPEN_TH
    n = sum(ext.values())
    detail = "食%.1f 中%.1f 无名%.1f 小%.1f d49=%.2f" % (
        angles["index"], angles["middle"], angles["ring"],
        angles["pinky"], d49)
    if n == 4:
        return "五指", detail
    if ext["index"] and ext["middle"] and not ext["ring"] and not ext["pinky"]:
        return "V字", detail
    if n == 0:
        return ("点赞" if thumb_open else "拳头"), detail
    return "未知", detail


def bbox_iou(box_a, box_b):
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, x2-x1)*max(0.0, y2-y1)
    area_a = max(0.0, float(box_a[2]-box_a[0]))*max(
        0.0, float(box_a[3]-box_a[1]))
    area_b = max(0.0, float(box_b[2]-box_b[0]))*max(
        0.0, float(box_b[3]-box_b[1]))
    return inter/(area_a+area_b-inter+1e-9)


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
        fallback_palms, palm_score = _collect_palm_candidates(
            self.palm_session, self.anchors, bgr, PALM_FALLBACK_SCORE_TH)
        if not fallback_palms:
            return None, None, 0.0, None, "palm 未检出（最高 %.3f）" % palm_score

        normal_palms = [candidate for candidate in fallback_palms
                        if candidate[1] > SCORE_TH]
        primary = normal_palms[0] if normal_palms else None
        if primary is not None:
            alternatives = [candidate for candidate in fallback_palms
                            if candidate[2] != primary[2]]
        else:
            alternatives = fallback_palms
        alternatives = alternatives[:PALM_FALLBACK_MAX_ALTERNATIVES]

        if DEBUG_GESTURE:
            _save_palm_debug(bgr, (primary or alternatives[0])[0])

        def run_candidate(candidate, preprocess_mode="current"):
            palm = candidate[0]
            if preprocess_mode == "lm_square3.0":
                pre = preprocess_square(
                    bgr, palm, "lm", LM_SQUARE_RESCUE_SCALE)
            elif preprocess_mode == "bbox_square2.5":
                pre = preprocess_square(
                    bgr, palm, "bbox", BBOX_SQUARE_RESCUE_SCALE)
            else:
                pre = preprocess(bgr, palm)
            if pre is None:
                return None, "手部裁剪失败"
            blob, rb, ang, rm, pad_bias = pre
            result = postprocess(
                self.hand_session.infer([blob]), rb, ang, rm, pad_bias)
            if result is None:
                return None, "handpose 输出解析失败"
            return result, None

        best_result = None
        best_candidate = None
        last_error = "手部裁剪失败"
        if primary is not None:
            best_result, last_error = run_candidate(primary)
            if best_result is None:
                return None, None, 0.0, None, last_error
            best_candidate = primary
            if best_result[1] >= HAND_CONF_TH:
                alternatives = []

        for rank, candidate in enumerate(alternatives, 1):
            result, error = run_candidate(candidate)
            if result is None:
                last_error = error
                continue
            if best_result is None or result[1] > best_result[1]:
                best_result = result
                best_candidate = candidate
            if DEBUG_GESTURE:
                _, candidate_score, candidate_idx = candidate
                source = "full" if candidate_idx < len(self.anchors) else "center"
                anchor_idx = candidate_idx % len(self.anchors)
                print("gesture fallback: rank=%d source=%s anchor=%d "
                      "palm_score=%.3f hand_conf=%.3f" % (
                          rank, source, anchor_idx, candidate_score, result[1]))

        if best_result is None:
            return None, None, 0.0, None, last_error
        lm, conf, handed = best_result
        rescue_accepted = False
        if conf < HAND_CONF_TH:
            original_conf = conf
            rescue, _ = run_candidate(best_candidate, "lm_square3.0")
            if rescue is not None:
                rescue_lm, rescue_conf, rescue_handed = rescue
                rescue_gesture, rescue_detail = classify(rescue_lm)
                if (rescue_conf >= HAND_CONF_TH and
                        rescue_gesture in GESTURE_VALID_CLASSES):
                    lm, conf, handed = rescue_lm, rescue_conf, rescue_handed
                    gesture, detail = rescue_gesture, rescue_detail
                    rescue_accepted = True
                    if DEBUG_GESTURE:
                        print("gesture roi rescue: type=lm_square3.0 "
                              "old_conf=%.3f new_conf=%.3f gesture=%s" % (
                                  original_conf, conf, gesture))
            if not rescue_accepted:
                return (None, None, original_conf, None,
                        "handpose 置信度 %.3f" % original_conf)

        if not rescue_accepted:
            gesture, detail = classify(lm)
            if gesture == "未知":
                rescue, _ = run_candidate(best_candidate, "bbox_square2.5")
                if rescue is not None:
                    rescue_lm, rescue_conf, rescue_handed = rescue
                    rescue_gesture, rescue_detail = classify(rescue_lm)
                    if (rescue_conf >= HAND_CONF_TH and
                            rescue_gesture in GESTURE_VALID_CLASSES and
                            rescue_conf-conf >= UNKNOWN_RESCUE_CONF_GAIN):
                        old_conf = conf
                        lm, conf, handed = rescue_lm, rescue_conf, rescue_handed
                        gesture, detail = rescue_gesture, rescue_detail
                        rescue_accepted = True
                        if DEBUG_GESTURE:
                            print("gesture roi rescue: type=bbox_square2.5 "
                                  "old=未知 old_conf=%.3f new=%s "
                                  "new_conf=%.3f" % (
                                      old_conf, gesture, conf))

        if not rescue_accepted and gesture in GESTURE_VALID_CLASSES:
            selected_score = best_candidate[1]
            selected_bbox = best_candidate[0][0:4]
            qualifying = []
            for order, candidate in enumerate(fallback_palms):
                if (candidate[2] == best_candidate[2] or
                        candidate[1] <= SCORE_TH):
                    continue
                alternative, _ = run_candidate(
                    candidate, "bbox_square2.5")
                if alternative is None:
                    continue
                alt_lm, alt_conf, alt_handed = alternative
                alt_gesture, alt_detail = classify(alt_lm)
                score_ratio = candidate[1]/selected_score
                conf_gain = alt_conf-conf
                iou = bbox_iou(selected_bbox, candidate[0][0:4])
                if (alt_conf >= HAND_CONF_TH and
                        alt_gesture in GESTURE_VALID_CLASSES and
                        alt_gesture != gesture and
                        candidate[1] >= MULTI_PALM_MIN_SCORE and
                        score_ratio >= MULTI_PALM_MIN_SCORE_RATIO and
                        conf_gain >= MULTI_PALM_MIN_CONF_GAIN and
                        iou <= MULTI_PALM_MAX_IOU):
                    qualifying.append((
                        alt_conf, candidate[1], -order, alternative,
                        alt_gesture, alt_detail, score_ratio, conf_gain, iou))

            if qualifying:
                selected = max(qualifying, key=lambda item: item[0:3])
                old_gesture, old_conf = gesture, conf
                old_palm_score = selected_score
                _, new_palm_score, _, alternative, gesture, detail, \
                    score_ratio, conf_gain, iou = selected
                lm, conf, handed = alternative
                if DEBUG_GESTURE:
                    print("gesture multi-palm switch: old=%s old_conf=%.3f "
                          "old_palm=%.3f new=%s new_conf=%.3f "
                          "new_palm=%.3f ratio=%.3f gain=%.3f iou=%.3f" % (
                              old_gesture, old_conf, old_palm_score,
                              gesture, conf, new_palm_score, score_ratio,
                              conf_gain, iou))
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

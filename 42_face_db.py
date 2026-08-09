# -*- coding: utf-8 -*-
"""
42_face_db.py
人脸录入系统核心库：特征库的增删查、多张取平均、质量把关。
enrollment.py 和 compare.py 都 import 这个模块，不要在别处重复实现这些函数。

YuNet 检测/解码逻辑、SFace 对齐/特征提取逻辑原样复用自已验证的
06/07_yunet_check.py/detect.py、08_face_feature.py、13_sface_discrim.py、
41_face_enroll_threshold.py，未做任何改动。
新增内容仅为：json 特征库读写、多张特征取平均、按检测置信度做质量把关。
"""
import os
import json
import datetime
import numpy as np
import cv2

YUNET = "/root/echo/atc_work/yunet_640.om"
SFACE = "/root/echo/atc_work/sface_112.om"
DB_PATH = "/root/echo/face_db.json"

SIZE, DEVICE_ID = 640, 0
STRIDES = [8, 16, 32]
NMS_TH = 0.30
DIV255 = False  # B11 已确认：raw 区分度 0.887 远高于 div255 的 0.143

# 两套不同场景的检测置信度门槛（同一个 YuNet 输出，只是用途不同）：
# 注册场景可控性更高（本人配合、静态拍照），门槛更严，从源头挡掉低质量样本
ENROLL_CONF_TH = 0.75
# 识别场景更不可控（现场光线、摄像头画质、抓拍时机），门槛沿用原有验证值
MATCH_CONF_TH = 0.60

# 41_face_enroll_threshold.py 实测确定的判定阈值：
# 同人相似度 0.4938~0.9358，不同人 0.0421~0.3210，安全区间 [0.3210, 0.4938]
MATCH_THRESHOLD = 0.40

REF5 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


# ---- 以下函数原样复用自已验证脚本，逻辑未改动 ----

def build_index(sess):
    keys = ["%s_%d" % (p, s) for p in ("cls", "obj", "bbox", "kps") for s in STRIDES]
    try:
        names = [d.name for d in sess.get_outputs()]
    except Exception:
        names = []
    idx = {}
    for k in keys:
        for i, n in enumerate(names):
            if n.endswith(k):
                idx[k] = i; break
    return idx if len(idx) == len(keys) else {k: i for i, k in enumerate(keys)}


def yunet_decode(outs, idx, W, H):
    sx, sy = W / float(SIZE), H / float(SIZE)
    B, S, K = [], [], []
    for s in STRIDES:
        cls = np.array(outs[idx["cls_%d" % s]]).astype(np.float32).reshape(-1)
        obj = np.array(outs[idx["obj_%d" % s]]).astype(np.float32).reshape(-1)
        bb  = np.array(outs[idx["bbox_%d" % s]]).astype(np.float32).reshape(-1, 4)
        kp  = np.array(outs[idx["kps_%d" % s]]).astype(np.float32).reshape(-1, 10)
        cols = SIZE // s
        ar = np.arange(bb.shape[0])
        c = (ar % cols).astype(np.float32); r = (ar // cols).astype(np.float32)
        score = np.sqrt(np.clip(cls, 0, 1) * np.clip(obj, 0, 1))
        cx = (c + bb[:, 0]) * s; cy = (r + bb[:, 1]) * s
        w = np.exp(bb[:, 2]) * s; h = np.exp(bb[:, 3]) * s
        B.append(np.stack([(cx - w/2)*sx, (cy - h/2)*sy, (cx + w/2)*sx, (cy + h/2)*sy], 1))
        S.append(score)
        k = np.empty((bb.shape[0], 5, 2), dtype=np.float32)
        for j in range(5):
            k[:, j, 0] = (c + kp[:, 2*j])     * s * sx
            k[:, j, 1] = (r + kp[:, 2*j + 1]) * s * sy
        K.append(k)
    return np.concatenate(B), np.concatenate(S), np.concatenate(K)


def nms(boxes, scores, th):
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
    order = scores.argsort()[::-1]; keep = []
    while order.size:
        i = order[0]; keep.append(i)
        if order.size == 1: break
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
        order = order[1:][inter / (areas[i] + areas[order[1:]] - inter + 1e-9) <= th]
    return keep


def detect_face(sess, idx, bgr, conf_th):
    """返回最高分的一张脸 (score, box, kps5)，没有则 None。conf_th 由调用方传入，
    以便注册/识别用不同门槛。"""
    H, W = bgr.shape[:2]
    im = cv2.resize(bgr, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    blob = np.ascontiguousarray(np.expand_dims(im.transpose(2, 0, 1), 0).astype(np.float32))
    outs = sess.infer([blob])
    b, s, k = yunet_decode(outs, idx, W, H)
    m = s > conf_th
    if m.sum() == 0:
        return None
    b, s, k = b[m], s[m], k[m]
    keep = nms(b, s, NMS_TH)
    i = keep[0]
    return float(s[i]), b[i], k[i]


def align(bgr, kps5):
    M, _ = cv2.estimateAffinePartial2D(kps5.astype(np.float32), REF5, method=cv2.LMEDS)
    if M is None:
        return None
    return cv2.warpAffine(bgr, M, (112, 112), flags=cv2.INTER_LINEAR)


def sface_feat(sess, face112, div255):
    x = face112.astype(np.float32)
    if div255:
        x = x / 255.0
    x = np.ascontiguousarray(np.expand_dims(x.transpose(2, 0, 1), 0).astype(np.float32))
    out = sess.infer([x])
    return np.array(out[0]).astype(np.float32).reshape(-1)


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---- 以下是本脚本新增部分：特征库读写 + 质量把关 + 多张平均 ----

def load_db():
    if not os.path.isfile(DB_PATH):
        return {}
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_db(db):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


def extract_feature_for_enroll(det, idx, rec, bgr):
    """
    注册场景专用：用更严格的 ENROLL_CONF_TH 做质量把关。
    返回 (feat, score) 成功；返回 (None, score_or_None, reason) 失败，
    reason 用于提示用户具体是什么问题（未检出人脸 / 置信度不足 / 对齐失败）。
    """
    r = detect_face(det, idx, bgr, ENROLL_CONF_TH)
    if r is None:
        return None, None, "未检测到清晰人脸（置信度需 > %.2f），建议正对摄像头、光线充足后重拍" % ENROLL_CONF_TH
    sc, box, k5 = r
    a = align(bgr, k5)
    if a is None:
        return None, sc, "人脸关键点定位异常，建议重拍"
    feat = sface_feat(rec, a, DIV255)
    return feat, sc, None


def extract_feature_for_match(det, idx, rec, bgr):
    """识别场景专用：用较宽松的 MATCH_CONF_TH。返回同上（不含质量把关提示语的严格版本）。"""
    r = detect_face(det, idx, bgr, MATCH_CONF_TH)
    if r is None:
        return None, None, "未检测到人脸"
    sc, box, k5 = r
    a = align(bgr, k5)
    if a is None:
        return None, sc, "人脸关键点定位异常"
    feat = sface_feat(rec, a, DIV255)
    return feat, sc, None


def enroll_person(name, feats):
    """
    feats: list of np.ndarray(128,)，通常是同一人多张照片各自提取的特征。
    多张取平均后归一化存入库，覆盖式写入（重复注册同名会覆盖旧记录）。
    """
    if not feats:
        raise ValueError("feats 不能为空")
    avg = np.mean(np.stack(feats, axis=0), axis=0)
    db = load_db()
    db[name] = {
        "feature": avg.tolist(),
        "enrolled_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sample_count": len(feats),
    }
    save_db(db)
    return db[name]


def match_feature(feat, threshold=MATCH_THRESHOLD):
    """
    在库中找相似度最高的人。
    返回 (name, similarity) 若相似度 >= threshold；
    否则返回 (None, best_similarity)，best_similarity 可能是 0.0（库为空）。
    """
    db = load_db()
    if not db:
        return None, 0.0
    best_name, best_sim = None, -1.0
    for name, rec in db.items():
        db_feat = np.array(rec["feature"], dtype=np.float32)
        sim = cos(feat, db_feat)
        if sim > best_sim:
            best_name, best_sim = name, sim
    if best_sim >= threshold:
        return best_name, best_sim
    return None, best_sim


def list_enrolled():
    db = load_db()
    return [(name, rec.get("sample_count", 1), rec.get("enrolled_at", "?"))
            for name, rec in db.items()]


def delete_person(name):
    """从特征库中删除指定姓名的记录。返回 True 表示删除成功，False 表示该姓名不存在。"""
    db = load_db()
    if name not in db:
        return False
    del db[name]
    save_db(db)
    return True

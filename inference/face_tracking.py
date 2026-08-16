"""从既有 YuNet 输出派生候选框，并维护轻量人脸目标锁定。"""

import math

import numpy as np

from inference.face_db_adapter import MATCH_CONF_TH, NMS_TH, nms, yunet_decode
from inference.target_tracker import normalized_bbox_center


FACE_LOCK_MAX_MISSES = 3
FACE_LOCK_MIN_IOU = 0.05
FACE_LOCK_CENTER_DISTANCE_FACTOR = 1.5


def decode_face_boxes(outputs, output_index, frame_shape,
                      confidence_threshold=MATCH_CONF_TH):
    """复用一次已完成的 YuNet infer 输出，返回 NMS 后的全部人脸框。"""
    if outputs is None:
        return []
    height, width = frame_shape[:2]
    boxes, scores, _keypoints = yunet_decode(
        outputs, output_index, width, height)
    mask = scores > float(confidence_threshold)
    if not np.any(mask):
        return []
    boxes = boxes[mask]
    scores = scores[mask]
    keep = nms(boxes, scores, NMS_TH)
    return [boxes[index].astype(np.float32).tolist() for index in keep]


def normalized_face_center(box, frame_shape):
    """将 [x1, y1, x2, y2] 人脸框中心转换为 [-1, 1] 误差。"""
    x1, y1, x2, y2 = [float(value) for value in box]
    return normalized_bbox_center(
        (x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)),
        frame_shape)


class FaceTargetLock(object):
    """面积初始化、IoU/中心距离保持以及连续丢失后重选。"""

    def __init__(self, max_misses=FACE_LOCK_MAX_MISSES):
        self.max_misses = max(1, int(max_misses))
        self._locked_box = None
        self._misses = 0

    @property
    def locked_box(self):
        if self._locked_box is None:
            return None
        return self._locked_box.copy()

    @property
    def misses(self):
        return self._misses

    def reset(self):
        self._locked_box = None
        self._misses = 0

    def update(self, candidates):
        valid = [_as_valid_box(box) for box in candidates]
        valid = [box for box in valid if box is not None]
        if self._locked_box is None:
            return self._select_largest(valid)

        matched = self._match_locked(valid)
        if matched is not None:
            self._locked_box = matched
            self._misses = 0
            return matched.copy()

        self._misses += 1
        if self._misses < self.max_misses:
            return None

        self.reset()
        return self._select_largest(valid)

    def _select_largest(self, candidates):
        if not candidates:
            return None
        selected = max(candidates, key=_box_area)
        self._locked_box = selected
        self._misses = 0
        return selected.copy()

    def _match_locked(self, candidates):
        if not candidates:
            return None
        ranked = sorted(
            candidates,
            key=lambda box: (-_box_iou(self._locked_box, box),
                             _center_distance(self._locked_box, box)),
        )
        best = ranked[0]
        if _box_iou(self._locked_box, best) >= FACE_LOCK_MIN_IOU:
            return best
        previous_diagonal = math.hypot(
            self._locked_box[2] - self._locked_box[0],
            self._locked_box[3] - self._locked_box[1],
        )
        if (_center_distance(self._locked_box, best) <=
                previous_diagonal * FACE_LOCK_CENTER_DISTANCE_FACTOR):
            return best
        return None


def _as_valid_box(box):
    try:
        values = np.asarray(box, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if values.size < 4 or not np.all(np.isfinite(values[:4])):
        return None
    result = values[:4].copy()
    if result[2] <= result[0] or result[3] <= result[1]:
        return None
    return result


def _box_area(box):
    return float((box[2] - box[0]) * (box[3] - box[1]))


def _box_iou(first, second):
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = _box_area(first) + _box_area(second) - intersection
    return intersection / union if union > 0.0 else 0.0


def _center_distance(first, second):
    first_x = (float(first[0]) + float(first[2])) * 0.5
    first_y = (float(first[1]) + float(first[3])) * 0.5
    second_x = (float(second[0]) + float(second[2])) * 0.5
    second_y = (float(second[1]) + float(second[3])) * 0.5
    return math.hypot(second_x - first_x, second_y - first_y)

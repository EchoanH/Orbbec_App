"""轻量目标跟踪：稀疏光流平移 + 预测框附近的局部模板校正。"""

import cv2
import numpy as np


# 以下参数均为教学演示初始值，需要在 Atlas 板端按实际目标与帧率调整。
MIN_ROI_SIZE = 16
MAX_CORNERS = 80
QUALITY_LEVEL = 0.01
MIN_FEATURE_DISTANCE = 5
FEATURE_BLOCK_SIZE = 7
MIN_FLOW_POINTS = 5
REDETECT_POINT_COUNT = 14
LK_WIN_SIZE = (21, 21)
LK_MAX_LEVEL = 3
LK_MAX_ERROR = 30.0
MOTION_RESIDUAL_FLOOR_PX = 3.0
MOTION_RESIDUAL_SCALE = 2.5
TEMPLATE_CHECK_INTERVAL = 5
TEMPLATE_SEARCH_MARGIN_RATIO = 0.45
TEMPLATE_SEARCH_MARGIN_MIN_PX = 18
TEMPLATE_MATCH_THRESHOLD = 0.60
TEMPLATE_CORRECTION_BLEND = 0.35
TEMPLATE_MAX_SIDE = 160


def normalized_bbox_center(bbox, frame_shape):
    """返回 bbox 中心相对画面中心的归一化坐标，范围 [-1, 1]。"""
    if bbox is None:
        return None
    frame_height, frame_width = frame_shape[:2]
    if frame_width <= 0 or frame_height <= 0:
        return None
    x, y, width, height = bbox
    center_x = float(x) + float(width) / 2.0
    center_y = float(y) + float(height) / 2.0
    nx = (center_x - frame_width / 2.0) / (frame_width / 2.0)
    ny = (center_y - frame_height / 2.0) / (frame_height / 2.0)
    return (float(np.clip(nx, -1.0, 1.0)),
            float(np.clip(ny, -1.0, 1.0)))


class TargetTracker(object):
    IDLE = "IDLE"
    TRACKING = "TRACKING"
    LOST = "LOST"

    def __init__(self):
        self.state = self.IDLE
        self.bbox = None
        self._previous_gray = None
        self._points = None
        self._template = None
        self._template_scale = 1.0
        self._frame_index = 0
        self.last_match_score = None

    def clear(self):
        self.state = self.IDLE
        self.bbox = None
        self._previous_gray = None
        self._points = None
        self._template = None
        self._template_scale = 1.0
        self._frame_index = 0
        self.last_match_score = None

    def initialize(self, bgr_frame, roi):
        """用原始 BGR 帧和 (x, y, w, h) ROI 初始化跟踪。"""
        self.clear()
        if bgr_frame is None or getattr(bgr_frame, "size", 0) == 0:
            return False
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        bbox = self._sanitize_roi(roi, gray.shape)
        if bbox is None:
            return False
        x, y, width, height = [int(round(value)) for value in bbox]
        template = gray[y:y + height, x:x + width]
        if template.size == 0:
            return False

        self.bbox = tuple(float(value) for value in bbox)
        self._previous_gray = gray
        self._points = self._detect_features(gray, self.bbox)
        self._template, self._template_scale = self._prepare_template(template)
        self._frame_index = 0
        self.state = self.TRACKING
        return True

    def update(self, bgr_frame):
        """更新一帧；成功返回 (x, y, w, h)，丢失返回 None。"""
        if self.state != self.TRACKING or self.bbox is None:
            return None
        if bgr_frame is None or getattr(bgr_frame, "size", 0) == 0:
            self._mark_lost()
            return None

        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        self._frame_index += 1
        predicted_bbox = self.bbox
        tracked_points = None
        flow_reliable = False

        if self._points is not None and len(self._points) >= MIN_FLOW_POINTS:
            next_points, status, errors = cv2.calcOpticalFlowPyrLK(
                self._previous_gray, gray, self._points, None,
                winSize=LK_WIN_SIZE, maxLevel=LK_MAX_LEVEL,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                          20, 0.03))
            if next_points is not None and status is not None:
                valid = status.reshape(-1).astype(bool)
                if errors is not None:
                    valid &= errors.reshape(-1) <= LK_MAX_ERROR
                old_points = self._points.reshape(-1, 2)[valid]
                new_points = next_points.reshape(-1, 2)[valid]
                if len(new_points) >= MIN_FLOW_POINTS:
                    displacement = new_points - old_points
                    median_motion = np.median(displacement, axis=0)
                    residual = np.linalg.norm(
                        displacement - median_motion, axis=1)
                    median_residual = float(np.median(residual))
                    residual_limit = max(
                        MOTION_RESIDUAL_FLOOR_PX,
                        median_residual * MOTION_RESIDUAL_SCALE)
                    inliers = residual <= residual_limit
                    if int(np.count_nonzero(inliers)) >= MIN_FLOW_POINTS:
                        median_motion = np.median(displacement[inliers], axis=0)
                        predicted_bbox = self._move_bbox(
                            self.bbox, median_motion[0], median_motion[1],
                            gray.shape)
                        tracked_points = new_points[inliers].reshape(-1, 1, 2)
                        flow_reliable = True

        needs_template = (
            not flow_reliable or
            self._frame_index % TEMPLATE_CHECK_INTERVAL == 0 or
            tracked_points is None or
            len(tracked_points) < REDETECT_POINT_COUNT)
        template_bbox = None
        template_score = None
        if needs_template:
            template_bbox, template_score = self._match_template(
                gray, predicted_bbox)
            self.last_match_score = template_score

        template_reliable = (
            template_bbox is not None and
            template_score is not None and
            template_score >= TEMPLATE_MATCH_THRESHOLD)
        template_corrected = False
        if template_reliable:
            if flow_reliable:
                predicted_bbox = self._blend_bbox(
                    predicted_bbox, template_bbox,
                    TEMPLATE_CORRECTION_BLEND, gray.shape)
            else:
                predicted_bbox = template_bbox
            template_corrected = True
        elif not flow_reliable:
            self._mark_lost()
            return None

        self.bbox = predicted_bbox
        self._previous_gray = gray
        if (template_corrected or tracked_points is None or
                len(tracked_points) < REDETECT_POINT_COUNT):
            self._points = self._detect_features(gray, self.bbox)
        else:
            self._points = tracked_points
        return self.bbox

    def _mark_lost(self):
        self.state = self.LOST
        self.bbox = None
        self._previous_gray = None
        self._points = None

    @staticmethod
    def _sanitize_roi(roi, frame_shape):
        if roi is None or len(roi) != 4:
            return None
        frame_height, frame_width = frame_shape[:2]
        x, y, width, height = [float(value) for value in roi]
        if width < 0:
            x += width
            width = -width
        if height < 0:
            y += height
            height = -height
        x1 = max(0.0, min(float(frame_width), x))
        y1 = max(0.0, min(float(frame_height), y))
        x2 = max(0.0, min(float(frame_width), x + width))
        y2 = max(0.0, min(float(frame_height), y + height))
        width = x2 - x1
        height = y2 - y1
        if width < MIN_ROI_SIZE or height < MIN_ROI_SIZE:
            return None
        return x1, y1, width, height

    @staticmethod
    def _move_bbox(bbox, delta_x, delta_y, frame_shape):
        x, y, width, height = bbox
        frame_height, frame_width = frame_shape[:2]
        x = float(np.clip(x + float(delta_x), 0.0,
                          max(0.0, frame_width - width)))
        y = float(np.clip(y + float(delta_y), 0.0,
                          max(0.0, frame_height - height)))
        return x, y, width, height

    @staticmethod
    def _blend_bbox(flow_bbox, template_bbox, amount, frame_shape):
        x = flow_bbox[0] * (1.0 - amount) + template_bbox[0] * amount
        y = flow_bbox[1] * (1.0 - amount) + template_bbox[1] * amount
        return TargetTracker._move_bbox(
            (x, y, flow_bbox[2], flow_bbox[3]), 0.0, 0.0, frame_shape)

    @staticmethod
    def _prepare_template(template):
        height, width = template.shape[:2]
        longest = max(width, height)
        if longest <= TEMPLATE_MAX_SIDE:
            return template.copy(), 1.0
        scale = TEMPLATE_MAX_SIDE / float(longest)
        resized = cv2.resize(
            template,
            (max(1, int(round(width * scale))),
             max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA)
        return resized, scale

    @staticmethod
    def _detect_features(gray, bbox):
        x, y, width, height = [int(round(value)) for value in bbox]
        roi = gray[y:y + height, x:x + width]
        if roi.size == 0:
            return None
        points = cv2.goodFeaturesToTrack(
            roi, maxCorners=MAX_CORNERS, qualityLevel=QUALITY_LEVEL,
            minDistance=MIN_FEATURE_DISTANCE, blockSize=FEATURE_BLOCK_SIZE)
        if points is None:
            return None
        points = points.astype(np.float32)
        points[:, 0, 0] += x
        points[:, 0, 1] += y
        return points

    def _match_template(self, gray, predicted_bbox):
        if self._template is None or predicted_bbox is None:
            return None, None
        frame_height, frame_width = gray.shape[:2]
        x, y, width, height = predicted_bbox
        margin_x = max(TEMPLATE_SEARCH_MARGIN_MIN_PX,
                       int(round(width * TEMPLATE_SEARCH_MARGIN_RATIO)))
        margin_y = max(TEMPLATE_SEARCH_MARGIN_MIN_PX,
                       int(round(height * TEMPLATE_SEARCH_MARGIN_RATIO)))
        search_x1 = max(0, int(np.floor(x)) - margin_x)
        search_y1 = max(0, int(np.floor(y)) - margin_y)
        search_x2 = min(frame_width, int(np.ceil(x + width)) + margin_x)
        search_y2 = min(frame_height, int(np.ceil(y + height)) + margin_y)
        search = gray[search_y1:search_y2, search_x1:search_x2]
        if search.size == 0:
            return None, None

        if self._template_scale != 1.0:
            search = cv2.resize(
                search,
                (max(1, int(round(search.shape[1] * self._template_scale))),
                 max(1, int(round(search.shape[0] * self._template_scale)))),
                interpolation=cv2.INTER_AREA)
        template_height, template_width = self._template.shape[:2]
        if (search.shape[0] < template_height or
                search.shape[1] < template_width):
            return None, None
        scores = cv2.matchTemplate(
            search, self._template, cv2.TM_CCOEFF_NORMED)
        _, max_score, _, max_location = cv2.minMaxLoc(scores)
        if not np.isfinite(max_score):
            return None, None
        match_x = search_x1 + max_location[0] / self._template_scale
        match_y = search_y1 + max_location[1] / self._template_scale
        matched = self._move_bbox(
            (match_x, match_y, width, height), 0.0, 0.0, gray.shape)
        return matched, float(max_score)

"""Shared depth sampling helpers for color-frame landmark coordinates."""

import numpy as np


def get_distance(depth_mm, x, y):
    """读取指定位置附近的深度中值，单位为毫米。"""
    height, width = depth_mm.shape
    x = max(6, min(x, width - 7))
    y = max(6, min(y, height - 7))
    area = depth_mm[y - 6:y + 7, x - 6:x + 7]
    valid = area[(area > 100) & (area < 10000)]
    if valid.size == 0:
        return 0.0
    return float(np.median(valid))


def color_to_depth_point(x, y, color_shape, depth_shape):
    """Map a point from a BGR image shape to a depth array shape."""
    color_height, color_width = color_shape[:2]
    depth_height, depth_width = depth_shape[:2]
    if color_width <= 0 or color_height <= 0:
        return 0, 0
    depth_x = int(x * depth_width / float(color_width))
    depth_y = int(y * depth_height / float(color_height))
    return depth_x, depth_y

# -*- coding: utf-8 -*-
"""UI 层轻量绘制：等效替代 inference/face14_infer.py 里的可视化函数。

背景（P0 性能优化）：
    原 draw_face_guide 为了画一个半透明椭圆，做了三次全帧运算：
        canvas = img.copy()          # 2.76MB
        overlay = canvas.copy()      # 2.76MB
        cv2.addWeighted(...)         # 读 2×2.76MB、写 2.76MB
    在 A55 上这部分开销占单帧绘制的大头，而视觉产出仅是一个引导线框。

本模块的处理：
    1. 去掉引导椭圆（用户确认不再需要）；
    2. 无人脸时直接返回原帧，零拷贝；
    3. 有人脸时只做一次必要拷贝，其余绘制均为局部小面积操作。

注意：
    inference/ 目录为只读，本模块不修改其中任何代码，只是提供等效实现供
    ui/pages/ 调用。颜色常量与 14 点分部位映射仍从 inference 侧读取，
    保证配色与已确认的 14 点体系完全一致。
"""

import cv2
import numpy as np

import inference.face14_infer as _f14


def draw_pedestrians_fast(bgr, dets):
    """绘制行人框，中文标签由页面通过 Qt 绘制。"""
    canvas = bgr.copy()
    for _, _, box in dets:
        x1, y1, x2, y2 = [int(value) for value in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return canvas


def draw_face14_fast(img_bgr, face14=None, face_box=None, score=None):
    """等效 draw_face14，但去掉引导椭圆与多余全帧拷贝。

    img_bgr 不会被就地修改（该数组同时被推理 worker 引用，必须只读）。
    无检测结果时直接返回原数组，不做任何拷贝。
    """
    if face14 is None or face_box is None or score is None:
        return img_bgr

    part_map = _f14.FACE14_BY_PART
    if not part_map:
        return img_bgr

    # 唯一一次全帧拷贝：避免污染 worker 正在使用的原始帧。
    canvas = img_bgr.copy()
    height, width = canvas.shape[:2]

    x1, y1, x2, y2 = [int(v) for v in face_box]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    cv2.rectangle(canvas, (x1, y1), (x2, y2), _f14.FACE_BOX_COLOR, 2)

    for part, names in part_map.items():
        color = _f14.PART_COLOR[part]
        for name in names:
            point = face14.get(name)
            if point is None:
                continue
            x, y = int(point[0]), int(point[1])
            if x < 0 or y < 0 or x >= width or y >= height:
                continue
            cv2.circle(canvas, (x, y), 4, color, -1)
            cv2.circle(canvas, (x, y), 4, (255, 255, 255), 1)
    return canvas

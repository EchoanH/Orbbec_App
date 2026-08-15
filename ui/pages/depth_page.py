"""深度伪彩与中心测距页面。"""

import cv2

from inference.depth_utils import get_distance
from ui.draw_utils import draw_text_box_bgr

from .base_page import BasePage


MAX_DISPLAY_DEPTH_MM = 5000


class DepthPage(BasePage):
    page_title = "深度识别"
    page_hint = "Orbbec 深度图 · 实时测距"

    def process_frame(self, bgr_frame, bgr_display=None, depth_frame=None):
        if depth_frame is None:
            fallback = bgr_display if bgr_display is not None else bgr_frame
            return fallback, "深度数据不可用"

        depth_height, depth_width = depth_frame.shape
        center_x = depth_width // 2
        center_y = depth_height // 2
        center_distance_mm = get_distance(
            depth_frame, center_x, center_y)

        # uint16 毫米深度按 0～5000 mm 饱和映射到 uint8，避免 NumPy
        # 除法产生 float64 全帧临时数组。
        depth_8bit = cv2.convertScaleAbs(
            depth_frame, alpha=255.0 / MAX_DISPLAY_DEPTH_MM)
        rendered = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
        cv2.drawMarker(
            rendered, (center_x, center_y), (255, 255, 255),
            markerType=cv2.MARKER_CROSS, markerSize=32, thickness=2)

        target_width, target_height, target_scale = self.compute_target_size(
            depth_width, depth_height)
        if target_scale < 1.0:
            rendered = cv2.resize(
                rendered, (target_width, target_height),
                interpolation=cv2.INTER_LINEAR)

        if center_distance_mm > 0.0:
            distance_text = "中心距离 %.1f cm" % (center_distance_mm / 10.0)
            status_text = "深度数据正常 · %s" % distance_text
        else:
            distance_text = "中心距离 -- cm"
            status_text = "等待有效深度数据"
        rendered = draw_text_box_bgr(
            rendered, distance_text, 20, 20, font_size=15,
            text_color=(8, 19, 31),
            background_color=(74, 158, 255))
        return rendered, status_text

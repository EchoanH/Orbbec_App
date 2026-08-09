"""Small Qt-backed helpers for drawing Unicode text on BGR frames."""

import numpy as np
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter


def draw_text_box_bgr(img_bgr, text, x, y, font_size=16,
                      text_color=(255, 255, 255), background_color=None,
                      padding=5):
    """Draw one Unicode text line into a small BGR region."""
    if not text:
        return img_bgr
    canvas = np.ascontiguousarray(img_bgr)
    height, width = canvas.shape[:2]
    if width <= 0 or height <= 0:
        return canvas

    font = QFont("Droid Sans Fallback", font_size, QFont.Bold)
    metrics = QFontMetrics(font)
    box_width = min(width, metrics.horizontalAdvance(text) + padding * 2)
    box_height = min(height, metrics.height() + padding * 2)
    x = max(0, min(int(x), width - box_width))
    y = max(0, min(int(y), height - box_height))

    roi = canvas[y:y + box_height, x:x + box_width]
    rgb = roi[:, :, ::-1].copy()
    image = QImage(rgb.data, box_width, box_height, int(rgb.strides[0]),
                   QImage.Format_RGB888).copy()
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setFont(font)
    if background_color is not None:
        painter.fillRect(QRectF(0, 0, box_width, box_height),
                         QColor(*background_color))
    painter.setPen(QColor(*text_color))
    painter.drawText(QRectF(padding, 0, box_width - padding, box_height),
                     Qt.AlignVCenter | Qt.AlignLeft, text)
    painter.end()

    rendered = image.convertToFormat(QImage.Format_RGB888)
    ptr = rendered.bits()
    ptr.setsize(rendered.byteCount())
    rows = np.frombuffer(ptr, dtype=np.uint8).reshape(
        box_height, rendered.bytesPerLine())
    drawn_rgb = rows[:, :box_width * 3].reshape(box_height, box_width, 3)
    canvas[y:y + box_height, x:x + box_width] = drawn_rgb[:, :, ::-1]
    return canvas

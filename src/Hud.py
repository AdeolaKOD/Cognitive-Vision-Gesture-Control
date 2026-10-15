import cv2
import numpy as np

from HandTrackingModule import BONE_COLOR, RING_COLOR, RING_HOLE_COLOR

FONT = cv2.FONT_HERSHEY_SIMPLEX
TRACK_COLOR = (30, 30, 30)  # Dark translucent track behind each bar
FILL_COLOR = BONE_COLOR  # Same light grey as the finger lines
TEXT_COLOR = (255, 255, 255)


def _text(img, text, org, scale, thickness=1):
    cv2.putText(img, text, org, FONT, scale, TEXT_COLOR, thickness, cv2.LINE_AA)


def _centered_text(img, text, cx, y, scale, thickness=1):
    (tw, _th), _ = cv2.getTextSize(text, FONT, scale, thickness)
    _text(img, text, (cx - tw // 2, y), scale, thickness)


def drawLevelBar(img, x, label, percent, top=150, bottom=400, width=26):
    """Square, borderless level bar in the hand overlay colours: dark track, grey fill, white level edge."""
    percent = max(0.0, min(100.0, float(percent)))
    cx = x + width // 2

    # Translucent track, blended only inside the bar's own area to stay cheap
    roi = img[top:bottom, x:x + width]
    cv2.addWeighted(roi, 0.45, np.full_like(roi, TRACK_COLOR), 0.55, 0, roi)

    # Fill from the bottom up to the current level
    level = int(bottom - (bottom - top) * percent / 100)
    cv2.rectangle(img, (x, level), (x + width - 1, bottom - 1), FILL_COLOR, cv2.FILLED)
    cv2.rectangle(img, (x, level), (x + width - 1, min(bottom - 1, level + 2)), RING_COLOR, cv2.FILLED)

    _centered_text(img, label, cx, top - 14, 0.5, 1)
    _centered_text(img, f"{int(percent)}%", cx, bottom + 30, 0.6, 1)


def drawFps(img, fps):
    _text(img, f"{int(fps)} FPS", (20, 32), 0.6, 1)


def drawPinch(img, p1, p2):
    """Volume gesture: grey line between thumb and index, ends styled like the joints."""
    cv2.line(img, p1, p2, BONE_COLOR, 2, cv2.LINE_AA)
    for p in (p1, p2):
        cv2.circle(img, p, 10, RING_COLOR, cv2.FILLED, cv2.LINE_AA)
        cv2.circle(img, p, 8, RING_HOLE_COLOR, cv2.FILLED, cv2.LINE_AA)


def drawPalmRing(img, center, radius):
    """Brightness gesture: outlined ring on the palm that grows with hand size."""
    cv2.circle(img, center, radius, RING_COLOR, 2, cv2.LINE_AA)
    cv2.circle(img, center, max(3, radius // 6), RING_HOLE_COLOR, cv2.FILLED, cv2.LINE_AA)

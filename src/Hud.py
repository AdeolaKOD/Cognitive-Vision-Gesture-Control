import cv2

from HandTrackingModule import BONE_COLOR, RING_COLOR, RING_HOLE_COLOR

FONT = cv2.FONT_HERSHEY_SIMPLEX
TRACK_COLOR = (30, 30, 30)  # Dark translucent track behind each bar
TEXT_COLOR = (255, 255, 255)
SHADOW_COLOR = (20, 20, 20)


def _text(img, text, org, scale, thickness=1):
    # Soft shadow under white text keeps it readable on bright backgrounds
    x, y = org
    cv2.putText(img, text, (x + 1, y + 1), FONT, scale, SHADOW_COLOR, thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, scale, TEXT_COLOR, thickness, cv2.LINE_AA)


def _centered_text(img, text, cx, y, scale, thickness=1):
    (tw, _th), _ = cv2.getTextSize(text, FONT, scale, thickness)
    _text(img, text, (cx - tw // 2, y), scale, thickness)


def _pill(img, x, top, bottom, width, color, thickness=cv2.FILLED):
    # Vertical capsule: a thick line with round caps
    cx = x + width // 2
    r = width // 2
    if thickness == cv2.FILLED:
        cv2.line(img, (cx, top + r), (cx, bottom - r), color, width, cv2.LINE_AA)
    else:
        cv2.ellipse(img, (cx, top + r), (r, r), 0, 180, 360, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (cx, bottom - r), (r, r), 0, 0, 180, color, thickness, cv2.LINE_AA)
        cv2.line(img, (x, top + r), (x, bottom - r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x + width, top + r), (x + width, bottom - r), color, thickness, cv2.LINE_AA)


def drawLevelBar(img, x, label, percent, top=150, bottom=400, width=26):
    """Rounded vertical level bar with a ring knob, styled like the hand overlay."""
    percent = max(0.0, min(100.0, float(percent)))
    cx = x + width // 2
    r = width // 2

    # Translucent track, blended only inside the bar's own area to stay cheap
    pad = 4
    y0, y1 = max(0, top - pad), min(img.shape[0], bottom + pad)
    x0, x1 = max(0, x - pad), min(img.shape[1], x + width + pad)
    roi = img[y0:y1, x0:x1]
    overlay = roi.copy()
    _pill(overlay, x - x0, top - y0, bottom - y0, width, TRACK_COLOR)
    cv2.addWeighted(overlay, 0.55, roi, 0.45, 0, roi)
    _pill(img, x, top, bottom, width, BONE_COLOR, 1)

    # Fill from the bottom up to the current level
    level = int(bottom - r - (bottom - top - 2 * r) * percent / 100)
    inner = width - 8
    cv2.line(img, (cx, bottom - r), (cx, level), RING_COLOR, inner, cv2.LINE_AA)

    # Donut knob marks the level, matching the fingertip rings
    cv2.circle(img, (cx, level), r + 3, RING_COLOR, cv2.FILLED, cv2.LINE_AA)
    cv2.circle(img, (cx, level), max(2, (r + 3) // 3), RING_HOLE_COLOR, cv2.FILLED, cv2.LINE_AA)

    _centered_text(img, label.upper(), cx, top - 14, 0.5, 1)
    _centered_text(img, f"{int(percent)}%", cx, bottom + 32, 0.7, 2)


def drawFps(img, fps):
    _text(img, f"{int(fps)} FPS", (20, 32), 0.6, 1)


def drawPinch(img, p1, p2):
    """Volume gesture: grey line between thumb and index with ring ends."""
    cv2.line(img, p1, p2, BONE_COLOR, 3, cv2.LINE_AA)
    for p in (p1, p2):
        cv2.circle(img, p, 12, RING_COLOR, cv2.FILLED, cv2.LINE_AA)
        cv2.circle(img, p, 4, RING_HOLE_COLOR, cv2.FILLED, cv2.LINE_AA)


def drawPalmRing(img, center, radius):
    """Brightness gesture: outlined ring on the palm that grows with hand size."""
    cv2.circle(img, center, radius, RING_COLOR, 3, cv2.LINE_AA)
    cv2.circle(img, center, max(3, radius // 6), RING_COLOR, cv2.FILLED, cv2.LINE_AA)

import csv
import math
import os
import sys
import time

import cv2

import HandTrackingModule as hd
import Hud as hud
from BrightnessControl import BrightnessControl
from MediaControl import MediaControl
from VolumeControl import VolumeControl

# Webcam capture resolution
wCam, hCam = 640, 480

# Telemetry CSVs go to <project root>/testResult, next to the exe when frozen
if getattr(sys, "frozen", False):
    PROJECT_ROOT = os.path.dirname(sys.executable)
else:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(PROJECT_ROOT, "testResult")

LATENCY_HEADER = [
    "Acquisition",
    "Pre-processing",
    "Inference",
    "Logic",
    "Actuation",
    "Display",
]


def open_telemetry(name, header):
    """Open a telemetry CSV for writing; returns (file, writer) or (None, None)."""
    try:
        os.makedirs(RESULT_DIR, exist_ok=True)
        f = open(os.path.join(RESULT_DIR, name), "w", newline="")
    except OSError as e:
        # Telemetry is optional: never stop the app because the folder is read-only
        print(f"Telemetry disabled for {name}: {e}")
        return None, None
    writer = csv.writer(f)
    writer.writerow(header)
    return f, writer


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)  # CAP_PROP_FRAME_WIDTH
    cap.set(4, hCam)  # CAP_PROP_FRAME_HEIGHT

    detector = hd.HandDetector(detectionCon=0.7, maxHands=2)
    vol_ctrl = VolumeControl()
    bright_ctrl = BrightnessControl()
    media_ctrl = MediaControl()

    # Resizable so it can be snapped side-by-side with other apps
    cv2.namedWindow("Gesture Control", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    # Telemetry output files
    latency_file, latency_writer = open_telemetry("latency_data.csv", LATENCY_HEADER)
    transfer_file, transfer_writer = open_telemetry(
        "transfer_data.csv", ["Distance", "Percentage"]
    )

    try:
        run_loop(
            cap,
            detector,
            vol_ctrl,
            bright_ctrl,
            media_ctrl,
            latency_writer,
            transfer_writer,
        )
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()
        for f in (latency_file, transfer_file):
            if f is not None:
                f.close()


def run_loop(
    cap, detector, vol_ctrl, bright_ctrl, media_ctrl, latency_writer, transfer_writer
):
    pTime = 0

    # Seed the UI bars from the current system state
    volBar, volPer = vol_ctrl.get_volume_state()
    brightBar, brightPer = bright_ctrl.get_brightness_state()

    while True:
        # Acquisition
        t0 = time.perf_counter()
        success, img = cap.read()
        if not success:
            print("Failed to read webcam")
            break

        # Pre-processing: mirror so the left hand appears on the left of the screen
        t1 = time.perf_counter()
        img = cv2.flip(img, 1)
        t2 = time.perf_counter()

        # Inference: detect hands, landmarks and Left/Right classification
        # (findHands also does the BGR->RGB conversion and draws the landmark overlay)
        img = detector.findHands(img)
        hands_info = detector.findHandsInfo(img, draw=False)
        t3 = time.perf_counter()

        # Logic; time spent in OS calls is summed separately as actuation
        actuation = 0.0
        for hand in hands_info:
            handType = hand["type"]
            lmList = hand["lmList"]

            if len(lmList) != 0:
                fingers = detector.fingersUp(lmList, handType)  # 1 = up, 0 = down

                # Left hand: play/pause and media scrubbing via thumb-index pinch
                if handType == "Left":
                    x1, y1 = lmList[4][1], lmList[4][2]  # thumb tip
                    x2, y2 = lmList[8][1], lmList[8][2]  # index tip
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    length = math.hypot(
                        x2 - x1, y2 - y1
                    )  # Calculate the Euclidean distance between Thumb and Index
                    # Pass the data to the MediaControl module to process gestures
                    media_ctrl.process_gestures(img, length, cx, cy)
                    actuation += media_ctrl.actuation_time

                # Right hand: volume and brightness
                elif handType == "Right":
                    x_thumb, y_thumb = lmList[4][1], lmList[4][2]
                    x_index, y_index = lmList[8][1], lmList[8][2]
                    x_wrist, y_wrist = lmList[0][1], lmList[0][2]
                    x_mid_mcp, y_mid_mcp = (
                        lmList[9][1],
                        lmList[9][2],
                    )  # middle finger base

                    # Volume clutch: middle, ring and pinky down; thumb-index distance sets level
                    if (
                        len(fingers) == 5
                        and fingers[2] == 0
                        and fingers[3] == 0
                        and fingers[4] == 0
                    ):
                        length_vol = math.hypot(x_index - x_thumb, y_index - y_thumb)

                        hud.drawPinch(img, (x_thumb, y_thumb), (x_index, y_index))

                        ta = time.perf_counter()
                        volBar, volPer = vol_ctrl.set_volume(length_vol)
                        actuation += time.perf_counter() - ta

                        # Telemetry: transfer function (distance -> volume %)
                        if transfer_writer:
                            transfer_writer.writerow([length_vol, volPer])

                    # Brightness clutch: index, middle, ring and pinky all up
                    elif (
                        len(fingers) == 5
                        and fingers[1] == 1
                        and fingers[2] == 1
                        and fingers[3] == 1
                        and fingers[4] == 1
                    ):
                        # Wrist-to-middle-MCP length as a proxy for hand distance from camera
                        palm_size = math.hypot(x_mid_mcp - x_wrist, y_mid_mcp - y_wrist)

                        cx_palm, cy_palm = lmList[9][1], lmList[9][2]
                        hud.drawPalmRing(img, (cx_palm, cy_palm), int(palm_size / 4))

                        ta = time.perf_counter()
                        brightBar, brightPer = bright_ctrl.set_brightness(palm_size)
                        actuation += time.perf_counter() - ta

        t4 = time.perf_counter()

        # Display: FPS, HUD bars and the preview window
        cTime = time.perf_counter()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime

        hud.drawFps(img, fps)
        hud.drawLevelBar(img, 50, "Brightness", brightPer)
        hud.drawLevelBar(img, 560, "Volume", volPer)

        img_display = cv2.resize(img, (1280, 960))
        cv2.imshow("Gesture Control", img_display)
        key = cv2.waitKey(1) & 0xFF  # waitKey is what actually paints the window
        t5 = time.perf_counter()

        # Telemetry: per-stage latency in ms
        if latency_writer:
            latency_writer.writerow(
                [
                    (t1 - t0) * 1000,
                    (t2 - t1) * 1000,
                    (t3 - t2) * 1000,
                    (t4 - t3 - actuation) * 1000,
                    actuation * 1000,
                    (t5 - t4) * 1000,
                ]
            )

        # Exit on 'q' or when the window is closed
        if key == ord("q"):
            break

        if cv2.getWindowProperty("Gesture Control", cv2.WND_PROP_VISIBLE) < 1:
            break


if __name__ == "__main__":
    main()

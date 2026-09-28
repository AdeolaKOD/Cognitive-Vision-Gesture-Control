import cv2
import time
import math
import numpy as np
import csv
import os

import HandTrackingModule as hd
from VolumeControl import VolumeControl
from BrightnessControl import BrightnessControl
from MediaControl import MediaControl

# Webcam capture resolution
wCam, hCam = 640, 480


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)  # CAP_PROP_FRAME_WIDTH
    cap.set(4, hCam)  # CAP_PROP_FRAME_HEIGHT
    pTime = 0

    detector = hd.HandDetector(detectionCon=0.7, maxHands=2)
    vol_ctrl = VolumeControl()
    bright_ctrl = BrightnessControl()
    media_ctrl = MediaControl()

    # Seed the UI bars from the current system state
    volBar, volPer = vol_ctrl.get_volume_state()
    brightBar, brightPer = bright_ctrl.get_brightness_state()

    # Resizable so it can be snapped side-by-side with other apps
    cv2.namedWindow("Gesture Control", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    # Telemetry output files
    os.makedirs("testResult", exist_ok=True)
    latency_file = open("testResult/latency_data.csv", "w", newline="")
    latency_writer = csv.writer(latency_file)
    latency_writer.writerow(
        ["Acquisition", "Pre-processing", "Inference", "Logic", "Actuation"]
    )

    transfer_file = open("testResult/transfer_data.csv", "w", newline="")
    transfer_writer = csv.writer(transfer_file)
    transfer_writer.writerow(["Distance", "Percentage"])

    while True:
        # Acquisition
        t0 = time.time()
        success, img = cap.read()
        if not success:
            print("Failed to read webcam")
            break

        # Pre-processing: mirror so the left hand appears on the left of the screen
        t1 = time.time()
        img = cv2.flip(img, 1)
        t2 = time.time()

        # Inference: detect hands, landmarks and Left/Right classification
        img = detector.findHands(img)
        hands_info = detector.findHandsInfo(img, draw=False)
        t3 = time.time()

        # Logic
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

                    length = math.hypot(x2 - x1, y2 - y1)
                    media_ctrl.process_gestures(img, length, cx, cy)

                # Right hand: volume and brightness
                elif handType == "Right":
                    x_thumb, y_thumb = lmList[4][1], lmList[4][2]
                    x_index, y_index = lmList[8][1], lmList[8][2]
                    x_wrist, y_wrist = lmList[0][1], lmList[0][2]
                    x_mid_mcp, y_mid_mcp = lmList[9][1], lmList[9][2]  # middle finger base

                    # Volume clutch: middle, ring and pinky down; thumb-index distance sets level
                    if (
                        len(fingers) == 5
                        and fingers[2] == 0
                        and fingers[3] == 0
                        and fingers[4] == 0
                    ):
                        length_vol = math.hypot(x_index - x_thumb, y_index - y_thumb)

                        cv2.circle(img, (x_thumb, y_thumb), 10, (0, 255, 0), cv2.FILLED)
                        cv2.circle(img, (x_index, y_index), 10, (0, 255, 0), cv2.FILLED)
                        cv2.line(
                            img, (x_thumb, y_thumb), (x_index, y_index), (0, 255, 0), 3
                        )

                        volBar, volPer = vol_ctrl.set_volume(length_vol)

                        # Telemetry: transfer function (distance -> volume %)
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
                        cv2.circle(
                            img,
                            (cx_palm, cy_palm),
                            int(palm_size / 4),
                            (255, 255, 0),
                            cv2.FILLED,
                        )

                        brightBar, brightPer = bright_ctrl.set_brightness(palm_size)

        # FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime

        cv2.putText(
            img,
            f"FPS: {int(fps)}",
            (40, 50),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (255, 0, 0),
            3,
        )

        # Brightness bar (left)
        cv2.rectangle(img, (50, 150), (85, 400), (255, 255, 0), 3)
        cv2.rectangle(img, (50, int(brightBar)), (85, 400), (255, 255, 0), cv2.FILLED)
        cv2.putText(
            img,
            f"{int(brightPer)} %",
            (40, 450),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (255, 255, 0),
            3,
        )
        cv2.putText(
            img, "Bright", (40, 140), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 0), 2
        )

        # Volume bar (right)
        cv2.rectangle(img, (550, 150), (585, 400), (0, 255, 0), 3)
        cv2.rectangle(img, (550, int(volBar)), (585, 400), (0, 255, 0), cv2.FILLED)
        cv2.putText(
            img,
            f"{int(volPer)} %",
            (540, 450),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 255, 0),
            3,
        )
        cv2.putText(
            img, "Vol", (540, 140), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2
        )

        img_display = cv2.resize(img, (1280, 960))
        t4 = time.time()

        # Actuation
        cv2.imshow("Gesture Control", img_display)
        t5 = time.time()

        # Telemetry: per-stage latency in ms
        latency_writer.writerow(
            [
                (t1 - t0) * 1000,
                (t2 - t1) * 1000,
                (t3 - t2) * 1000,
                (t4 - t3) * 1000,
                (t5 - t4) * 1000,
            ]
        )

        # Exit on 'q' or when the window is closed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        if cv2.getWindowProperty("Gesture Control", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    latency_file.close()
    transfer_file.close()


if __name__ == "__main__":
    main()

# Import OpenCV for accessing the webcam and image processing
import cv2
import cv2
# Import time for calculating Frames Per Second (FPS)
import time
# Import math for calculating distances between hand landmarks
import math
# Import numpy for numerical operations and interpolations
import numpy as np
import csv
import os

# Custom Modules
# Import the custom hand tracking module
import HandTrackingModule as hd
# Import the volume control logic class
from VolumeControl import VolumeControl
# Import the brightness control logic class
from BrightnessControl import BrightnessControl
# Import the media control logic class (play/pause, scrub)
from MediaControl import MediaControl

################################
# Set the desired width and height for the webcam feed
wCam, hCam = 640, 480
################################

# Main function where the application logic runs
def main():
    # Initialize the webcam capture object (0 is usually the default built-in webcam)
    cap = cv2.VideoCapture(0)
    # Set the width property (ID 3) of the webcam capture
    cap.set(3, wCam)
    # Set the height property (ID 4) of the webcam capture
    cap.set(4, hCam)
    # Variable to keep track of previous time for FPS calculation
    pTime = 0

    # Initialize the HandDetector class with a 70% confidence threshold and max 2 hands
    detector = hd.HandDetector(detectionCon=0.7, maxHands=2)
    # Initialize the VolumeControl class instance
    vol_ctrl = VolumeControl()
    # Initialize the BrightnessControl class instance
    bright_ctrl = BrightnessControl()
    # Initialize the MediaControl class instance
    media_ctrl = MediaControl()

    # Initial UI states for the visual bars on the screen
    # Get actual initial states instead of hardcoding 0
    volBar, volPer = vol_ctrl.get_volume_state()
    brightBar, brightPer = bright_ctrl.get_brightness_state()

    # Create a resizable window so the user can snap it side-by-side with other apps
    cv2.namedWindow("Gesture Control", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    # Ensure testResult directory exists
    os.makedirs('testResult', exist_ok=True)
    # Open CSV files for writing live telemetry
    latency_file = open('testResult/latency_data.csv', 'w', newline='')
    latency_writer = csv.writer(latency_file)
    latency_writer.writerow(['Acquisition', 'Pre-processing', 'Inference', 'Logic', 'Actuation'])

    transfer_file = open('testResult/transfer_data.csv', 'w', newline='')
    transfer_writer = csv.writer(transfer_file)
    transfer_writer.writerow(['Distance', 'Percentage'])

    # Infinite loop to continuously read frames from the webcam
    while True:
        # Read a frame from the webcam. success is True if successful, img is the frame
        t0 = time.time()
        success, img = cap.read()
        # If the frame could not be read, exit the loop
        if not success:
            print("Failed to read webcam")
            break

        t1 = time.time()
        # Invert the image horizontally for a mirror effect so left physical hand is on left side of screen
        img = cv2.flip(img, 1)
        t2 = time.time()

        # Detect hands in the current frame and draw landmarks
        img = detector.findHands(img)
        # Get detailed information about the detected hands (landmarks and Left/Right classification)
        hands_info = detector.findHandsInfo(img, draw=False)
        t3 = time.time()
        
        # Iterate over each hand detected in the current frame
        for hand in hands_info:
            # Extract whether the hand is 'Left' or 'Right'
            handType = hand["type"]
            # Extract the list of 21 landmark coordinates for this hand
            lmList = hand["lmList"]
            
            # Ensure the landmark list is not empty
            if len(lmList) != 0:
                # Get the state of all 5 fingers (1 for up, 0 for down)
                fingers = detector.fingersUp(lmList, handType)
                
                # Check if it's the Left Hand
                if handType == "Left":
                    # Play/Pause & Media Scrubbing (Left Hand)
                    # Get coordinates of the Thumb tip (landmark 4)
                    x1, y1 = lmList[4][1], lmList[4][2] 
                    # Get coordinates of the Index finger tip (landmark 8)
                    x2, y2 = lmList[8][1], lmList[8][2] 
                    # Calculate the center point between Thumb and Index
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    # Calculate the Euclidean distance between Thumb and Index tips
                    length = math.hypot(x2 - x1, y2 - y1)
                    # Pass the data to the MediaControl module to process gestures
                    media_ctrl.process_gestures(img, length, cx, cy)

                # Check if it's the Right Hand
                elif handType == "Right":
                    # Coordinates for Thumb tip
                    x_thumb, y_thumb = lmList[4][1], lmList[4][2]
                    # Coordinates for Index finger tip
                    x_index, y_index = lmList[8][1], lmList[8][2]
                    # Coordinates for the Wrist (landmark 0)
                    x_wrist, y_wrist = lmList[0][1], lmList[0][2]
                    # Coordinates for Middle finger MCP (base joint, landmark 9)
                    x_mid_mcp, y_mid_mcp = lmList[9][1], lmList[9][2]
                    
                    # Volume Clutch Gesture: Middle (index 2), Ring (index 3), and Pinky (index 4) are DOWN
                    if len(fingers) == 5 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
                        # Calculate distance between Thumb and Index tips for volume level
                        length_vol = math.hypot(x_index - x_thumb, y_index - y_thumb)
                        
                        # Draw green circles and a line between Thumb and Index to show volume control is active
                        cv2.circle(img, (x_thumb, y_thumb), 10, (0, 255, 0), cv2.FILLED)
                        cv2.circle(img, (x_index, y_index), 10, (0, 255, 0), cv2.FILLED)
                        cv2.line(img, (x_thumb, y_thumb), (x_index, y_index), (0, 255, 0), 3)
                        
                        # Set the system volume and update the UI variables
                        volBar, volPer = vol_ctrl.set_volume(length_vol)
                        
                        # ==========================================
                        # TELEMETRY: Transfer Function Raw Data
                        # Source: 'length_vol' (Euclidean distance) and 'volPer' (Actuation Percentage)
                        # ==========================================
                        transfer_writer.writerow([length_vol, volPer])
                        
                    # Brightness Clutch Gesture: All 4 fingers (Index, Middle, Ring, Pinky) are UP
                    elif len(fingers) == 5 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:
                        # Calculate Z-Distance of Right Hand using distance from Wrist to Middle MCP as a proxy
                        palm_size = math.hypot(x_mid_mcp - x_wrist, y_mid_mcp - y_wrist)
                        
                        # Get the coordinates of the palm center (using Middle MCP)
                        cx_palm, cy_palm = lmList[9][1], lmList[9][2]
                        # Draw a yellow circle on the palm, size based on Z-distance
                        cv2.circle(img, (cx_palm, cy_palm), int(palm_size/4), (255, 255, 0), cv2.FILLED)
                        
                        # Set the system brightness and update the UI variables
                        brightBar, brightPer = bright_ctrl.set_brightness(palm_size)

        # FPS Calculation
        # Get current time
        cTime = time.time()
        # Calculate FPS: 1 divided by the time difference between current and previous frame
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        # Update previous time
        pTime = cTime
        
        # Display the FPS text in the top-left corner
        cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 
                    1, (255, 0, 0), 3)
        
        # UI Bars Section
        # Brightness Bar (Drawn on the Left Side)
        # Draw the outline of the brightness bar in yellow
        cv2.rectangle(img, (50, 150), (85, 400), (255, 255, 0), 3)
        # Draw the filled portion of the brightness bar
        cv2.rectangle(img, (50, int(brightBar)), (85, 400), (255, 255, 0), cv2.FILLED)
        # Display the brightness percentage text below the bar
        cv2.putText(img, f'{int(brightPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 
                    1, (255, 255, 0), 3)
        # Display a label above the bar
        cv2.putText(img, "Bright", (40, 140), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 0), 2)

        # Volume Bar (Drawn on the Right Side)
        # Draw the outline of the volume bar in green
        cv2.rectangle(img, (550, 150), (585, 400), (0, 255, 0), 3)
        # Draw the filled portion of the volume bar
        cv2.rectangle(img, (550, int(volBar)), (585, 400), (0, 255, 0), cv2.FILLED)
        # Display the volume percentage text below the bar
        cv2.putText(img, f'{int(volPer)} %', (540, 450), cv2.FONT_HERSHEY_COMPLEX, 
                    1, (0, 255, 0), 3)
        # Display a label above the bar
        cv2.putText(img, "Vol", (540, 140), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
        
        # Resize the final image to be larger (1280x960) before displaying
        img_display = cv2.resize(img, (1280, 960))
        t4 = time.time()
        # Show the image in a window named "Gesture Control"
        cv2.imshow("Gesture Control", img_display)
        t5 = time.time()
        
        # ==========================================
        # TELEMETRY: Latency Breakdown Raw Data
        # Source: Time differences between pipeline stages (t0 to t5) in milliseconds
        # ==========================================
        latency_writer.writerow([(t1-t0)*1000, (t2-t1)*1000, (t3-t2)*1000, (t4-t3)*1000, (t5-t4)*1000])
        
        # Check for key presses, waiting 1 millisecond between frames
        key = cv2.waitKey(1) & 0xFF
        # If the 'q' key is pressed, break the loop to exit
        if key == ord('q'):
            break
            
        # Check if the user clicked the 'X' button to close the window
        # WND_PROP_VISIBLE returns 0 if the window is closed
        if cv2.getWindowProperty("Gesture Control", cv2.WND_PROP_VISIBLE) < 1:
            break

    # Release the webcam resource
    cap.release()
    # Destroy all OpenCV windows
    cv2.destroyAllWindows()
    latency_file.close()
    transfer_file.close()

# Entry point of the script: run main() if the script is executed directly
if __name__ == "__main__":
    main()

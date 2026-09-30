# Import OpenCV for image processing and drawing
# Import math for calculating distances between points
import math

import cv2
import mediapipe as mp

# Import MediaPipe for hand tracking machine learning models


# Explicitly import the submodules so IDEs and PyInstaller can resolve them reliably


# Define a class for detecting and tracking hands
class HandDetector:
    # Initialize the hand detector with MediaPipe parameters
    def __init__(
        self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5
    ):
        # Static image mode (False means tracking is prioritized)
        self.mode = mode
        # Maximum number of hands to detect
        self.maxHands = maxHands
        # Complexity of the model (1 is default, balanced speed/accuracy)
        self.modelComplexity = modelComplexity
        # Minimum confidence threshold for detection
        self.detectionCon = detectionCon
        # Minimum confidence threshold for tracking
        self.trackCon = trackCon

        # Access MediaPipe's hand solution
        self.mpHands = mp.solutions.hands  # pyright: ignore[reportAttributeAccessIssue]
        # Initialize the Hands object with our parameters
        self.hands = self.mpHands.Hands(
            self.mode,
            self.maxHands,
            self.modelComplexity,
            self.detectionCon,
            self.trackCon,
        )
        # Access MediaPipe's drawing utilities for rendering landmarks
        self.mpDraw = mp.solutions.drawing_utils  # pyright: ignore[reportAttributeAccessIssue]
        # Landmark IDs for the tips of the 5 fingers (Thumb, Index, Middle, Ring, Pinky)
        self.tipIds = [4, 8, 12, 16, 20]

    # Method to find and draw hands on a given image
    def findHands(self, img, draw=True):
        # MediaPipe requires RGB images, so convert BGR (OpenCV default) to RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Process the image and get hand detection results
        self.results = self.hands.process(imgRGB)

        # If hands are detected in the frame
        if self.results.multi_hand_landmarks:
            # Iterate through each detected hand
            for handLms in self.results.multi_hand_landmarks:
                # If drawing is enabled, draw the connections and landmarks
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS
                    )
        # Return the modified image
        return img

    # Original position finding method (legacy/basic version)
    def findPosition(self, img, handNo=0, draw=True):
        # Create an empty list to store landmarks
        self.lmList = []
        # Check if hands were detected
        if self.results.multi_hand_landmarks:
            # Get the requested hand (default is index 0)
            myHand = self.results.multi_hand_landmarks[handNo]
            # Enumerate through all 21 landmarks of the hand
            for id, lm in enumerate(myHand.landmark):
                # Get the height, width, and channels of the image
                h, w, _c = img.shape
                # Convert normalized coordinates to pixel coordinates
                cx, cy = int(lm.x * w), int(lm.y * h)
                # Append the landmark ID and its coordinates
                self.lmList.append([id, cx, cy])
                # Draw a circle on the landmark if drawing is enabled
                if draw:
                    cv2.circle(img, (cx, cy), 5, (230, 230, 250), cv2.FILLED)
        # Return the list of landmarks
        return self.lmList

    # Enhanced method to find hand landmarks and classify them as Left or Right
    def findHandsInfo(self, img, draw=True):
        # List to store information about all detected hands
        hands_info = []
        # Check if any hands were detected
        if self.results.multi_hand_landmarks:
            # Iterate through all detected hands
            for i, handLms in enumerate(self.results.multi_hand_landmarks):
                # Get the dimensions of the image
                h, w, _c = img.shape
                # Get the X-coordinate of the wrist (landmark 0) to determine hand position
                wrist_x = int(handLms.landmark[0].x * w)

                # Since the image is horizontally flipped (mirror), left physical hand is on the left
                # Assign hand type based on its position on the screen
                if wrist_x < w // 2:
                    handType = "Left"  # Left side of screen = Media
                else:
                    handType = "Right"  # Right side of screen = Volume

                # Extract landmark coordinates for this hand
                lmList = []
                for id, lm in enumerate(handLms.landmark):
                    h, w, _c = img.shape
                    # Convert normalized coordinates (0.0 - 1.0) to pixel coordinates
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # Append the ID and coordinates
                    lmList.append([id, cx, cy])
                    # Optionally draw circles on landmarks
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (230, 230, 250), cv2.FILLED)
                # Store the hand's type and its landmark list
                hands_info.append({"type": handType, "lmList": lmList})

            # Logic to handle overlapping hands or edge cases where spatial classification fails
            # If exactly two hands are detected
            if len(hands_info) == 2:
                # Compare their wrist X-coordinates to enforce left-right assignment
                if hands_info[0]["lmList"][0][1] < hands_info[1]["lmList"][0][1]:
                    hands_info[0]["type"] = "Left"
                    hands_info[1]["type"] = "Right"
                else:
                    hands_info[0]["type"] = "Right"
                    hands_info[1]["type"] = "Left"

        # Return the list containing dictionaries for each hand
        return hands_info

    # Method to determine which fingers are currently pointing up
    def fingersUp(self, lmList, handType):
        fingers = []
        # Thumb Logic
        # Because the thumb joint moves differently, check X-coordinates relative to handedness
        if handType == "Right":
            # If the thumb tip is to the left of the joint below it (in a mirrored feed)
            if lmList[self.tipIds[0]][1] < lmList[self.tipIds[0] - 1][1]:
                fingers.append(1)  # Thumb is up/out
            else:
                fingers.append(0)  # Thumb is down/in
        else:  # Left Hand
            # If the thumb tip is to the right of the joint below it
            if lmList[self.tipIds[0]][1] > lmList[self.tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        # 4 Fingers Logic (Index, Middle, Ring, Pinky)
        for id in range(1, 5):
            # If the Y-coordinate of the tip is higher (smaller value) than the joint two segments down
            if lmList[self.tipIds[id]][2] < lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)  # Finger is up
            else:
                fingers.append(0)  # Finger is down
        return fingers

    # Helper method to find the distance between two specific landmarks
    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        # Get coordinates for the first point
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        # Get coordinates for the second point
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        # Calculate the center point between the two landmarks
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # If drawing is enabled, draw a line and circles connecting them
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)

        # Calculate the Euclidean distance (hypotenuse) between the points
        length = math.hypot(x2 - x1, y2 - y1)
        # Return the distance, the image, and the coordinates
        return length, img, [x1, y1, x2, y2, cx, cy]

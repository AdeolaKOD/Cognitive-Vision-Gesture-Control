# Import OpenCV for image processing and drawing
# Import math for calculating distances between points
import math

import cv2
import mediapipe as mp
import numpy as np

# Import MediaPipe for hand tracking machine learning models


# Explicitly import the submodules so IDEs and PyInstaller can resolve them reliably


# Hand overlay style (colours are BGR)
RING_COLOR = (255, 255, 255)  # Near joints: white ring
RING_HOLE_COLOR = (150, 150, 150)  # Grey centre that makes the ring hollow
BONE_COLOR = (225, 225, 225)  # Near bones: light grey
FAR_COLOR = (45, 45, 45)  # Far joints and bones fade to this dark grey
PALM_COLOR = (30, 30, 30)  # Thin dark V-shaped outline of the palm

# Finger chains from base to tip (thumb, index, middle, ring, pinky)
FINGER_CHAINS = [
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
]
# Lines across the knuckles and from the wrist to the thumb base
KNUCKLE_LINKS = [(0, 1), (5, 9), (9, 13), (13, 17)]
# Joint size relative to the hand size: tips biggest, knuckles and wrist smallest
JOINT_SCALE = {0: 0.05, 1: 0.05, 5: 0.05, 9: 0.05, 13: 0.05, 17: 0.05}
for _chain in FINGER_CHAINS:
    for _joint, _scale in zip(_chain[1:], (0.08, 0.10, 0.13)):
        JOINT_SCALE[_joint] = _scale


def _mix(far, near, t):
    return tuple(int(f + (n - f) * t) for f, n in zip(far, near))


def drawHandStyle(img, points, depths=None):
    """Draw one hand like MediaPipe's demo: near joints are big white rings,
    far ones shrink into dark dots, with a thin dark V across the palm.

    points: 21 (x, y) pixel tuples. depths: 21 MediaPipe z values (negative is
    closer to the camera than the wrist); without them every joint is drawn as near.
    """
    # Hand size (wrist to middle knuckle) keeps the overlay proportional at any distance
    handSize = math.hypot(points[9][0] - points[0][0], points[9][1] - points[0][1])
    lineW = max(1, int(handSize * 0.025))

    # Nearness per joint in 0..1, measured in hand lengths so it works at any distance
    if depths is None:
        near = [1.0] * 21
    else:
        scale = max(handSize / img.shape[1], 1e-3)  # z shares the x scale
        near = [min(1.0, max(0.0, 0.5 - z / scale * 1.5)) for z in depths]

    # Bones first so the joints sit on top of them
    for chain in FINGER_CHAINS:
        for a, b in zip(chain, chain[1:]):
            c = _mix(FAR_COLOR, BONE_COLOR, (near[a] + near[b]) / 2)
            cv2.line(img, points[a], points[b], c, lineW, cv2.LINE_AA)
    for a, b in KNUCKLE_LINKS:
        c = _mix(FAR_COLOR, BONE_COLOR, (near[a] + near[b]) / 2)
        cv2.line(img, points[a], points[b], c, lineW, cv2.LINE_AA)
    cv2.polylines(img, [np.array([points[5], points[0], points[17]])], False,
                  PALM_COLOR, lineW + 1, cv2.LINE_AA)

    # Joints: size and colour follow nearness, and only near ones get the hollow centre
    for i, base in JOINT_SCALE.items():
        t = near[i]
        r = max(2, int(handSize * base * (0.4 + 0.8 * t)))
        cv2.circle(img, points[i], r, _mix(FAR_COLOR, RING_COLOR, t), cv2.FILLED, cv2.LINE_AA)
        if t > 0.5 and r >= 5:
            cv2.circle(img, points[i], max(1, r // 3), RING_HOLE_COLOR, cv2.FILLED, cv2.LINE_AA)
    return img


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
        # Landmark IDs for the tips of the 5 fingers (Thumb, Index, Middle, Ring, Pinky)
        self.tipIds = [4, 8, 12, 16, 20]

    # Method to find and draw hands on a given image
    def findHands(self, img, draw=True):
        # MediaPipe requires RGB images, so convert BGR (OpenCV default) to RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Process the image and get hand detection results
        self.results = self.hands.process(imgRGB)

        # If hands are detected in the frame and drawing is enabled
        if draw and self.results.multi_hand_landmarks:
            h, w, _c = img.shape
            # Iterate through each detected hand
            for handLms in self.results.multi_hand_landmarks:
                # Convert normalized landmarks to pixel points and draw the custom overlay
                points = [(int(lm.x * w), int(lm.y * h)) for lm in handLms.landmark]
                depths = [lm.z for lm in handLms.landmark]
                drawHandStyle(img, points, depths)
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

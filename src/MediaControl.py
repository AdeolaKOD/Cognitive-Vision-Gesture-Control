# Import pyautogui to simulate keyboard presses for media control
import pyautogui
# Import time module to track timing of gestures (taps and holds)
import time
# Import OpenCV for drawing text and shapes on the webcam feed
import cv2

# Define a class for handling media gestures like play/pause and scrubbing
class MediaControl:
    # Initialize state variables to track gestures over time
    def __init__(self):
        # Time when the last tap occurred (used for double-tap detection)
        self.last_tap_time = 0
        # Counter for the number of consecutive taps
        self.tap_count = 0
        # Boolean state indicating if fingers are currently touching
        self.is_fingers_touching = False
        # Distance threshold (in pixels) to consider fingers as "touching" or "pinching"
        self.tap_threshold_dist = 40
        # Maximum delay (in seconds) allowed between taps to count as a double-tap
        self.double_tap_delay = 0.5
        # The X-coordinate where the pinch started, used as an anchor for scrubbing
        self.scrub_anchor_x = None
        # Time of the last scrub action to prevent overly rapid scrubbing
        self.last_scrub_time = 0
        # Time when the current touch gesture started, to differentiate taps from holds
        self.touch_start_time = 0
        # Boolean state indicating if a scrub gesture is actively happening
        self.is_scrubbing = False

    # Main method to process incoming hand data and determine if a gesture occurred
    def process_gestures(self, img, length, cx, cy):
        # Get the current time for evaluating gesture durations
        current_time = time.time()
        
        # When fingers are pinching (touching) based on distance threshold
        if length < self.tap_threshold_dist:
            # Draw a filled circle at the pinch center to visually indicate touching
            cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED)
            
            # If fingers were NOT touching in the previous frame
            if not self.is_fingers_touching:
                # Update state: they just touched
                self.is_fingers_touching = True
                # Record the start time of this touch
                self.touch_start_time = current_time
                # Set the starting X-coordinate for a potential scrub gesture
                self.scrub_anchor_x = cx
                # Reset scrubbing state since it's a fresh touch
                self.is_scrubbing = False
                # Reset the last scrub time
                self.last_scrub_time = current_time
            # If fingers were ALREADY touching (a continuous hold/pinch)
            else:
                # Calculate horizontal movement from the anchor point
                dx = cx - self.scrub_anchor_x
                # If the hand moved horizontally by more than 25 pixels, consider it a scrub
                if abs(dx) > 25:
                    # Update state to scrubbing
                    self.is_scrubbing = True
                    # Cancel any potential tap sequences since we are now scrubbing
                    self.tap_count = 0 
                    
                    # Ensure scrubbing actions don't happen too fast (debounce)
                    if (current_time - self.last_scrub_time) > 0.2:
                        # If hand moved right, fast-forward
                        if dx > 25:
                            # Simulate right arrow key press to skip forward
                            pyautogui.press('right')
                            # Display "Forward" text on the screen for user feedback
                            cv2.putText(img, "Forward", (cx - 30, cy - 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
                            # Update the anchor so the user can continue scrubbing
                            self.scrub_anchor_x = cx
                            # Update the last scrub time
                            self.last_scrub_time = current_time
                        # If hand moved left, rewind
                        elif dx < -25:
                            # Simulate left arrow key press to rewind
                            pyautogui.press('left')
                            # Display "Rewind" text on the screen
                            cv2.putText(img, "Rewind", (cx - 30, cy - 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
                            # Update the anchor so the user can continue scrubbing
                            self.scrub_anchor_x = cx
                            # Update the last scrub time
                            self.last_scrub_time = current_time
        # When fingers release (distance is larger than threshold)
        else:
            # If fingers WERE touching in the previous frame, they just released
            if self.is_fingers_touching:
                # Update state: they are no longer touching
                self.is_fingers_touching = False
                
                # Calculate how long the fingers were held together
                touch_duration = current_time - self.touch_start_time
                # If they didn't scrub, and the tap was quick enough (< 0.4s), it's a valid tap
                if not self.is_scrubbing and touch_duration < 0.4:
                    # Check if the time since the last tap is within the double-tap window
                    if (current_time - self.last_tap_time) < self.double_tap_delay:
                        # Increment tap count
                        self.tap_count += 1
                    else:
                        # It's a fresh tap sequence
                        self.tap_count = 1
                    
                    # Record this tap's time
                    self.last_tap_time = current_time
                    
                    # If two consecutive taps are detected
                    if self.tap_count == 2:
                        # Simulate media play/pause key
                        pyautogui.press('playpause')
                        # Display "Play/Pause" text on screen for feedback
                        cv2.putText(img, "Play/Pause", (cx, cy - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
                        # Reset tap count after triggering the action
                        self.tap_count = 0
            
            # Reset the scrub anchor since fingers are separated
            self.scrub_anchor_x = None

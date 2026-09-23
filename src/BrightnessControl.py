# Import the screen_brightness_control library to manage the display's brightness
import screen_brightness_control as sbc
# Import numpy for numerical operations, specifically for interpolating ranges
import numpy as np

# Define a class to encapsulate brightness control logic
class BrightnessControl:
    # Initialize the class (constructor)
    def __init__(self):
        # No specific initialization needed for brightness control right now
        pass
        
    # Method to calculate and set the brightness based on the distance (length) provided
    def set_brightness(self, length):
        # Map the input distance 'length' (expected between 40 and 120) to a brightness percentage (0 to 100)
        bright = np.interp(length, [40, 120], [0, 100])
        # Map the input distance to the UI brightness bar's Y-coordinate (from 400 at bottom to 150 at top)
        brightBar = np.interp(length, [40, 120], [400, 150])
        # Map the input distance to a brightness percentage specifically for UI display (0 to 100)
        brightPer = np.interp(length, [40, 120], [0, 100])
        
        # Try-except block to safely attempt setting the brightness without crashing if it fails
        try:
            # Set the actual system display brightness to the integer value of 'bright'
            sbc.set_brightness(int(bright))
        except:
            # If an error occurs (e.g., library not supported on this monitor), ignore and continue
            pass
            
        # Return the calculated bar coordinate and percentage to be drawn on the screen
        return brightBar, brightPer
        
    # Method to retrieve the current system brightness and calculate its UI representation
    def get_brightness_state(self):
        try:
            # Fetch the current system brightness list or value
            bright_list = sbc.get_brightness()
            # Handle cases where the library returns a list (e.g., multiple monitors) or a single value
            brightPer = bright_list[0] if isinstance(bright_list, list) else bright_list
            # Map the current brightness percentage to the UI bar's Y-coordinate
            brightBar = np.interp(brightPer, [0, 100], [400, 150])
            # Return the calculated bar coordinate and percentage
            return brightBar, brightPer
        except:
            # If an error occurs (e.g. brightness unsupported), return default UI values (0%)
            return 400, 0

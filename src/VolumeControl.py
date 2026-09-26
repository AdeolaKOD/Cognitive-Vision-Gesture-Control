# Import numpy for numerical operations, particularly mapping values using interpolation
import numpy as np
# Import necessary components from ctypes to interact with C-style data types
from ctypes import cast, POINTER
# Import comtypes component needed for interacting with the Windows audio API
from comtypes import CLSCTX_ALL
# Import AudioUtilities and IAudioEndpointVolume from pycaw for Windows system audio control
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Define a class to encapsulate the system volume control logic
class VolumeControl:
    # Initialize the class and set up the connection to the system's audio interface
    def __init__(self):
        # Get all audio output devices (speakers) connected to the system
        devices = AudioUtilities.GetSpeakers()
        # Access the endpoint volume interface of the audio devices
        interface = devices.EndpointVolume # type: ignore
        # Store the interface in a class variable for later use
        self.volume = interface
        # Get the supported volume range (minimum, maximum, and step values) from the system
        self.volRange = self.volume.GetVolumeRange()
        # Extract the minimum volume limit (usually a negative number in decibels, like -65.25)
        self.minVol = self.volRange[0]
        # Extract the maximum volume limit (usually 0.0 or a positive number in decibels)
        self.maxVol = self.volRange[1]
        
    # Method to calculate and set the volume based on the input distance (length)
    def set_volume(self, length):
        # Map the input finger distance (expected between 50 and 200) to the system's volume range in decibels
        vol = np.interp(length, [50, 200], [self.minVol, self.maxVol])
        # Map the distance to the UI volume bar's Y-coordinate on the screen (400 at bottom to 150 at top)
        volBar = np.interp(length, [50, 200], [400, 150])
        # Map the distance to a percentage value (0 to 100) for display on the UI
        volPer = np.interp(length, [50, 200], [0, 100])
        
        # Set the master volume level of the system to the calculated decibel value 'vol'
        self.volume.SetMasterVolumeLevel(vol, None)
        
        # Return the bar coordinate and percentage so the main UI can draw them
        return volBar, volPer
        
    # Method to retrieve the current system volume level and calculate its UI representation
    def get_volume_state(self):
        try:
            # Get the current master volume level as a scalar value between 0.0 and 1.0
            vol_scalar = self.volume.GetMasterVolumeLevelScalar()
            # Convert the scalar value to a percentage (0 to 100)
            volPer = vol_scalar * 100
            # Map the volume percentage to the UI volume bar's Y-coordinate on the screen
            volBar = np.interp(volPer, [0, 100], [400, 150])
            # Return the bar coordinate and percentage so the UI can draw them correctly
            return volBar, volPer
        except:
            # If an error occurs while getting the volume, return default UI values (0%)
            return 400, 0

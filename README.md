# Hand Gesture Control

An intuitive, real-time hand gesture control system built with Python, OpenCV, MediaPipe, and PyAutoGUI. This application allows users to control system volume, screen brightness, and media playback using webcam hand gestures.

## Features

- 🖐️ **Hand Tracking**: Powered by MediaPipe Hands for high-accuracy 21-landmark 3D hand keypoint detection.
- 🔊 **Volume Control**: Pinch gestures between thumb and index finger to adjust system master volume.
- ☀️ **Brightness Control**: Gesture-based screen brightness adjustments.
- ⏯️ **Media Control**: Play/pause, track skip, and media scrubbing via intuitive finger gestures.
- 📊 **Telemetry & Performance Diagnostics**: Built-in benchmark suite to log latency and CPU utilization metrics.

## Project Structure

```
HandGestureControl/
├── src/
│   ├── main.py                   # Main application entry point
│   ├── HandTrackingModule.py     # MediaPipe hand tracking wrapper
│   ├── VolumeControl.py          # System volume control module (PyCAW)
│   ├── BrightnessControl.py      # Screen brightness control module
│   └── MediaControl.py           # PyAutoGUI media controls
├── test/
│   ├── cpu_utilization.py        # Benchmark for CPU usage
│   ├── latency_breakdown.py      # Latency breakdown tests
│   └── transfer_function.py     # Control curve benchmarks
├── requirements.txt              # Python dependencies
├── HandGestureControl.spec       # PyInstaller build specification
└── build.bat                     # Windows build script
```

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/HandGestureControl.git
   cd HandGestureControl
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application:**
   ```bash
   python src/main.py
   ```

## Requirements

- Python 3.8+
- Webcam / Camera device
- Windows OS (for PyCAW volume control and screen-brightness-control)

## License

MIT License

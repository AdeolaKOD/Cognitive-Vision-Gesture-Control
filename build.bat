@echo off
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt

echo Building executable using PyInstaller...
pyinstaller --name "HandGestureControl" --onefile --windowed --collect-all mediapipe --add-data "models\hand_landmarker.task;models" src/main.py

echo Build complete. The executable should be located in the "dist" directory.
pause

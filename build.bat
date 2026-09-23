@echo off
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt

echo Building executable using PyInstaller...
pyinstaller --name "HandGestureControl" --onefile --windowed --collect-all mediapipe src/main.py

echo Build complete. The executable should be located in the "dist" directory.
pause

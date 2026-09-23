<div align="center">

# Cognitive Vision-Based Hand Gesture Recognition System

[![Python](https://img.shields.io/badge/Python-3.8%2B-242836?style=flat-square&logo=python)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.11-242836?style=flat-square)](https://google.github.io/mediapipe/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Python-242836?style=flat-square)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-242836?style=flat-square)](LICENSE)

**An edge-computing, real-time computer vision interface for bi-manual control of system volume, display brightness, and media navigation using geometric regression of hand landmarks.**

[Overview](#overview) • [Key Features](#key-features) • [Architecture](#architecture) • [Performance](#performance) • [Installation](#installation)

</div>

---

## Overview

The evolution of Human-Computer Interaction (HCI) continuously aims to reduce the "Gulf of Execution" between human intent and computer response. Traditional tactile input devices (mice, keyboards) present accessibility barriers and hygiene concerns in specific environments.

This repository contains the implementation of a **Cognitive Vision-Based Hand Gesture Recognition System**. By leveraging Google MediaPipe for CPU-optimized landmark tracking and utilizing geometric Euclidean distance models, this system translates real-world 3D hand poses into 1-dimensional continuous scalar values (e.g., system volume) without requiring expensive GPU accelerators or depth-sensing LiDAR cameras.

---

## Key Features

### 1. Geometric Regression (No Neural Network Classification)
Unlike legacy methods (HSV segmentation, Haar Cascades) which rely on rigid, discrete classifications of hand shapes, this framework utilizes **continuous regression**:
*   **Euclidean Distance Mapping:** Calculates the precise distance between specific nodes (e.g., Thumb tip and Index tip) to control continuous settings like volume (65 dB to 0 dB).
*   **Z-Axis Proxies:** Uses palm bounding box size as a proxy for depth (Z-axis) to control screen brightness, creating a natural push/pull interaction model.

### 2. Bi-Manual Topology Model
The system independently tracks and delegates tasks based on handedness:
*   **Right Hand:** Controls system parameters (Volume & Brightness) and toggles system states (Play/Pause double tap).
*   **Left Hand:** Controls timeline navigation (Media Scrubbing) via horizontal delta ($\Delta x$) tracking of the pinch centroid.

### 3. Edge Computing & Privacy
*   **Local Execution:** Fully operational offline. No video frames are transmitted to external servers.
*   **Volatile Processing:** Video frames reside in RAM for ~30ms during processing and are immediately discarded.

---

## Architecture

The system is designed as a Linear Pipeline with a Branched Logic Layer.

1.  **Acquisition Layer:** Captures high-speed video frames via OpenCV.
2.  **Pre-processing Layer:** Executes BGR-to-RGB color space conversions.
3.  **Inference Layer:** Deploys the MediaPipe graph (BlazePalm Detector + Landmark Model) to extract 21 2.5D coordinates per hand.
4.  **Logic Layer (Branched):**
    *   **Right Hand:** Computes Euclidean distances for actuation scalars. Includes a *clutch mechanism* (folded fingers) to prevent false-positive adjustments.
    *   **Left Hand:** Calculates the $\Delta x$ of the hand centroid across sequential frames.
5.  **Actuation Layer:** Interfaces with OS-level APIs (WASAPI via PyCaw for audio, DDC/CI for monitor backlight, and PyAutoGUI for temporal keyboard emulation).

---

## Performance

The system achieves high robustness and low latency on standard consumer CPUs. 

To view the latest diagnostic results, you can run the benchmarking suite included in the `test/` directory. The results are generated as high-resolution PDFs:

*   [System CPU Utilization Analysis](testResult/cpu_utilization.pdf) (Averages ~17.8% overhead)
*   [End-to-End Pipeline Latency](testResult/latency_breakdown.pdf) (Averages ~35ms perception-to-actuation delay)
*   [Mathematical Transfer Function](testResult/transfer_function.pdf) (Mapping pixel distance to percentage scalars)

---

## Installation

### Prerequisites
*   Windows 10/11 (Required for WASAPI audio and DDC/CI brightness control)
*   Python 3.8+
*   Webcam

### Steps

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/adeolaex/Cognitive-Vision-Gesture-Control.git
    cd Cognitive-Vision-Gesture-Control
    ```

2.  **Initialize Virtual Environment:**
    ```bash
    python -m venv .venv
    .venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Launch the Application:**
    ```bash
    python src/main.py
    ```

5.  **Run Diagnostics (Optional):**
    ```bash
    python test/cpu_utilization.py
    python test/latency_breakdown.py
    python test/transfer_function.py
    ```

---

*This project was developed as part of an undergraduate final year project at Nile University of Nigeria.*

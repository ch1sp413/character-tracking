# Offline Character Tracking with YOLOv5 and OpenCV CSRT

This document provides an **end-to-end workflow** for labeling, training, and performing offline detection + tracking of a single videogame character using **YOLOv5** for detection and **OpenCV CSRT** for tracking in **Python 3**. It also references a **Comet ML** integration for experiment logging and metrics tracking.

---

## 1. Project Overview

1. **Goal**  
   Track a single videogame character’s bounding box in recorded gameplay.

2. **Approach**  
   - **Offline detection** in each frame with YOLOv5.  
   - **OpenCV CSRT** tracker used between detections to reduce overhead.  
   - If the tracker fails, YOLO re-detects to reacquire the character.

3. **Environment**  
   - **Python 3** (system-wide, no separate virtualenv).  
   - **OpenCV** (with contrib modules) for tracking.  
   - **PyTorch** for YOLOv5.  
   - **LabelImg** for annotating ~50 screenshots.  
   - **Comet ML** for logging experiment data (e.g., IoU, bounding-box area, confidence, reacquisition count, etc.).

---

## 2. Installation Steps

Below are commands for a macOS or Linux system. Adjust them as needed for Windows.

1. **Python 3**

   Make sure you have **Python 3** installed (e.g., 3.10 or 3.11).  
   ```bash
   python3 --version

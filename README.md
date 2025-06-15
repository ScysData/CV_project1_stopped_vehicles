
# Stopped Vehicle Detection \& Counting

**A Computer Vision Portfolio Project by** **_Samson Chan_**

---

## Overview

This project demonstrates my ability to apply deep learning and computer vision techniques to real-world problems. The goal is to **detect and count stopped vehicles** in video footage using the YOLOv8 object detection framework and the SORT tracking algorithm. This portfolio piece showcases my skills in Python, deep learning, video processing, and practical solution deployment.

---

## Key Features

- **Vehicle Detection:** Uses YOLOv8 for accurate and real-time vehicle detection in video streams.
- **Object Tracking:** Integrates the SORT (Simple Online and Realtime Tracking) algorithm to track individual vehicles across frames.
- **Stopped Vehicle Counting:** Identifies and counts vehicles that have stopped in the frame.

---
## Quick Demo



https://github.com/user-attachments/assets/c6d68144-2409-433e-ab87-59229e049348


---


## Potential Business Use Cases:

- **Temporary Traffic Management for Construction Sites**: This model enables real-time assessment of the impact for temporary traffic management measures, teams can quickly adjust and monitor activities to fine-tune traffic plans.

- **Construction Site Security and Safety**: The model can identify unauthorized vehicles in restricted areas and ensure that vehicles stay within their designated lanes.

- **Emergency Response**: It enables the rapid detection of stopped vehicles or accidents, facilitating swift emergency intervention.

Furthermore, this model can be extended to detect other objects, such as construction defects, passengers, or animals, making it versatile for various business scenarios.


---
## Project Structure

```
CV_project1_stopped_vehicles/
│
├── YOLO_weights/              # Pre-trained YOLO model weights
├── inputs/                    # Input videos for testing
├── outputs/                   # Annotated output videos and results
├── utils/                     # Utility scripts and helper functions
│
├── Car_counting.py            # Main detection and counting script
├── Reading_videos.py          # Video reading and preprocessing
├── CUDA_setup_testing.ipynb   # Notebook for CUDA setup verification
├── requirements.txt           # Python dependencies
├── LICENSE
└── .gitignore
```


---

## How to Run

1. **Clone the repository:**

```bash
git clone https://github.com/ScysData/CV_project1_stopped_vehicles.git
cd CV_project1_stopped_vehicles
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Run the main script:**

```bash
python Car_counting.py
```

Output videos and logs will appear in the `outputs/` directory.



---

## References

- **YOLOv8:**
    - [Object Detection with YOLOv8][^1]
- **SORT:**
    - [SORT GitHub Repository][^2]

---


**References:**
[^1]:  https://yolov8.org
[^2]:  https://github.com/abewley/sort




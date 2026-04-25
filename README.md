# Pose Estimation & Activity Classification

A computer vision pipeline that extracts human pose keypoints from video footage and classifies activities using multiple pose estimation models and machine learning classifiers.

A computer vision pipeline that extracts human body keypoints from videos, normalizes them across multiple pose models, and trains classifiers to recognize activities like cooking, dancing, and gym workouts.

## Overview

The system processes raw video footage by extracting frames, detecting human poses using three different models, and feeding the resulting keypoints into traditional ML classifiers for activity recognition. It supports multi-person detection and produces annotated prediction outputs.

## Features

- **Multi-model pose estimation** — YOLOv8-Pose (17 keypoints), MediaPipe (33 → 17), BlazePose (33 → 17)
- **Multi-person support** — YOLO object detector crops individual persons before pose extraction
- **COCO keypoint normalization** — all models mapped to a consistent 17-keypoint format
- **Multiple classifiers** — Logistic Regression, SVM, Random Forest, XGBoost
- **Evaluation metrics** — accuracy, precision, recall, F1, confusion matrix

## Project Structure

```text
├── Data/
│   ├── Frames/         # Extracted frames from videos (per class)
│   ├── Poses/          # Saved keypoint arrays (.npy) per model
│   ├── Predictions/    # Annotated prediction outputs
│   └── Models/         # Saved trained classifiers (.joblib)
├── Pose_Estimation_End.ipynb   # Main notebook
└── yolov8n-pose.pt             # YOLOv8 pose weights
```

## Supported Activity Classes

- Cooking
- Dancing
- Gym

## Pipeline

1. **Frame extraction** — sample frames from class-labeled videos at a configurable frame rate
2. **Pose extraction** — detect keypoints per person using YOLOv8, MediaPipe, or BlazePose
3. **Keypoint normalization** — map all models to 17 COCO-compatible keypoints
4. **Model training** — train and evaluate ML classifiers on the extracted keypoints
5. **Prediction** — run inference on new footage and save annotated outputs

## Getting Started

### Prerequisites

```bash
pip install ultralytics mediapipe opencv-python scikit-learn xgboost matplotlib seaborn pandas joblib
```

### Run

Open and run `Pose_Estimation_End.ipynb` cell by cell. Set your data directory and video paths at the top of the notebook:

```python
DATA_DIR = "./Data"
video = "./Data/your_video.mp4"
```

## Models

| Pose Model | Keypoints | Notes |
|---|---|---|
| YOLOv8-Pose | 17 (COCO) | Native multi-person support |
| MediaPipe | 33 → 17 | YOLO used for person cropping |
| BlazePose | 33 → 17 | YOLO used for person cropping |

| Classifier | Library |
|---|---|
| Logistic Regression | scikit-learn |
| SVM | scikit-learn |
| Random Forest | scikit-learn |
| XGBoost | xgboost |

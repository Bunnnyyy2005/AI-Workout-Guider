# AI-Powered Biomechanical Posture Tracker & Rep Counter

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8.0-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.0-yellow)

## 📌 Overview
* Engineered time-series AI models (1D CNN, LSTM) to replace hardcoded thresholds for real-time biomechanical posture tracking.
* Pipelined OpenCV, MediaPipe, and TensorFlow to extract 3D skeletal geometry for dynamic workout sequence classification.
* Developed a state-driven UI dashboard featuring live angle calculations and multi-model routing for strict rep counting.

This project goes beyond standard static-image classification by utilizing a **sliding temporal window approach**. Instead of judging single frames, the system evaluates the temporal transition of a movement over a ~1-second sequence, allowing the neural network to implicitly learn correct biomechanics without brittle, hardcoded joint angle rules.

## ✨ Key Features
* **Threshold-Free Form Validation:** Uses sequence-based deep learning to judge movement quality, making it robust to different body proportions and camera angles.
* **Specialized Expert Models:** Routes data to highly focused, exercise-specific models (Squats, Pushups, Curls) for higher accuracy.
* **Live Geometry Tracking:** Calculates and renders dynamic joint angles in real-time on the video feed.
* **Interactive UI Dashboard:** State-driven interface to configure workouts, target reps, and sets dynamically via keyboard controls.
* **Intelligent Rep Counting:** Only increments the rep counter if the deep learning model classifies the entire movement phase as "Good Form".

## 🛠️ Tech Stack
* **Computer Vision:** OpenCV, Google MediaPipe (Pose)
* **Deep Learning:** TensorFlow / Keras (1D CNN, LSTM)
* **Data Processing:** NumPy, Pandas, Scikit-learn
* **Language:** Python

## 📂 Project Structure
```text
posture_rep_counter/
│
├── data/
│   ├── raw_videos/           # Raw .mp4/.MOV training files 
│   └── extracted_features/   # Parsed MediaPipe 3D coordinate CSVs
│
├── models/                   
│   ├── squats_expert.h5      # Trained Squat validation model
│   ├── pushups_expert.h5     # Trained Pushup validation model
│   └── curls_expert.h5       # Trained Bicep Curl validation model
│
├── src/
│   ├── extract_landmarks.py  # Parses raw video into temporal CSV data
│   ├── train_experts.py      # Builds & trains the 1D CNN-LSTM networks
│   └── run_inference.py      # Main application script with UI dashboard
│
├── requirements.txt          
└── README.md

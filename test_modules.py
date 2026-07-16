import cv2
import mediapipe as mp
import tensorflow as tf # Change to 'import torch' if using PyTorch
import numpy as np

print(f"OpenCV Version: {cv2.__version__}")
print(f"MediaPipe Version: {mp.__version__}")
print(f"NumPy Version: {np.__version__}")

# Check if Deep Learning framework detects the CPU/GPU
print(f"TensorFlow Version: {tf.__version__}")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
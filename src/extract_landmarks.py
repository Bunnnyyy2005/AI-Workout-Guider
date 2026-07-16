import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

def extract_landmarks():
    # 1. Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False, 
        min_detection_confidence=0.7, 
        min_tracking_confidence=0.7
    )

    # 2. Define Paths based on your structure
    RAW_DIR = os.path.join("data", "raw_videos")
    EXTRACT_DIR = os.path.join("data", "extracted_features")
    
    exercises = ["squats", "pushups", "curls"]
    forms = ["good", "bad"]

    # 3. Create column names for our CSV (33 landmarks * 4 values each = 132 columns)
    cols = []
    for i in range(33):
        cols.extend([f"x{i}", f"y{i}", f"z{i}", f"v{i}"])

    # 4. Loop through the directory structure
    for exercise in exercises:
        for form in forms:
            input_folder = os.path.join(RAW_DIR, exercise, form)
            output_folder = os.path.join(EXTRACT_DIR, exercise, form)

            # Create the output directory if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)

            # Skip if the input folder doesn't exist yet
            if not os.path.exists(input_folder):
                continue

            # Process every video in the current folder
            for file_name in os.listdir(input_folder):
                if not file_name.lower().endswith(('.mp4', '.mov')):
                    continue # Skip hidden files or non-videos

                video_path = os.path.join(input_folder, file_name)
                csv_name = os.path.splitext(file_name)[0] + ".csv"
                csv_path = os.path.join(output_folder, csv_name)

                # Skip if we already extracted this video (saves time if you restart)
                if os.path.exists(csv_path):
                    print(f"Skipping {file_name} (Already processed)")
                    continue

                print(f"Processing: {exercise}/{form}/{file_name}...")
                
                cap = cv2.VideoCapture(video_path)
                video_data = []

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break # End of video

                    # MediaPipe requires RGB images
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(frame_rgb)

                    # If a person is detected, extract coordinates
                    if results.pose_landmarks:
                        frame_landmarks = []
                        for landmark in results.pose_landmarks.landmark:
                            frame_landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                        video_data.append(frame_landmarks)
                    else:
                        # If no person is found in this frame, append zeros to keep timeline intact
                        video_data.append(list(np.zeros(132)))

                cap.release()

                # Save the video's data to a CSV
                df = pd.DataFrame(video_data, columns=cols)
                df.to_csv(csv_path, index=False)
                print(f"  -> Saved {len(df)} frames to {csv_name}")

    print("\nExtraction Complete! All CSVs are saved in data/extracted_features/")

if __name__ == "__main__":
    extract_landmarks()
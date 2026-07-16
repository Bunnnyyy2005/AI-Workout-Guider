import cv2
import mediapipe as mp
import numpy as np
import math
from tensorflow.keras.models import load_model
from collections import deque

# --- CONFIGURATION & PRELOADING ---
print("Preloading expert models... (This might take a few seconds)")
MODELS = {
    "squats": load_model("models/squats_expert.keras"),
    "pushups": load_model("models/pushups_expert.keras"),
    "curls": load_model("models/curls_expert.keras")
}
print("Models loaded successfully!")

WINDOW_SIZE = 30

# Workout specific tracking data
WORKOUT_CONFIG = {
    "squats":  {"joint_y": 24, "angle_joints": [24, 26, 28]}, # Hip, Knee, Ankle
    "pushups": {"joint_y": 12, "angle_joints": [12, 14, 16]}, # Shoulder, Elbow, Wrist
    "curls":   {"joint_y": 16, "angle_joints": [12, 14, 16]}  # Shoulder, Elbow, Wrist
}

def calculate_angle(a, b, c):
    """Calculates the 2D angle between three MediaPipe landmarks."""
    radians = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
    angle = abs(radians * 180.0 / math.pi)
    if angle > 180.0:
        angle = 360 - angle
    return int(angle)

def draw_dashboard(frame, workout, target_reps, target_sets):
    """Draws the pre-workout UI overlay."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (50, 50), (800, 400), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame) # Translucent background

    cv2.putText(frame, "--- WORKOUT DASHBOARD ---", (70, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Workout: {workout.upper()} (Press 1-squats, 2-pushups, 3-curls)", (70, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Target Reps: {target_reps} (Press w to increase, s to decrease)", (70, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Target Sets: {target_sets} (Press e to increase, d to decrease)", (70, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.putText(frame, "[PRESS SPACE TO START]", (70, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

def main():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    cap = cv2.VideoCapture(0)
    
    # --- APP STATE VARIABLES ---
    app_state = "DASHBOARD"  # Can be "DASHBOARD" or "WORKOUT"
    current_workout = "squats"
    target_reps = 10
    target_sets = 3
    
    # --- WORKOUT VARIABLES ---
    current_set = 1
    rep_count = 0
    movement_stage = "up"
    form_status = "Ready"
    form_color = (0, 255, 255)
    window = deque(maxlen=WINDOW_SIZE)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1) # Mirror feed
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # --- DASHBOARD LOGIC ---
        if app_state == "DASHBOARD":
            draw_dashboard(frame, current_workout, target_reps, target_sets)
            
        # --- WORKOUT LOGIC ---
        elif app_state == "WORKOUT":
            results = pose.process(frame_rgb)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # Extract skeletal data for the AI Model
                frame_landmarks = []
                for lm in landmarks:
                    frame_landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
                window.append(frame_landmarks)
                
                # Real-time Angle Calculation overlay
                angle_pts = WORKOUT_CONFIG[current_workout]["angle_joints"]
                a, b, c = landmarks[angle_pts[0]], landmarks[angle_pts[1]], landmarks[angle_pts[2]]
                angle = calculate_angle(a, b, c)
                
                # Draw the angle on the joint with a dark outline for visibility
                h, w, _ = frame.shape
                joint_px = (int(b.x * w), int(b.y * h))
                cv2.putText(frame, str(angle), joint_px, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)       # Outline
                cv2.putText(frame, str(angle), joint_px, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 100), 2) # Inner text

                # AI Prediction (Only if window is full)
                if len(window) == WINDOW_SIZE:
                    model_input = np.expand_dims(np.array(window), axis=0)
                    prediction = MODELS[current_workout].predict(model_input, verbose=0)[0][0]
                    
                    if prediction > 0.65:
                        form_status, form_color = "BAD FORM", (0, 0, 255)
                    else:
                        form_status, form_color = "GOOD FORM", (0, 255, 0)
                
                # Rep & Set Counting Logic
                joint_y = landmarks[WORKOUT_CONFIG[current_workout]["joint_y"]].y
                if joint_y > 0.7 and movement_stage == "up":
                    movement_stage = "down"
                elif joint_y < 0.5 and movement_stage == "down":
                    movement_stage = "up"
                    if form_status == "GOOD FORM":
                        rep_count += 1
                        
                        # Set completion logic
                        if rep_count >= target_reps:
                            current_set += 1
                            rep_count = 0
                            if current_set > target_sets:
                                app_state = "DASHBOARD" # Workout finished! Return to menu
                                current_set = 1
            else:
                window.append(list(np.zeros(132)))
            
            # --- IMPROVED LIVE WORKOUT HUD ---
            # 1. Draw Translucent Background Panel for text visibility
            hud_overlay = frame.copy()
            cv2.rectangle(hud_overlay, (0, 0), (320, 160), (0, 0, 0), -1)
            cv2.addWeighted(hud_overlay, 0.6, frame, 0.4, 0, frame)

            # 2. Draw HUD Text over the dark panel
            cv2.putText(frame, f"Workout: {current_workout.upper()}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Set: {current_set}/{target_sets} | Reps: {rep_count}/{target_reps}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, form_status, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, form_color, 3)
            
            # Bottom screen pause prompt (with outline for visibility)
            bottom_y = frame.shape[0] - 20
            cv2.putText(frame, "[PRESS SPACE TO PAUSE]", (10, bottom_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3) # Outline
            cv2.putText(frame, "[PRESS SPACE TO PAUSE]", (10, bottom_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("AI Posture Tracker", frame)
        
        # --- KEYBOARD CONTROLS ---
        key = cv2.waitKey(10) & 0xFF
        
        if key == ord('q'): 
            break
        elif key == ord(' '): # Spacebar to toggle state
            app_state = "WORKOUT" if app_state == "DASHBOARD" else "DASHBOARD"
            window.clear() # Reset memory when pausing/starting
            
        if app_state == "DASHBOARD":
            if key == ord('1'): current_workout = "squats"
            elif key == ord('2'): current_workout = "pushups"
            elif key == ord('3'): current_workout = "curls"
            elif key == ord('w'): target_reps += 1
            elif key == ord('s') and target_reps > 1: target_reps -= 1
            elif key == ord('e'): target_sets += 1
            elif key == ord('d') and target_sets > 1: target_sets -= 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
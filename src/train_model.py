import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout

# --- Hyperparameters ---
WINDOW_SIZE = 60  # 60 frames per sequence (~2 seconds of movement)
STRIDE = 15       # Overlap windows by 15 frames to maximize data efficiency
FEATURES = 132    # 33 landmarks * 4 coordinates (x, y, z, visibility)

def load_workout_data(exercise, data_dir="data/extracted_features"):
    """Loads and windows data specifically for a single exercise."""
    X, y = [], []
    forms = {"good": 0, "bad": 1} # Binary mapping: 0 for Good, 1 for Bad

    for form, label_idx in forms.items():
        folder_path = os.path.join(data_dir, exercise, form)
        
        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found {folder_path}")
            continue
            
        for file in os.listdir(folder_path):
            if not file.endswith(".csv"):
                continue
                
            df = pd.read_csv(os.path.join(folder_path, file))
            frames = df.values # Extract underlying numpy array
            
            # Create sliding windows along the timeline of the video
            for start in range(0, len(frames) - WINDOW_SIZE + 1, STRIDE):
                window = frames[start : start + WINDOW_SIZE]
                X.append(window)
                y.append(label_idx)
                
    return np.array(X), np.array(y)

def build_binary_expert_model(input_shape):
    """Builds a highly focused binary classifier for posture verification."""
    model = Sequential([
        # 1D Convolutional layers to capture quick spatial deviations in posture
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        
        # LSTM layer to track the execution flow across the temporal window
        LSTM(64, return_sequences=False),
        Dropout(0.5), 
        
        # Binary output layer (0 = Good Form, 1 = Bad Form)
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid') 
    ])
    
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

def main():
    exercises = ["squats", "pushups", "curls"]
    os.makedirs("models", exist_ok=True)

    for exercise in exercises:
        print(f"\n=========================================")
        print(f"Starting Training for: {exercise.upper()} EXPERT")
        print(f"=========================================")
        
        # 1. Load data for this specific workout
        X, y = load_workout_data(exercise)
        
        if len(X) == 0:
            print(f"Skipping {exercise}: No data sequences found.")
            continue
            
        print(f"Generated {X.shape[0]} sequences for training.")
        
        # 2. Train/Test Split (80% training, 20% validation)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 3. Initialize the model architecture
        model = build_binary_expert_model(input_shape=(WINDOW_SIZE, FEATURES))
        
        # 4. Fit the network weights
        epochs = 25
        batch_size = 32
        
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=1
        )
        
        # 5. Save the unique model
        model_path = os.path.join("models", f"{exercise}_expert.keras")
        model.save(model_path)
        print(f"Successfully saved {exercise.upper()} model to {model_path}!")

if __name__ == "__main__":
    main()
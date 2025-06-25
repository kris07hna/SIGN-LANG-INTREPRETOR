import cv2
import mediapipe as mp
import numpy as np
import os
import time
from sign_utils import MediaPipeLandmarksModel

class DataCollector:
    def __init__(self):
        self.landmark_model = MediaPipeLandmarksModel()
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        
    def collect_sign_data(self, sign_name, num_sequences=30, sequence_length=30):
        """Collect data for a specific sign."""
        
        # Create directory for the sign
        data_dir = f"dataset/{sign_name}"
        os.makedirs(data_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(0)
        
        # Arrays to store sequences
        sequences = []
        
        print(f"Collecting data for sign: {sign_name}")
        print(f"Will collect {num_sequences} sequences of {sequence_length} frames each")
        print("Press 's' to start collecting, 'q' to quit")
        
        collecting = False
        sequence = []
        sequence_count = 0
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Extract landmarks
            landmarks = self.landmark_model.predict_landmarks(frame)
            
            # Draw landmarks
            if landmarks:
                frame = self.landmark_model.draw_landmarks(frame, landmarks)
            
            # Display status
            if collecting:
                cv2.putText(frame, f'Collecting: {sign_name}', (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f'Sequence: {sequence_count + 1}/{num_sequences}', (20, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f'Frame: {frame_count}/{sequence_length}', (20, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if landmarks:
                    # Extract features
                    features = self.landmark_model.extract_features(landmarks)
                    sequence.append(features)
                    frame_count += 1
                    
                    if frame_count >= sequence_length:
                        # Save sequence
                        sequences.append(sequence.copy())
                        sequence = []
                        frame_count = 0
                        sequence_count += 1
                        
                        if sequence_count >= num_sequences:
                            break
                        
                        # Wait before next sequence
                        time.sleep(2)
            else:
                cv2.putText(frame, f'Ready to collect: {sign_name}', (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, "Press 's' to start", (20, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow('Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and not collecting:
                collecting = True
                print("Started collecting...")
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Save collected data
        if sequences:
            np.save(f"{data_dir}/sequences.npy", np.array(sequences))
            print(f"Saved {len(sequences)} sequences for {sign_name}")
        
        return sequences

def main():
    collector = DataCollector()
    
    # List of signs to collect
    signs = ['hello', 'thank_you', 'please', 'sorry', 'yes', 'no', 'good', 'bad', 'help', 'stop']
    
    print("Sign Language Data Collection Tool")
    print("Available signs:", signs)
    
    while True:
        print("\nOptions:")
        print("1. Collect data for a specific sign")
        print("2. Collect data for all signs")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            sign_name = input("Enter sign name: ").lower().replace(' ', '_')
            num_sequences = int(input("Number of sequences (default 30): ") or 30)
            collector.collect_sign_data(sign_name, num_sequences)
            
        elif choice == '2':
            for sign in signs:
                print(f"\nPrepare to collect data for: {sign}")
                input("Press Enter when ready...")
                collector.collect_sign_data(sign, 30)
                
        elif choice == '3':
            break
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()

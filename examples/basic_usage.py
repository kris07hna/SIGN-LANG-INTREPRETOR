"""
Basic usage example of the Sign Language Interpreter system.
This script demonstrates how to:
1. Initialize the models
2. Process video input
3. Get predictions
4. Visualize results
"""

import cv2
import numpy as np
from sign_utils import SignLanguagePredictor, MediaPipeLandmarksModel

def main():
    # Initialize models
    predictor = SignLanguagePredictor(
        model_path="../models/model.tflite",
        labels_path="../models/labels.txt"
    )
    landmark_model = MediaPipeLandmarksModel()
    
    # Open camera
    cap = cv2.VideoCapture(0)
    
    print("Press 'q' to quit")
    print("Press 's' to toggle sign detection")
    
    detect_signs = True
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        if detect_signs:
            # Get landmarks
            landmarks = landmark_model.predict_landmarks(frame)
            
            if landmarks:
                # Draw landmarks
                frame = landmark_model.draw_landmarks(frame, landmarks)
                
                # Get prediction
                prediction, confidence = predictor.predict(frame)
                
                # Draw prediction
                if confidence > 0.5:
                    cv2.putText(
                        frame,
                        f"{prediction} ({confidence:.2f})",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2
                    )
        
        # Show status
        status = "Detection: ON" if detect_signs else "Detection: OFF"
        cv2.putText(
            frame,
            status,
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if detect_signs else (0, 0, 255),
            2
        )
        
        # Display frame
        cv2.imshow("Sign Language Interpreter", frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            detect_signs = not detect_signs
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

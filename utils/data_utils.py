import cv2
import numpy as np
import mediapipe as mp
import os
from typing import List, Tuple

class DataProcessor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )

    def process_video(self, video_path: str) -> List[np.ndarray]:
        """Process video and extract frames."""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frames.append(frame)
            
        cap.release()
        return frames

    def extract_landmarks(self, frame: np.ndarray) -> List[Tuple[float, float, float]]:
        """Extract hand landmarks from a frame."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    landmarks.append((landmark.x, landmark.y, landmark.z))
                    
        return landmarks

    def prepare_dataset(self, dataset_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare dataset from video folders."""
        X, y = [], []
        labels = []
        
        # Process each class folder
        for class_name in os.listdir(dataset_path):
            class_path = os.path.join(dataset_path, class_name)
            if not os.path.isdir(class_path):
                continue
                
            label = len(labels)
            labels.append(class_name)
            
            # Process each video in the class
            for video_file in os.listdir(class_path):
                if not video_file.endswith(('.mp4', '.avi')):
                    continue
                    
                video_path = os.path.join(class_path, video_file)
                frames = self.process_video(video_path)
                
                for frame in frames:
                    landmarks = self.extract_landmarks(frame)
                    if landmarks:
                        X.append(landmarks)
                        y.append(label)
        
        # Save labels
        with open('models/labels.txt', 'w') as f:
            f.write('\n'.join(labels))
            
        return np.array(X), np.array(y)

    def augment_landmarks(self, landmarks: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """Apply data augmentation to landmarks."""
        augmented = []
        
        # Scale
        scale = np.random.uniform(0.8, 1.2)
        scaled = [(x * scale, y * scale, z) for x, y, z in landmarks]
        augmented.append(scaled)
        
        # Rotate
        theta = np.random.uniform(-15, 15)
        c, s = np.cos(theta), np.sin(theta)
        rotated = [(x * c - y * s, x * s + y * c, z) for x, y, z in landmarks]
        augmented.append(rotated)
        
        # Add noise
        noise = np.random.normal(0, 0.01, (len(landmarks), 3))
        noisy = [(x + nx, y + ny, z + nz) for (x, y, z), (nx, ny, nz) in zip(landmarks, noise)]
        augmented.append(noisy)
        
        return augmented

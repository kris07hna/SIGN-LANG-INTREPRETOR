import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional

# Try to import TensorFlow, handle gracefully if not available
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. Some features may be limited.")

class MediaPipeLandmarksModel:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, 
            max_num_hands=2, 
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.sequence = []
        self.sequence_length = 15  # Reduced for faster response

    def predict_landmarks(self, image: np.ndarray) -> Optional[List[Tuple[float, float, float]]]:
        """Extract hand landmarks from image."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        if results.multi_hand_landmarks:
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    landmarks.append((landmark.x, landmark.y, landmark.z))
            return landmarks
        return None

    def draw_landmarks(self, image: np.ndarray, landmarks) -> np.ndarray:
        """Draw hand landmarks on image."""
        if landmarks:
            results = self.hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return image

    def normalize_landmarks(self, landmarks: List[Tuple[float, float, float]]) -> np.ndarray:
        """Normalize landmarks relative to wrist position."""
        if not landmarks:
            return np.zeros((21, 3))
        
        landmarks_array = np.array(landmarks)
        
        # Get wrist position (first landmark)
        wrist = landmarks_array[0]
        
        # Normalize relative to wrist
        normalized = landmarks_array - wrist
        
        # Scale by hand size (distance from wrist to middle finger tip)
        if len(landmarks_array) >= 12:
            hand_size = np.linalg.norm(landmarks_array[12] - wrist)
            if hand_size > 0:
                normalized = normalized / hand_size
        
        return normalized.flatten()

    def extract_features(self, landmarks: List[Tuple[float, float, float]]) -> np.ndarray:
        """Extract features from landmarks for model input."""
        if not landmarks:
            return np.zeros(63)  # 21 landmarks * 3 coordinates
        
        # Ensure we have exactly 21 landmarks (one hand)
        if len(landmarks) > 21:
            landmarks = landmarks[:21]  # Take first hand only
        elif len(landmarks) < 21:
            # Pad with zeros if we have fewer landmarks
            while len(landmarks) < 21:
                landmarks.append((0.0, 0.0, 0.0))
        
        # Convert to numpy array
        landmarks_array = np.array(landmarks)
        
        # Get wrist position (first landmark)
        wrist = landmarks_array[0]
        
        # Normalize relative to wrist
        normalized = landmarks_array - wrist
        
        # Scale by hand size (distance from wrist to middle finger tip)
        hand_size = np.linalg.norm(landmarks_array[12] - wrist) if len(landmarks_array) >= 12 else 1.0
        if hand_size > 0:
            normalized = normalized / hand_size
            
        # Flatten to 63 dimensions (21 landmarks * 3 coordinates)
        features = normalized.flatten()
        
        # Ensure we have exactly 63 dimensions
        if len(features) > 63:
            features = features[:63]
        elif len(features) < 63:
            features = np.pad(features, (0, 63 - len(features)))
            
        return features

    def update_sequence(self, landmarks: List[Tuple[float, float, float]]) -> np.ndarray:
        """Update sequence for temporal modeling."""
        features = self.extract_features(landmarks)
        self.sequence.append(features)
        
        # Keep only last sequence_length frames
        if len(self.sequence) > self.sequence_length:
            self.sequence.pop(0)
        
        # Pad sequence if needed
        if len(self.sequence) < self.sequence_length:
            padding = np.zeros((self.sequence_length - len(self.sequence), features.shape[0]))
            sequence_array = np.vstack([padding, np.array(self.sequence)])
        else:
            sequence_array = np.array(self.sequence)
        
        return sequence_array

    def reset_sequence(self):
        """Reset the sequence buffer."""
        self.sequence = []

class SignLanguagePredictor:
    def __init__(self, model_path: str, labels_path: str):
        self.landmark_model = MediaPipeLandmarksModel()
        
        # Load labels
        try:
            with open(labels_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            print(f"Warning: Labels file not found at {labels_path}")
            self.labels = ["Unknown"]
        
        # Initialize TFLite interpreter if available
        if TF_AVAILABLE:
            try:
                self.interpreter = tf.lite.Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                self.model_loaded = True
            except Exception as e:
                print(f"Warning: Could not load model: {e}")
                self.model_loaded = False
        else:
            print("Warning: TensorFlow not available. Running in landmark-only mode.")
            self.model_loaded = False

    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """Predict sign from image using static prediction for compatibility."""
        # Use static prediction instead of temporal for now to avoid dimension issues
        return self.predict_static(image)

    def predict_static(self, image: np.ndarray) -> Tuple[str, float]:
        """Predict sign from static image (single frame)."""
        landmarks = self.landmark_model.predict_landmarks(image)
        
        if not landmarks:
            return "No hand detected", 0.0
            
        if not self.model_loaded:
            return "Hand detected (Model not available)", 0.5
            
        try:
            # Extract features
            features = self.landmark_model.extract_features(landmarks)
            
            # Get expected input shape from model
            input_shape = self.input_details[0]['shape']
            
            # Reshape features to match expected input shape
            if len(input_shape) == 2:  # If model expects 2D input (batch_size, features)
                input_data = features.reshape(1, -1).astype(np.float32)
            elif len(input_shape) == 3:  # If model expects 3D input (batch_size, time_steps, features)
                input_data = features.reshape(1, 1, -1).astype(np.float32)
            else:
                raise ValueError(f"Unexpected input shape: {input_shape}")
            
            # Run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            
            # Get prediction
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            predicted_class = np.argmax(output[0])
            confidence = float(np.max(output[0]))
            
            return self.labels[predicted_class], confidence
        except Exception as e:
            print(f"Error during static prediction: {e}")
            return "Error during prediction", 0.0

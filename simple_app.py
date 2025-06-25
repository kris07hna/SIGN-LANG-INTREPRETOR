import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from av import VideoFrame
import mediapipe as mp
import tensorflow as tf

# Page config
st.set_page_config(
    page_title="Sign Language Digits Interpreter",
    page_icon="🔢",
    layout="wide"
)

# Load model and labels
@st.cache_resource
def load_model():
    try:
        interpreter = tf.lite.Interpreter(model_path="models/model.tflite")
        interpreter.allocate_tensors()
        
        with open("models/labels.txt", 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        
        return interpreter, labels
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, []

interpreter, labels = load_model()

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_hand_features(landmarks):
    """Extract 63 features from hand landmarks."""
    if not landmarks:
        return np.zeros(63)
    
    # Convert landmarks to numpy array
    landmarks_array = np.array([(lm.x, lm.y, lm.z) for lm in landmarks.landmark])
    
    # Normalize relative to wrist (first landmark)
    wrist = landmarks_array[0]
    normalized = landmarks_array - wrist
    
    # Scale by hand size
    hand_size = np.linalg.norm(landmarks_array[12] - wrist)
    if hand_size > 0:
        normalized = normalized / hand_size
    
    # Flatten to 63 features
    features = normalized.flatten()
    
    # Ensure exactly 63 features
    if len(features) > 63:
        features = features[:63]
    elif len(features) < 63:
        features = np.pad(features, (0, 63 - len(features)))
    
    return features

def predict_digit(features):
    """Predict digit from features."""
    if interpreter is None:
        return "Model not loaded", 0.0
    
    try:
        # Prepare input
        input_data = np.expand_dims(features, axis=0).astype(np.float32)
        
        # Run inference
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # Get prediction
        output = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(output[0])
        confidence = float(np.max(output[0]))
        
        return labels[predicted_class] if predicted_class < len(labels) else str(predicted_class), confidence
    except Exception as e:
        return f"Error: {str(e)}", 0.0

class DigitRecognizer(VideoTransformerBase):
    def __init__(self):
        self.prediction = "Show your hand"
        self.confidence = 0.0
        self.frame_count = 0

    def recv(self, frame: VideoFrame) -> VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        # Process every 5th frame to reduce computation
        if self.frame_count % 5 == 0:
            # Convert to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_image)
            
            if results.multi_hand_landmarks:
                # Get first hand
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract features and predict
                features = extract_hand_features(hand_landmarks)
                self.prediction, self.confidence = predict_digit(features)
            else:
                self.prediction = "No hand detected"
                self.confidence = 0.0
        
        # Add text overlay
        color = (0, 255, 0) if self.confidence > 0.5 else (0, 0, 255)
        cv2.putText(
            image,
            f"{self.prediction} ({self.confidence:.2f})",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2
        )
        
        return VideoFrame.from_ndarray(image, format="bgr24")

# Main app
st.title("🔢 Sign Language Digits Interpreter")
st.markdown("Show hand signs for digits 0-9 to the camera!")

# Sidebar controls
st.sidebar.title("Controls")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

# Create columns
col1, col2 = st.columns([2, 1])

with col1:
    # Webcam component
    ctx = webrtc_streamer(
        key="digit_recognition",
        video_processor_factory=DigitRecognizer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.markdown("### Current Prediction")
    
    if ctx.video_processor:
        # Display current prediction
        prediction_placeholder = st.empty()
        confidence_placeholder = st.empty()
        
        if hasattr(ctx.video_processor, 'prediction'):
            if ctx.video_processor.confidence > confidence_threshold:
                prediction_placeholder.markdown(f"# {ctx.video_processor.prediction}")
            else:
                prediction_placeholder.markdown("# -")
            
            confidence_placeholder.progress(int(ctx.video_processor.confidence * 100))
            st.text(f"Confidence: {ctx.video_processor.confidence:.2f}")
    else:
        st.warning("Turn on your webcam to start!")

# Instructions
st.markdown("---")
st.markdown("""
### Instructions:
1. Allow camera access when prompted
2. Show your hand clearly in the camera view
3. Make digit signs (0-9) with your hand
4. Hold the sign steady for best results
5. Adjust confidence threshold if needed

### Tips:
- Ensure good lighting
- Keep your hand in the center of the frame
- Make clear, distinct digit signs
- Wait a moment between different signs
""")

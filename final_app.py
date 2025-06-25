import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from av import VideoFrame
import mediapipe as mp
import tensorflow as tf
import threading
import time

# Page config
st.set_page_config(
    page_title="Sign Language Digits Interpreter",
    page_icon="🔢",
    layout="wide"
)

# RTC Configuration for better WebRTC performance
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# Global variables for thread-safe prediction sharing
prediction_lock = threading.Lock()
current_prediction = {"digit": "Show your hand", "confidence": 0.0}

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


@st.cache_resource
def get_mediapipe_hands():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    return mp_hands, mp_drawing, hands

mp_hands, mp_drawing, hands = get_mediapipe_hands()

def extract_hand_features(landmarks):
    """Extract 63 correct features from hand landmarks."""
    try:
        # Convert landmarks to numpy array
        landmarks_array = np.array([(lm.x, lm.y, lm.z) for lm in landmarks.landmark])
        
        # Ensure we have exactly 21 landmarks
        if len(landmarks_array) != 21:
            return np.zeros(63)
        
        # Normalize relative to wrist (first landmark)
        wrist = landmarks_array[0]
        normalized = landmarks_array - wrist
        
        # Scale by hand size (distance from wrist to middle finger tip)
        hand_size = np.linalg.norm(landmarks_array[12] - wrist)
        if hand_size > 0:
            normalized = normalized / hand_size
        
        # Flatten to 63 features
        features = normalized.flatten()
        
        # Ensure exactly 63 features
        if len(features) != 63:
            features = np.pad(features, (0, max(0, 63 - len(features))))[:63]
        
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return np.zeros(63)

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
        
        # Apply softmax to get better confidence scores
        exp_output = np.exp(output[0] - np.max(output[0]))
        softmax_output = exp_output / np.sum(exp_output)
        confidence = float(softmax_output[predicted_class])
        
        digit = labels[predicted_class] if predicted_class < len(labels) else str(predicted_class)
        return digit, confidence
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error", 0.0

class DigitRecognizer(VideoTransformerBase):
    def __init__(self):
        self.frame_count = 0
        self.last_prediction_time = 0
        self.prediction_interval = 0.2  # Predict every 200ms

    def recv(self, frame: VideoFrame) -> VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        current_time = time.time()
        
        # Process prediction at regular intervals
        if current_time - self.last_prediction_time > self.prediction_interval:
            self.last_prediction_time = current_time
            
            try:
                # Convert to RGB for MediaPipe
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_image)
                
                if results.multi_hand_landmarks:
                    # Get first hand
                    hand_landmarks = results.multi_hand_landmarks[0]
                    
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                    )
                    
                    # Extract features and predict
                    features = extract_hand_features(hand_landmarks)
                    digit, confidence = predict_digit(features)
                    
                    # Update global prediction
                    with prediction_lock:
                        current_prediction["digit"] = digit
                        current_prediction["confidence"] = confidence
                else:
                    with prediction_lock:
                        current_prediction["digit"] = "No hand detected"
                        current_prediction["confidence"] = 0.0
            except Exception as e:
                print(f"Processing error: {e}")
                with prediction_lock:
                    current_prediction["digit"] = "Processing error"
                    current_prediction["confidence"] = 0.0
        
        # Add text overlay
        with prediction_lock:
            display_text = current_prediction["digit"]
            confidence = current_prediction["confidence"]
        
        color = (0, 255, 0) if confidence > 0.5 else (0, 0, 255)
        cv2.putText(
            image,
            f"{display_text} ({confidence:.2f})",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2
        )
        
        return VideoFrame.from_ndarray(image, format="bgr24")

# Main app
st.title("🤟 AI Sign Language Interpreter")
st.markdown("Show hand signs for numbers and letters to the camera!")

# Sidebar controls
st.sidebar.title("🎛️ Controls")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.1)
st.sidebar.markdown("---")
st.sidebar.markdown("### Model Info")
if interpreter:
    st.sidebar.success("✅ Model loaded successfully")
    st.sidebar.info(f"📊 Classes: {len(labels)}")
    st.sidebar.info("🔢 Numbers: 1,2,4,5,6,7,8")
    st.sidebar.info("🔤 Letters: A-X (except E)")
else:
    st.sidebar.error("❌ Model not loaded")

# Create columns
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📹 Live Camera Feed")
    # Webcam component
    ctx = webrtc_streamer(
        key="digit_recognition",
        video_processor_factory=DigitRecognizer,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.markdown("### 🎯 Current Prediction")
    
    # Create placeholders for dynamic updates
    prediction_placeholder = st.empty()
    confidence_bar = st.empty()
    confidence_text = st.empty()
    
    # Update display in real-time
    if ctx.state.playing:
        while ctx.state.playing:
            with prediction_lock:
                digit = current_prediction["digit"]
                confidence = current_prediction["confidence"]
            
            # Display prediction
            if confidence > confidence_threshold and digit not in ["No hand detected", "Processing error", "Error"]:
                prediction_placeholder.markdown(f"# {digit}")
                confidence_bar.progress(confidence)
                confidence_text.success(f"Confidence: {confidence:.2f}")
            else:
                prediction_placeholder.markdown("# -")
                confidence_bar.progress(0)
                if digit == "No hand detected":
                    confidence_text.info("Show your hand to the camera")
                else:
                    confidence_text.warning(f"Low confidence: {confidence:.2f}")
            
            time.sleep(0.1)  # Update every 100ms
    else:
        st.warning("📷 Turn on your webcam to start!")

# Instructions
st.markdown("---")
st.markdown("""
### 📋 Instructions:
1. **Allow camera access** when prompted
2. **Position your hand** clearly in the camera view
3. **Make gesture signs** for numbers or letters
4. **Hold the sign steady** for best results
5. **Adjust confidence threshold** if needed

### 💡 Tips for Better Recognition:
- ✅ Ensure **good lighting**
- ✅ Keep your **hand centered** in the frame
- ✅ Make **clear, distinct** gesture signs
- ✅ **Hold steady** for 1-2 seconds
- ✅ Try different **hand orientations** if needed
- ✅ **Avoid background clutter** behind your hand

### 🎯 Supported Gestures:
""")

# Display supported gestures
st.markdown("#### 🔢 Numbers:")
number_labels = [label for label in labels if label.isdigit()]
if number_labels:
    cols = st.columns(min(len(number_labels), 8))
    for i, label in enumerate(number_labels):
        with cols[i % len(cols)]:
            st.info(f"**{label}**")

st.markdown("#### 🔤 Letters:")
letter_labels = [label for label in labels if label.isalpha()]
if letter_labels:
    # Display letters in rows of 8
    for row_start in range(0, len(letter_labels), 8):
        row_labels = letter_labels[row_start:row_start + 8]
        cols = st.columns(len(row_labels))
        for i, label in enumerate(row_labels):
            with cols[i]:
                st.info(f"**{label}**")

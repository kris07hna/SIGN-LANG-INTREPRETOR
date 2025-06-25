import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import threading
import time
import google.generativeai as genai
from av import VideoFrame

# Page config
st.set_page_config(
    page_title="AI Sign Language Interpreter",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .prediction-box {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .interpretation-box {
        background-color: #f0f7ff;
        padding: 2rem;
        border-radius: 10px;
        border: 1px solid #cce5ff;
        margin-bottom: 1rem;
    }
    .stButton>button {
        background-color: #4c51bf;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #434190;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Gemini AI
@st.cache_resource
def initialize_gemini():
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel('gemini-pro')
        return model
    except Exception as e:
        st.error(f"Error initializing Gemini AI: {e}")
        return None

# RTC Configuration
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# Global variables for thread-safe prediction sharing
prediction_lock = threading.Lock()
current_prediction = {
    "gesture": "Show your hand",
    "confidence": 0.0,
    "interpretation": ""
}

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
gemini_model = initialize_gemini()

# MediaPipe setup
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
    """Extract features from hand landmarks."""
    try:
        landmarks_array = np.array([(lm.x, lm.y, lm.z) for lm in landmarks.landmark])
        if len(landmarks_array) != 21:
            return np.zeros(63)
        
        wrist = landmarks_array[0]
        normalized = landmarks_array - wrist
        hand_size = np.linalg.norm(landmarks_array[12] - wrist)
        
        if hand_size > 0:
            normalized = normalized / hand_size
        
        features = normalized.flatten()
        if len(features) != 63:
            features = np.pad(features, (0, max(0, 63 - len(features))))[:63]
        
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return np.zeros(63)

def predict_gesture(features):
    """Predict gesture from features."""
    if interpreter is None:
        return "Model not loaded", 0.0
    
    try:
        input_data = np.expand_dims(features, axis=0).astype(np.float32)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        output = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(output[0])
        
        exp_output = np.exp(output[0] - np.max(output[0]))
        softmax_output = exp_output / np.sum(exp_output)
        confidence = float(softmax_output[predicted_class])
        
        gesture = labels[predicted_class] if predicted_class < len(labels) else str(predicted_class)
        return gesture, confidence
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error", 0.0

def get_gesture_interpretation(gesture_sequence):
    """Get interpretation from Gemini AI."""
    if not gemini_model or not gesture_sequence:
        return ""
    
    try:
        prompt = f"""
        As a sign language interpreter, analyze this sequence of gestures and provide a natural interpretation:
        Gestures detected: {', '.join(gesture_sequence)}
        
        Provide a clear and concise interpretation. If the gestures form a word or phrase, explain its meaning.
        If they're individual letters/numbers, format them clearly.
        """
        
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini AI error: {e}")
        return ""

class GestureRecognizer(VideoTransformerBase):
    def __init__(self):
        self.frame_count = 0
        self.last_prediction_time = 0
        self.prediction_interval = 0.2
        self.gesture_history = []
        self.last_interpretation_time = 0
        self.interpretation_interval = 2.0

    def recv(self, frame: VideoFrame) -> VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_prediction_time > self.prediction_interval:
            self.last_prediction_time = current_time
            
            try:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_image)
                
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                    )
                    
                    features = extract_hand_features(hand_landmarks)
                    gesture, confidence = predict_gesture(features)
                    
                    if confidence > 0.5:
                        self.gesture_history.append(gesture)
                        self.gesture_history = self.gesture_history[-10:]  # Keep last 10 gestures
                    
                    with prediction_lock:
                        current_prediction["gesture"] = gesture
                        current_prediction["confidence"] = confidence
                        
                        # Update interpretation periodically
                        if current_time - self.last_interpretation_time > self.interpretation_interval:
                            self.last_interpretation_time = current_time
                            interpretation = get_gesture_interpretation(self.gesture_history)
                            if interpretation:
                                current_prediction["interpretation"] = interpretation
                else:
                    with prediction_lock:
                        current_prediction["gesture"] = "No hand detected"
                        current_prediction["confidence"] = 0.0
            except Exception as e:
                print(f"Processing error: {e}")
                with prediction_lock:
                    current_prediction["gesture"] = "Processing error"
                    current_prediction["confidence"] = 0.0
        
        # Add text overlay
        with prediction_lock:
            display_text = current_prediction["gesture"]
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

def main():
    st.title("🤟 AI Sign Language Interpreter")
    st.markdown("Transform sign language into text with real-time AI interpretation")
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📹 Live Camera Feed")
        ctx = webrtc_streamer(
            key="gesture_recognition",
            video_processor_factory=GestureRecognizer,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    
    with col2:
        st.markdown("### 🎯 Recognition Results")
        
        # Create placeholders for dynamic updates
        prediction_placeholder = st.empty()
        confidence_bar = st.empty()
        interpretation_placeholder = st.empty()
        
        if ctx.state.playing:
            while ctx.state.playing:
                with prediction_lock:
                    gesture = current_prediction["gesture"]
                    confidence = current_prediction["confidence"]
                    interpretation = current_prediction["interpretation"]
                
                # Display current gesture
                with prediction_placeholder.container():
                    st.markdown("""
                    <div class="prediction-box">
                        <h3 style="font-size: 1.5rem; margin-bottom: 0.5rem;">Current Gesture</h3>
                        <p style="font-size: 2rem; font-weight: bold; color: #4c51bf;">{}</p>
                    </div>
                    """.format(gesture), unsafe_allow_html=True)
                
                # Display confidence bar
                if confidence > 0.5 and gesture not in ["No hand detected", "Processing error", "Error"]:
                    confidence_bar.progress(confidence)
                    
                    # Display interpretation
                    if interpretation:
                        with interpretation_placeholder.container():
                            st.markdown("""
                            <div class="interpretation-box">
                                <h3 style="font-size: 1.5rem; margin-bottom: 0.5rem;">AI Interpretation</h3>
                                <p style="font-size: 1.2rem;">{}</p>
                            </div>
                            """.format(interpretation), unsafe_allow_html=True)
                else:
                    confidence_bar.progress(0)
                    if gesture == "No hand detected":
                        interpretation_placeholder.info("Show your hand to begin")
                    else:
                        interpretation_placeholder.warning("Gesture not recognized clearly")
                
                time.sleep(0.1)
        else:
            st.warning("📷 Turn on your webcam to start!")
    
    # Instructions and supported gestures
    st.markdown("---")
    
    # Display supported gestures in an organized grid
    st.markdown("### 🎯 Supported Gestures")
    
    # Numbers
    st.markdown("#### 🔢 Numbers")
    number_labels = [label for label in labels if label.isdigit()]
    if number_labels:
        cols = st.columns(len(number_labels))
        for i, label in enumerate(number_labels):
            with cols[i]:
                st.markdown(f"""
                <div style="text-align: center; padding: 1rem; background-color: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <h3 style="font-size: 1.5rem; color: #4c51bf;">{label}</h3>
                </div>
                """, unsafe_allow_html=True)
    
    # Letters
    st.markdown("#### 🔤 Letters")
    letter_labels = [label for label in labels if label.isalpha()]
    if letter_labels:
        # Display letters in rows of 8
        for i in range(0, len(letter_labels), 8):
            row_labels = letter_labels[i:i+8]
            cols = st.columns(len(row_labels))
            for j, label in enumerate(row_labels):
                with cols[j]:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; background-color: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        <h3 style="font-size: 1.5rem; color: #4c51bf;">{label}</h3>
                    </div>
                    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

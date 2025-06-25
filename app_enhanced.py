import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import mediapipe as mp
import google.generativeai as genai
from collections import deque
import time
import json

# Configure page
st.set_page_config(
    page_title="🤟 AI Sign Language Interpreter",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        color: #1a1a1a;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.2rem;
        margin: 1rem 0;
    }
    .confidence-meter {
        background: #e9ecef;
        border-radius: 10px;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
    .sentence-builder {
        background: #ffffff;
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        min-height: 100px;
        color: #1a1a1a;
        font-weight: 600;
        font-size: 1.1rem;
    }
    .control-panel {
        background: #ffffff;
        border: 2px solid #dee2e6;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: #1a1a1a;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
        st.error(f"Failed to initialize Gemini AI: {e}")
        return None

# Load model and labels
@st.cache_resource
def load_model_and_labels():
    try:
        interpreter = tf.lite.Interpreter(model_path="models/model.tflite")
        interpreter.allocate_tensors()
        
        with open("models/labels.txt", "r") as f:
            labels = [line.strip() for line in f.readlines()]
        
        return interpreter, labels
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, []

# Initialize MediaPipe
@st.cache_resource
def initialize_mediapipe():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils
    return hands, mp_drawing, mp_hands

class EnhancedSignInterpreter(VideoTransformerBase):
    def __init__(self, confidence_threshold=0.7, sequence_length=10):
        self.interpreter, self.labels = load_model_and_labels()
        self.hands, self.mp_drawing, self.mp_hands = initialize_mediapipe()
        self.gemini_model = initialize_gemini()
        
        # Enhanced features
        self.confidence_threshold = confidence_threshold
        self.sequence_length = sequence_length
        self.gesture_sequence = deque(maxlen=sequence_length)
        self.last_prediction = ""
        self.last_confidence = 0.0
        self.sentence_builder = []
        self.last_gesture_time = time.time()
        self.gesture_hold_time = 1.5  # seconds to hold gesture for confirmation
        
        # Prediction smoothing
        self.prediction_buffer = deque(maxlen=5)
        
    def extract_landmarks(self, image):
        """Extract hand landmarks using MediaPipe"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            # Pad or truncate to 63 features (21 landmarks * 3 coordinates)
            if len(landmarks) < 63:
                landmarks.extend([0.0] * (63 - len(landmarks)))
            else:
                landmarks = landmarks[:63]
            
            return np.array(landmarks, dtype=np.float32)
        return None
    
    def predict_gesture(self, landmarks):
        """Predict gesture from landmarks"""
        if self.interpreter is None or landmarks is None:
            return "No gesture", 0.0
        
        try:
            input_details = self.interpreter.get_input_details()
            output_details = self.interpreter.get_output_details()
            
            # Reshape landmarks for model input
            input_data = landmarks.reshape(1, -1)
            
            self.interpreter.set_tensor(input_details[0]['index'], input_data)
            self.interpreter.invoke()
            
            output_data = self.interpreter.get_tensor(output_details[0]['index'])
            predicted_class = np.argmax(output_data)
            confidence = float(np.max(output_data))
            
            if predicted_class < len(self.labels):
                return self.labels[predicted_class], confidence
            else:
                return "Unknown", confidence
                
        except Exception as e:
            return f"Error: {str(e)}", 0.0
    
    def smooth_prediction(self, prediction, confidence):
        """Smooth predictions using buffer"""
        self.prediction_buffer.append((prediction, confidence))
        
        # Get most common prediction in buffer
        predictions = [p[0] for p in self.prediction_buffer]
        confidences = [p[1] for p in self.prediction_buffer]
        
        # Find most frequent prediction
        unique_predictions = list(set(predictions))
        if unique_predictions:
            most_common = max(unique_predictions, key=predictions.count)
            avg_confidence = np.mean([c for p, c in self.prediction_buffer if p == most_common])
            return most_common, avg_confidence
        
        return prediction, confidence
    
    def update_sentence_builder(self, gesture, confidence):
        """Update sentence builder with confirmed gestures"""
        current_time = time.time()
        cooldown_period = 0.5  # Time to wait before accepting new gestures
        
        # Check if the gesture is valid and meets confidence threshold
        is_valid_gesture = (gesture not in ["No gesture", "Unknown"] and 
                          confidence > self.confidence_threshold)
        
        # Time elapsed since last gesture
        time_elapsed = current_time - self.last_gesture_time
        
        if is_valid_gesture:
            # Case 1: New gesture detected
            if gesture != self.last_prediction:
                self.last_prediction = gesture
                self.last_confidence = confidence
                self.last_gesture_time = current_time
                
            # Case 2: Gesture held long enough and cooldown period passed
            elif (time_elapsed > self.gesture_hold_time and 
                  (not self.sentence_builder or 
                   time_elapsed > self.gesture_hold_time + cooldown_period)):
                if not self.sentence_builder or self.sentence_builder[-1] != gesture:
                    self.sentence_builder.append(gesture)
                    self.last_gesture_time = current_time
                    self.last_prediction = ""  # Reset prediction to allow new gestures
        else:
            # Reset prediction if gesture is invalid or confidence is low
            if time_elapsed > cooldown_period:
                self.last_prediction = ""
    
    def generate_ai_interpretation(self, gestures):
        """Generate AI interpretation using Gemini"""
        if not gestures or not self.gemini_model:
            return "No interpretation available"
        
        try:
            gesture_text = " ".join(gestures)
            prompt = f"""
            Convert these sign language gestures into a natural, meaningful sentence: {gesture_text}
            
            Guidelines:
            - Form complete, grammatically correct sentences
            - Add appropriate articles, prepositions, and conjunctions
            - Consider context and common sign language patterns
            - If it's a sequence of letters, try to form words
            - If it's numbers, consider if they represent quantities, dates, or codes
            - Make the output conversational and natural
            
            Gestures: {gesture_text}
            
            Natural interpretation:
            """
            
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"AI interpretation error: {str(e)}"
    
    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        
        # Extract landmarks
        landmarks = self.extract_landmarks(image)
        
        if landmarks is not None:
            # Predict gesture
            gesture, confidence = self.predict_gesture(landmarks)
            
            # Smooth prediction
            gesture, confidence = self.smooth_prediction(gesture, confidence)
            
            # Update sentence builder
            self.update_sentence_builder(gesture, confidence)
            
            # Draw hand landmarks
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # Draw prediction info
            color = (0, 255, 0) if confidence > self.confidence_threshold else (0, 165, 255)
            cv2.putText(image, f"{gesture} ({confidence:.2f})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Store current prediction for UI
            st.session_state.current_gesture = gesture
            st.session_state.current_confidence = confidence
            st.session_state.sentence_builder = self.sentence_builder.copy()
        else:
            st.session_state.current_gesture = "No hand detected"
            st.session_state.current_confidence = 0.0
        
        return av.VideoFrame.from_ndarray(image, format="bgr24")

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤟 Enhanced AI Sign Language Interpreter</h1>
        <p>Advanced real-time interpretation with AI-powered sentence formation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'current_gesture' not in st.session_state:
        st.session_state.current_gesture = "Ready to start"
        st.session_state.current_confidence = 0.0
        st.session_state.sentence_builder = []
        st.session_state.ai_interpretation = ""
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("### 🎛️ Control Panel")
        
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.7, 
            step=0.1,
            help="Minimum confidence required for gesture recognition"
        )
        
        sequence_length = st.slider(
            "Sequence Memory", 
            min_value=5, 
            max_value=20, 
            value=10,
            help="Number of gestures to remember for sentence building"
        )
        
        gesture_hold_time = st.slider(
            "Gesture Hold Time (seconds)", 
            min_value=0.5, 
            max_value=3.0, 
            value=1.5, 
            step=0.1,
            help="Time to hold gesture for confirmation"
        )
        
        st.markdown("### 🎯 Quick Actions")
        if st.button("🗑️ Clear Sentence", use_container_width=True):
            st.session_state.sentence_builder = []
            st.session_state.ai_interpretation = ""
        
        if st.button("🤖 Generate AI Interpretation", use_container_width=True):
            if st.session_state.sentence_builder:
                interpreter = EnhancedSignInterpreter()
                st.session_state.ai_interpretation = interpreter.generate_ai_interpretation(
                    st.session_state.sentence_builder
                )
        
        # Display supported gestures
        st.markdown("### 📋 Supported Gestures")
        st.markdown("**Numbers:** 1, 2, 4, 5, 6, 7, 8")
        st.markdown("**Letters:** A-X (except E)")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📹 Live Camera Feed")
        
        # WebRTC configuration
        rtc_configuration = RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        })
        
        # Video streamer
        webrtc_ctx = webrtc_streamer(
            key="enhanced-sign-interpreter",
            video_transformer_factory=lambda: EnhancedSignInterpreter(
                confidence_threshold=confidence_threshold,
                sequence_length=sequence_length
            ),
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    
    with col2:
        st.markdown("### 🎯 Recognition Results")
        
        # Current prediction
        st.markdown(f"""
        <div class="prediction-box">
            <h3>{st.session_state.current_gesture}</h3>
            <div class="confidence-meter">
                <div style="background: {'#28a745' if st.session_state.current_confidence > confidence_threshold else '#ffc107'}; 
                           width: {st.session_state.current_confidence * 100}%; 
                           height: 20px; 
                           border-radius: 10px; 
                           transition: all 0.3s ease;">
                </div>
                <small>Confidence: {st.session_state.current_confidence:.1%}</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Sentence builder
        st.markdown("### 📝 Sentence Builder")
        sentence_text = " → ".join(st.session_state.sentence_builder) if st.session_state.sentence_builder else "Start making gestures..."
        
        st.markdown(f"""
        <div class="sentence-builder">
            <strong>Gestures:</strong><br>
            {sentence_text}
        </div>
        """, unsafe_allow_html=True)
        
        # AI Interpretation
        if st.session_state.ai_interpretation:
            st.markdown("### 🤖 AI Interpretation")
            st.markdown(f"""
            <div class="feature-card">
                <strong>Natural Language:</strong><br>
                {st.session_state.ai_interpretation}
            </div>
            """, unsafe_allow_html=True)
    
    # Statistics and tips
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>💡 Tips for Better Recognition</h4>
            <ul>
                <li>Ensure good lighting</li>
                <li>Keep hand centered in frame</li>
                <li>Hold gestures steady</li>
                <li>Avoid cluttered backgrounds</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="feature-card">
            <h4>📊 Session Statistics</h4>
            <ul>
                <li>Gestures Captured: {len(st.session_state.sentence_builder)}</li>
                <li>Current Confidence: {st.session_state.current_confidence:.1%}</li>
                <li>Threshold: {confidence_threshold:.1%}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h4>🚀 Enhanced Features</h4>
            <ul>
                <li>Confidence threshold control</li>
                <li>Gesture sequence memory</li>
                <li>AI sentence formation</li>
                <li>Real-time smoothing</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

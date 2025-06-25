import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
from av import VideoFrame
from sign_utils import SignLanguagePredictor, MediaPipeLandmarksModel

# Page config
st.set_page_config(
    page_title="Sign Language Interpreter",
    page_icon="🤟",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .prediction-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
    }
    .instruction-box {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize models
@st.cache_resource
def load_models():
    try:
        predictor = SignLanguagePredictor(
            model_path="models/model.tflite",
            labels_path="models/labels.txt"
        )
        return predictor
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

predictor = load_models()

class SignInterpreter(VideoTransformerBase):
    def __init__(self, confidence_threshold=0.7):
        self.predictor = predictor
        self.landmark_model = MediaPipeLandmarksModel()
        self.result_text = "Show your hand sign"
        self.confidence = 0.0
        self.landmarks = None
        self.prediction_counter = {}
        self.last_prediction = None
        self.confidence_threshold = confidence_threshold
        self.frame_count = 0
        self.prediction_history = []
        self.stable_prediction = None
        self.stable_confidence = 0.0
        
        # Enhanced prediction smoothing
        self.prediction_buffer = []
        self.buffer_size = 8  # Reduced for faster response
        self.min_consistent_frames = 3  # Reduced for faster response

    def recv(self, frame: VideoFrame) -> VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        if self.predictor:
            # Get landmarks
            self.landmarks = self.landmark_model.predict_landmarks(image)
            
            # Get prediction if hand is detected
            if self.landmarks:
                try:
                    # Use temporal prediction instead of static
                    prediction, raw_confidence = self.predictor.predict(image)
                    
                    # Add to prediction buffer with higher threshold for confidence
                    if raw_confidence > 0.4:  # Only add predictions with decent confidence
                        self.prediction_buffer.append((prediction, raw_confidence))
                        if len(self.prediction_buffer) > self.buffer_size:
                            self.prediction_buffer.pop(0)
                    
                    # Get smoothed prediction
                    smoothed_prediction, smoothed_confidence = self._get_smoothed_prediction()
                    
                    # Update current values
                    self.confidence = smoothed_confidence
                    
                    # Check if prediction meets confidence threshold
                    if smoothed_confidence > self.confidence_threshold:
                        # Count consecutive predictions
                        if smoothed_prediction not in self.prediction_counter:
                            self.prediction_counter = {smoothed_prediction: 1}
                        else:
                            self.prediction_counter[smoothed_prediction] += 1
                        
                        # Update stable prediction if we have enough consistent frames
                        max_pred = max(self.prediction_counter.items(), key=lambda x: x[1])
                        if max_pred[1] >= self.min_consistent_frames:
                            self.stable_prediction = smoothed_prediction
                            self.stable_confidence = smoothed_confidence
                            self.result_text = f"Sign: {smoothed_prediction}"
                            self.last_prediction = smoothed_prediction
                    else:
                        # Reset if confidence is too low
                        if self.frame_count % 30 == 0:  # Reset every 30 frames
                            self.prediction_counter = {}
                            self.result_text = "Show your hand sign"
                            
                except Exception as e:
                    self.result_text = "Processing..."
                    self.confidence = 0.0
                    self.prediction_counter = {}
            else:
                self.result_text = "No hand detected"
                self.confidence = 0.0
                self.prediction_counter = {}
                self.prediction_buffer = []
            
            # Draw landmarks with better visualization
            if self.landmarks:
                image = self.landmark_model.draw_landmarks(image, self.landmarks)
            
            # Enhanced text overlay with better positioning
            self._draw_prediction_overlay(image)
        
        return VideoFrame.from_ndarray(image, format="bgr24")
    
    def _get_smoothed_prediction(self):
        """Get smoothed prediction from buffer"""
        if not self.prediction_buffer:
            return "No prediction", 0.0
        
        # Get predictions from last few frames
        recent_predictions = self.prediction_buffer[-5:] if len(self.prediction_buffer) >= 5 else self.prediction_buffer
        
        # Count occurrences of each prediction
        prediction_counts = {}
        confidence_sums = {}
        
        for pred, conf in recent_predictions:
            if pred not in prediction_counts:
                prediction_counts[pred] = 0
                confidence_sums[pred] = 0.0
            prediction_counts[pred] += 1
            confidence_sums[pred] += conf
        
        # Get most frequent prediction
        if prediction_counts:
            most_frequent = max(prediction_counts.items(), key=lambda x: x[1])
            avg_confidence = confidence_sums[most_frequent[0]] / prediction_counts[most_frequent[0]]
            return most_frequent[0], avg_confidence
        
        return "No prediction", 0.0
    
    def _draw_prediction_overlay(self, image):
        """Draw enhanced prediction overlay"""
        height, width = image.shape[:2]
        
        # Background rectangle for better text visibility
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (width-10, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Main prediction text
        color = (0, 255, 0) if self.confidence > self.confidence_threshold else (0, 165, 255)
        cv2.putText(
            image,
            self.result_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2
        )
        
        # Confidence text
        cv2.putText(
            image,
            f"Confidence: {self.confidence:.2f}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )
        
        # Threshold indicator
        threshold_color = (0, 255, 0) if self.confidence > self.confidence_threshold else (0, 0, 255)
        cv2.putText(
            image,
            f"Threshold: {self.confidence_threshold:.2f}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            threshold_color,
            1
        )

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤟 Sign Language Interpreter</h1>
        <p>Real-time sign language recognition using computer vision</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar controls
    st.sidebar.title("⚙️ Controls")
    confidence_threshold = st.sidebar.slider(
        "Detection Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.5,  # Lowered for better live detection
        help="Adjust the confidence threshold for sign detection"
    )

    # Create two columns for layout
    col1, col2 = st.columns([2, 1])

    with col1:
        # Webcam component
        st.markdown("### 📹 Camera Feed")
        ctx = webrtc_streamer(
            key="sign_language",
            video_processor_factory=lambda: SignInterpreter(confidence_threshold),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with col2:
        # Current prediction display
        st.markdown("### 🎯 Current Sign")
        prediction_placeholder = st.empty()
        confidence_meter = st.progress(0)
        
        # Update prediction and confidence
        if ctx.video_transformer:
            if ctx.video_transformer.last_prediction and ctx.video_transformer.confidence > confidence_threshold:
                prediction_placeholder.markdown(f"""
                <div class="prediction-box">
                    <h2>{ctx.video_transformer.last_prediction}</h2>
                </div>
                """, unsafe_allow_html=True)
            else:
                prediction_placeholder.markdown("""
                <div class="prediction-box">
                    <h2>-</h2>
                </div>
                """, unsafe_allow_html=True)
            
            confidence_meter.progress(int(ctx.video_transformer.confidence * 100))
        else:
            st.info("📷 Start the webcam to begin recognition")

    # Instructions
    st.markdown("---")
    
    # Create three columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📝 Instructions")
        st.markdown("""
        1. Allow camera access when prompted
        2. Position your hand clearly in the camera view
        3. Make hand signs for:
           - **Numbers:** 1, 2, 4, 5, 6, 7, 8
           - **Letters:** A through X (except E)
        4. Hold each sign steady for best recognition
        5. Adjust the confidence threshold if needed
        """)
    
    with col2:
        st.markdown("### 💡 Tips for Better Recognition")
        st.markdown("""
        - Ensure good lighting on your hands
        - Keep your hand within the camera frame
        - Make clear, distinct gestures
        - Avoid rapid movements
        - Use a plain background
        - Keep hand at comfortable distance from camera
        """)
    
    with col3:
        st.markdown("### 📊 Supported Gestures")
        st.markdown("**Numbers:**")
        st.markdown("1, 2, 4, 5, 6, 7, 8")
        st.markdown("**Letters:**")
        st.markdown("A, B, C, D, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X")

if __name__ == "__main__":
    main()

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
from av import VideoFrame
import os
from sign_utils import SignLanguagePredictor, MediaPipeLandmarksModel

# Page config
st.set_page_config(
    page_title="Sign Language Digits Interpreter",
    page_icon="🔢",
    layout="wide"
)

# Sidebar
st.sidebar.title("🔢 Controls")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)
show_landmarks = st.sidebar.checkbox("Show Hand Landmarks", True)

# Main content
st.title("Sign Language Digits Interpreter")
st.markdown("""
This app recognizes hand signs for digits (0-9) in real-time. 
Show your hand signs to the camera and see the interpretations!

Instructions:
1. Allow camera access when prompted
2. Show your hand sign for a digit (0-9)
3. Hold the sign steady for best results
4. Adjust the confidence threshold if needed
""")

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

class DigitInterpreter(VideoTransformerBase):
    def __init__(self):
        self.predictor = predictor
        self.landmark_model = MediaPipeLandmarksModel()
        self.result_text = "Show your hand sign"
        self.confidence = 0.0
        self.landmarks = None
        self.frame_count = 0
        self.last_prediction = None
        self.prediction_counter = {}

    def recv(self, frame: VideoFrame) -> VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        if self.predictor:
            # Get landmarks
            self.landmarks = self.landmark_model.predict_landmarks(image)
            
            # Get prediction
            if self.landmarks:
                try:
                    # Use static prediction mode
                    prediction, self.confidence = self.predictor.predict_static(image)
                    
                    # Only update prediction if confidence is above threshold
                    if self.confidence > confidence_threshold:
                        # Count predictions to reduce flickering
                        if prediction not in self.prediction_counter:
                            self.prediction_counter = {prediction: 1}
                        else:
                            self.prediction_counter[prediction] += 1
                        
                        # Update result if we have consistent predictions
                        max_pred = max(self.prediction_counter.items(), key=lambda x: x[1])
                        if max_pred[1] >= 3:  # Require 3 consistent predictions
                            self.result_text = f"Digit: {prediction}"
                            self.last_prediction = prediction
                    else:
                        self.result_text = "Show your hand sign"
                        self.prediction_counter = {}
                except Exception as e:
                    self.result_text = "Processing..."
                    self.confidence = 0.0
                    self.prediction_counter = {}
            else:
                self.result_text = "No hand detected"
                self.confidence = 0.0
                self.prediction_counter = {}
            
            # Draw landmarks if enabled
            if show_landmarks and self.landmarks:
                image = self.landmark_model.draw_landmarks(image, self.landmarks)
            
            # Add text overlay
            cv2.putText(
                image,
                f"{self.result_text} ({self.confidence:.2f})",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0) if self.confidence > confidence_threshold else (0, 0, 255),
                2
            )
        
        return VideoFrame.from_ndarray(image, format="bgr24")

# Create two columns
col1, col2 = st.columns([2, 1])

with col1:
    # Webcam component
    ctx = webrtc_streamer(
        key="digit_recognition",
        video_processor_factory=DigitInterpreter,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    # Interpretation display
    st.markdown("### 🔢 Current Digit")
    if ctx.video_transformer:
        # Large digit display
        digit_display = st.empty()
        confidence_meter = st.progress(0)
        
        # Update interpretation and confidence
        if ctx.video_transformer.last_prediction is not None and ctx.video_transformer.confidence > confidence_threshold:
            digit_display.markdown(f"# {ctx.video_transformer.last_prediction}")
        else:
            digit_display.markdown("# -")
        confidence_meter.progress(int(ctx.video_transformer.confidence * 100))
        
        # Show confidence value
        st.text(f"Confidence: {ctx.video_transformer.confidence:.2f}")
    else:
        st.warning("📷 Turn on your webcam to start!")

# Instructions
st.markdown("---")
st.markdown("""
### How to Use
1. Make sure you're in a well-lit environment
2. Position your hand clearly in the camera view
3. Make hand signs for digits 0-9
4. Hold each sign steady for accurate recognition
5. Adjust the confidence threshold if needed:
   - Higher threshold = more accurate but less sensitive
   - Lower threshold = more sensitive but may have false positives
""")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using MediaPipe, TensorFlow, and Streamlit")

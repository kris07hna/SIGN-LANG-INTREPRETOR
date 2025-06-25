# 🤟  Sign Language Interpreter

A real-time sign language recognition system built with Python, MediaPipe, TensorFlow, and Streamlit. This application can detect and interpret sign language gestures in real-time using computer vision and machine learning.

## 🌟 Features

- **Real-time Hand Detection**: Uses MediaPipe for accurate hand landmark detection
- **Sign Language Recognition**: Supports 29 gestures including:
  - Numbers: 1, 2, 4, 5, 6, 7, 8
  - Letters: A-X (excluding E)
- **Live Webcam Processing**: Real-time gesture recognition through webcam
- **Interactive Web Interface**: Built with Streamlit for easy use
- **Adjustable Confidence**: Configurable detection confidence threshold
- **Optimized Performance**: Fast response with temporal prediction smoothing

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Webcam
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ai-sign-language-interpreter.git
cd ai-sign-language-interpreter
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run simple_sign_interpreter.py
```

4. **Open your browser** and navigate to `http://localhost:8501`

## 🎯 Usage

1. **Start the application** using the command above
2. **Allow camera access** when prompted by your browser
3. **Click "START"** to begin webcam capture
4. **Show hand gestures** to the camera
5. **Adjust confidence threshold** using the sidebar slider for better detection
6. **View real-time predictions** in the "Current Sign" panel

## 📁 Project Structure

```
ai-sign-language-interpreter/
├── simple_sign_interpreter.py    # Main Streamlit application
├── sign_utils.py                 # Core prediction and landmark detection
├── models/
│   ├── sign_model.tflite        # Trained TensorFlow Lite model
│   └── labels.txt               # Class labels for gestures
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore file
└── README.md                   # Project documentation
```

## 🔧 Technical Details

### Architecture
- **Frontend**: Streamlit web interface
- **Computer Vision**: MediaPipe for hand landmark detection
- **Machine Learning**: TensorFlow Lite for gesture classification
- **Real-time Processing**: Optimized prediction pipeline with temporal smoothing

### Model Information
- **Input**: 63-dimensional hand landmark features (21 landmarks × 3 coordinates)
- **Output**: 29 sign language gesture classes
- **Format**: TensorFlow Lite (.tflite) for optimized inference
- **Performance**: Real-time processing at 30+ FPS

### Key Components

1. **MediaPipeLandmarksModel**: Handles hand detection and landmark extraction
2. **SignLanguagePredictor**: Manages model inference and prediction smoothing
3. **Streamlit Interface**: Provides user-friendly web interface

## 🎛️ Configuration

### Adjustable Parameters
- **Detection Confidence**: Minimum confidence for hand detection (default: 0.5)
- **Tracking Confidence**: Minimum confidence for hand tracking (default: 0.5)
- **Prediction Threshold**: Minimum confidence for gesture prediction (default: 0.5)
- **Buffer Size**: Number of frames for prediction smoothing (default: 8)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MediaPipe** for hand landmark detection
- **TensorFlow** for machine learning framework
- **Streamlit** for the web interface
- **OpenCV** for computer vision utilities

## 📞 Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/kris07hna/SIGN-LANG-INTREPRETOR/issues) page
2. Create a new issue with detailed description
3. Include system information and error messages

## 🔮 Future Enhancements

- [ ] Support for more sign language gestures
- [ ] Multi-hand gesture recognition
- [ ] Sign language sentence formation
- [ ] Mobile app version
- [ ] Real-time translation to multiple languages

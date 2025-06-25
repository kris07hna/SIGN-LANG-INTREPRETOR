# 🤟 AI Sign Language Interpreter

A comprehensive, real-time sign language interpretation system powered by AI. This application uses computer vision and machine learning to recognize and interpret sign language gestures, providing instant translations with natural language understanding through Google's Gemini AI.

## ✨ Features

- 🎯 Real-time sign language recognition
- 🤖 AI-powered gesture interpretation
- 📹 Live webcam processing
- 🔤 Support for letters (A-X) and numbers (1-8)
- 💡 Natural language interpretation using Gemini AI
- 🎨 Beautiful, responsive user interface
- 📱 Mobile-friendly design

## 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements_new.txt
   ```

2. **Set Up Gemini API**
   - Create `.streamlit/secrets.toml` if not exists
   - Add your Gemini API key:
     ```toml
     GEMINI_API_KEY = "your-api-key-here"
     ```

3. **Run the Application**
   ```bash
   python run_app.py
   ```
   This will start both the landing page and the Streamlit application.
   - Landing page: http://localhost:5000
   - Streamlit app: http://localhost:8501

## 📖 Usage

1. Visit the landing page at http://localhost:5000
2. Click "Try it Out" to launch the interpreter
3. Allow camera access when prompted
4. Position your hand clearly in the camera view
5. Make sign language gestures
6. View real-time interpretations and AI-enhanced understanding

## 🎯 Supported Gestures

### Numbers
- 1, 2, 4, 5, 6, 7, 8

### Letters
- A through X (except E)

## 💡 Tips for Best Results

- Ensure good lighting conditions
- Keep your hand centered in the frame
- Make clear, distinct gestures
- Hold gestures steady for 1-2 seconds
- Maintain consistent hand orientation
- Avoid cluttered backgrounds

## 🛠️ Technical Details

### Components

- **Frontend**
  - Landing page: HTML/CSS with Tailwind CSS
  - Application UI: Streamlit
  - Real-time video: streamlit-webrtc

- **Backend**
  - Web Server: Flask
  - ML Framework: TensorFlow Lite
  - Hand Tracking: MediaPipe
  - AI Interpretation: Google Gemini

### Model Architecture

- Custom TensorFlow model trained on hand gesture dataset
- MediaPipe hand landmark detection
- 63-dimensional feature vector for gesture recognition
- Real-time processing optimizations

## 📊 Performance

- Real-time processing at 30+ FPS
- Low latency gesture recognition
- High accuracy on supported gestures
- Efficient memory usage

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- MediaPipe team for hand tracking technology
- Google for Gemini AI API
- Streamlit team for the amazing framework
- Open source community for various tools and libraries

## 📞 Support

For issues, questions, or contributions, please open an issue on GitHub.

---

Made with ❤️ for breaking communication barriers

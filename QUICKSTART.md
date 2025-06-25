# 🚀 Quick Start Guide - Sign Language Interpreter

This guide will help you get started with the Sign Language Interpreter system quickly.

## 📋 Prerequisites

- Python 3.8 or higher
- Webcam
- Git (optional)

## 🛠️ Installation

1. **Setup Environment**
```bash
# Clone repository (if using git)
git clone https://github.com/yourusername/sign-language-interpreter.git
cd sign-language-interpreter

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Run setup script
python setup.py
```

## 🎯 Using the Application

### 1. Quick Demo
To try the system with a pre-trained model:
```bash
streamlit run app.py
```
This will launch the web interface where you can:
- Adjust confidence threshold
- Toggle hand landmark visualization
- See real-time sign language interpretation

### 2. Collect Your Own Data
To create a custom dataset:
```bash
python data_collection.py
```
Follow the on-screen instructions to:
- Choose signs to collect
- Record sequences of hand gestures
- Save data for training

### 3. Train Custom Model
To train a model on your collected data:
```bash
python train_model.py
```
This will:
- Load your collected dataset
- Train a new model
- Save the trained model
- Generate performance metrics

## 🔍 Troubleshooting

1. **Camera not working?**
   - Check if another application is using the camera
   - Ensure camera permissions are granted
   - Try closing and reopening the application

2. **Poor recognition?**
   - Ensure good lighting
   - Keep hands within camera frame
   - Try adjusting the confidence threshold
   - Collect more training data

3. **Installation issues?**
   - Check Python version: `python --version`
   - Update pip: `python -m pip install --upgrade pip`
   - Install requirements manually: `pip install -r requirements.txt`

## 📚 Additional Resources

- Check `README.md` for detailed documentation
- View example signs in `examples/` directory
- See training tips in `docs/training_guide.md`

## 🆘 Need Help?

- Check the issues section on GitHub
- Join our Discord community
- Email support at: support@signlanguage.ai

## 🔄 Updates

To update the application:
```bash
git pull  # If using git
pip install -r requirements.txt  # Update dependencies
```

## 📝 Notes

- The system works best in well-lit environments
- Position yourself 2-3 feet from the camera
- Keep your hand gestures clear and deliberate
- Practice with the sample signs first

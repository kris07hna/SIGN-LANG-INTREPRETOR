# 📦 Installation Guide - Sign Language Interpreter

This guide provides detailed installation instructions for the Sign Language Interpreter system.

## 🔧 System Requirements

- **Python**: 3.8 - 3.11 (recommended: 3.9 or 3.10)
- **Operating System**: Windows 10/11, macOS 10.14+, or Linux
- **Hardware**: 
  - Webcam (built-in or external)
  - 4GB RAM minimum (8GB recommended)
  - 2GB free disk space
- **Internet**: Required for initial setup and package installation

## 🚀 Quick Installation

### Option 1: Automated Setup (Recommended)

1. **Download and Extract**
   ```bash
   # If using git
   git clone https://github.com/yourusername/sign-language-interpreter.git
   cd sign-language-interpreter
   
   # Or download and extract ZIP file
   ```

2. **Run Setup Script**
   ```bash
   python setup.py
   ```
   
   This will:
   - Create necessary directories
   - Install required packages
   - Create a sample model
   - Verify installation

### Option 2: Manual Installation

1. **Create Virtual Environment**
   ```bash
   python -m venv venv
   
   # Activate virtual environment
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create Directories**
   ```bash
   mkdir models dataset logs examples
   ```

4. **Create Sample Model**
   ```bash
   python create_simple_model.py
   ```

## 🔍 Verification

Run the system test to verify everything is working:

```bash
python test_system.py
```

Expected output:
```
🤟 Sign Language Interpreter - System Test
==================================================

Dependencies:
✅ All dependencies installed

Camera:
✅ Camera working

MediaPipe:
✅ MediaPipe working

Model Files:
✅ Model files found

Predictor:
✅ Predictor working - Sample prediction: hello (0.85)

==================================================
Test Results: 5/5 tests passed
🎉 All tests passed! System is ready to use.
```

## 🚀 Running the Application

### Web Interface (Recommended)
```bash
streamlit run app.py
```

### Command Line Interface
```bash
python examples/basic_usage.py
```

## 🛠️ Troubleshooting

### Common Issues

1. **Python Version Error**
   ```
   Error: This package requires Python 3.8-3.11
   ```
   **Solution**: Install a compatible Python version

2. **Camera Access Error**
   ```
   Error: Camera not accessible
   ```
   **Solutions**:
   - Close other applications using the camera
   - Check camera permissions
   - Try a different camera index

3. **Package Installation Error**
   ```
   Error: Could not find a version that satisfies the requirement
   ```
   **Solutions**:
   - Update pip: `python -m pip install --upgrade pip`
   - Try installing packages individually
   - Check Python version compatibility

4. **TensorFlow Installation Issues**
   ```
   Error: No matching distribution found for tensorflow
   ```
   **Solutions**:
   - Use Python 3.8-3.11
   - Install TensorFlow separately: `pip install tensorflow`
   - Use CPU-only version: `pip install tensorflow-cpu`

### Platform-Specific Issues

#### Windows
- **Visual C++ Error**: Install Microsoft Visual C++ Redistributable
- **Long Path Error**: Enable long path support in Windows settings

#### macOS
- **Permission Error**: Grant camera access in System Preferences
- **Homebrew Issues**: Update Homebrew and Python

#### Linux
- **Camera Error**: Install v4l-utils: `sudo apt install v4l-utils`
- **Permission Error**: Add user to video group: `sudo usermod -a -G video $USER`

## 🔄 Updates

To update the system:

1. **Update Code**
   ```bash
   git pull  # If using git
   ```

2. **Update Dependencies**
   ```bash
   pip install -r requirements.txt --upgrade
   ```

3. **Test System**
   ```bash
   python test_system.py
   ```

## 🆘 Getting Help

If you encounter issues:

1. **Check Documentation**
   - Read `README.md` for overview
   - Check `QUICKSTART.md` for basic usage
   - Review this installation guide

2. **Run Diagnostics**
   ```bash
   python test_system.py --verbose
   ```

3. **Community Support**
   - GitHub Issues: Report bugs and feature requests
   - Discord: Join our community chat
   - Email: support@signlanguage.ai

## 📋 Next Steps

After successful installation:

1. **Try the Demo**: `streamlit run app.py`
2. **Collect Data**: `python data_collection.py`
3. **Train Model**: `python train_model.py`
4. **Read Documentation**: Check `docs/` folder for guides

## 🔐 Security Notes

- The application only accesses your camera locally
- No data is sent to external servers
- All processing happens on your device
- You can review the source code for transparency

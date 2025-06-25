import cv2
import numpy as np
import os
import sys
from sign_utils import MediaPipeLandmarksModel, SignLanguagePredictor

def test_camera():
    """Test if camera is working."""
    print("Testing camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Camera not accessible")
        return False
    
    ret, frame = cap.read()
    if not ret:
        print("❌ Cannot read from camera")
        cap.release()
        return False
    
    print("✅ Camera working")
    cap.release()
    return True

def test_mediapipe():
    """Test MediaPipe hand detection."""
    print("Testing MediaPipe...")
    try:
        landmark_model = MediaPipeLandmarksModel()
        
        # Create a dummy image
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        landmarks = landmark_model.predict_landmarks(dummy_image)
        
        print("✅ MediaPipe working")
        return True
    except Exception as e:
        print(f"❌ MediaPipe error: {e}")
        return False

def test_model_files():
    """Test if model files exist."""
    print("Testing model files...")
    
    model_path = "models/model.tflite"
    labels_path = "models/labels.txt"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    if not os.path.exists(labels_path):
        print(f"❌ Labels file not found: {labels_path}")
        return False
    
    print("✅ Model files found")
    return True

def test_predictor():
    """Test the sign language predictor."""
    print("Testing predictor...")
    
    if not test_model_files():
        return False
    
    try:
        predictor = SignLanguagePredictor(
            model_path="models/model.tflite",
            labels_path="models/labels.txt"
        )
        
        # Create a dummy image
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        prediction, confidence = predictor.predict(dummy_image)
        
        print(f"✅ Predictor working - Sample prediction: {prediction} ({confidence:.2f})")
        return True
    except Exception as e:
        print(f"❌ Predictor error: {e}")
        return False

def test_dependencies():
    """Test if all dependencies are installed."""
    print("Testing dependencies...")
    
    required_modules = [
        'cv2', 'numpy', 'tensorflow', 'mediapipe', 
        'streamlit', 'sklearn'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"❌ Missing modules: {missing_modules}")
        return False
    
    print("✅ All dependencies installed")
    return True

def run_full_test():
    """Run complete system test."""
    print("🤟 Sign Language Interpreter - System Test")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Camera", test_camera),
        ("MediaPipe", test_mediapipe),
        ("Model Files", test_model_files),
        ("Predictor", test_predictor)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"  Fix this issue before proceeding")
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Run 'streamlit run app.py' to start the application")
        print("2. Or run 'python data_collection.py' to collect custom data")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        
        if not test_model_files():
            print("\n💡 Tip: Run 'python create_sample_model.py' to create a sample model")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Quick test - just check if basic components work
        if test_dependencies() and test_mediapipe():
            print("✅ Quick test passed!")
        else:
            print("❌ Quick test failed!")
    else:
        run_full_test()

if __name__ == "__main__":
    main()

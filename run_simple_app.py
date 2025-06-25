#!/usr/bin/env python3
"""
Simple startup script for the streamlined Sign Language Interpreter
"""

import subprocess
import sys
import os
import time

def check_requirements():
    """Check if required files exist"""
    required_files = ['simple_sign_interpreter.py', 'sign_utils.py', 'models/model.tflite', 'models/labels.txt']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all required files are present.")
        return False
    
    print("✅ All required files found.")
    return True

def start_streamlit():
    """Start the simple sign interpreter app"""
    print("🤟 Starting Simple Sign Language Interpreter...")
    print("=" * 50)
    
    if not check_requirements():
        return
    
    try:
        print("🚀 Starting Streamlit app on http://localhost:8501")
        print("📖 Instructions:")
        print("   1. Allow camera access when prompted")
        print("   2. Make sign language gestures for numbers (1,2,4,5,6,7,8) or letters (A-X)")
        print("   3. Hold gestures steady for best recognition")
        print("   4. Adjust confidence threshold if needed")
        print()
        print("Press Ctrl+C to stop the application")
        print("=" * 50)
        
        # Start Streamlit
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'simple_sign_interpreter.py',
            '--server.port=8501',
            '--server.headless=false'
        ])
        
    except KeyboardInterrupt:
        print("\n🛑 Application stopped.")
        print("👋 Thank you for using the Sign Language Interpreter!")
    except Exception as e:
        print(f"❌ Error starting application: {e}")

if __name__ == "__main__":
    start_streamlit()

#!/usr/bin/env python3
"""
Startup script for AI Sign Language Interpreter
This script starts both the landing page (Flask) and the main app (Streamlit)
"""

import subprocess
import threading
import time
import webbrowser
import sys
import os

def start_streamlit():
    """Start Streamlit app"""
    print("🚀 Starting Streamlit app...")
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'app_enhanced.py',
            '--server.port=8502',
            '--server.headless=true',
            '--server.runOnSave=true'
        ])
    except KeyboardInterrupt:
        print("Streamlit app stopped.")
    except Exception as e:
        print(f"Error starting Streamlit: {e}")

def start_flask():
    """Start Flask server for landing page"""
    print("🌐 Starting Flask server...")
    try:
        subprocess.run([sys.executable, 'server.py'])
    except KeyboardInterrupt:
        print("Flask server stopped.")
    except Exception as e:
        print(f"Error starting Flask: {e}")

def main():
    print("=" * 60)
    print("🤟 AI Sign Language Interpreter")
    print("=" * 60)
    print("Starting application servers...")
    print()
    
    # Check if required files exist
    required_files = ['app_enhanced.py', 'server.py', 'models/model.tflite', 'models/labels.txt']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all required files are present.")
        return
    
    print("✅ All required files found.")
    print()
    
    try:
        # Start Streamlit in a separate thread
        streamlit_thread = threading.Thread(target=start_streamlit, daemon=True)
        streamlit_thread.start()
        
        # Give Streamlit time to start
        print("⏳ Waiting for Streamlit to initialize...")
        time.sleep(5)
        
        # Start Flask (this will block)
        print("🌐 Flask server starting on http://localhost:5000")
        print("🤖 Streamlit app available on http://localhost:8501")
        print()
        print("📖 Instructions:")
        print("   1. Visit http://localhost:5000 for the landing page")
        print("   2. Click 'Try it Out' to access the AI interpreter")
        print("   3. Allow camera access when prompted")
        print("   4. Start making sign language gestures!")
        print()
        print("Press Ctrl+C to stop all servers")
        print("=" * 60)
        
        # Open browser automatically
        time.sleep(2)
        webbrowser.open('http://localhost:5000')
        
        # Start Flask server
        start_flask()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down servers...")
        print("👋 Thank you for using AI Sign Language Interpreter!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

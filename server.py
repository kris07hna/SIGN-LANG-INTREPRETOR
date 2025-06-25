from flask import Flask, render_template, redirect, url_for
import subprocess
import threading
import time
import os

app = Flask(__name__)

# Global variable to track Streamlit process
streamlit_process = None

def start_streamlit():
    """Start Streamlit app in background"""
    global streamlit_process
    try:
        streamlit_process = subprocess.Popen([
            'streamlit', 'run', 'app_enhanced.py', 
            '--server.port=8502',
            '--server.headless=true'
        ])
        print("Streamlit app started on port 8502")
    except Exception as e:
        print(f"Error starting Streamlit: {e}")

@app.route('/')
def landing_page():
    """Serve the landing page"""
    return render_template('index.html')

@app.route('/app')
def launch_app():
    """Redirect to Streamlit app"""
    return redirect('http://localhost:8502')

@app.route('/start-streamlit')
def start_streamlit_endpoint():
    """Start Streamlit if not running"""
    global streamlit_process
    if streamlit_process is None or streamlit_process.poll() is not None:
        threading.Thread(target=start_streamlit, daemon=True).start()
        time.sleep(3)  # Give Streamlit time to start
    return redirect('http://localhost:8502')

if __name__ == '__main__':
    # Start Streamlit in background
    threading.Thread(target=start_streamlit, daemon=True).start()
    
    # Start Flask server
    print("Starting Flask server on http://localhost:5000")
    print("Streamlit app will be available on http://localhost:8502")
    app.run(host='0.0.0.0', port=5000, debug=True)

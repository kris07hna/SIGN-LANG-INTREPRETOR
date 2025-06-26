FROM python:3.10-slim

WORKDIR /code

# Install system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y libgl1 ffmpeg

# Copy all project files into the container
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 7860

CMD ["streamlit", "run", "simple_sign_interpreter.py", "--server.port=7860", "--server.headless=true", "--server.enableCORS=false"]

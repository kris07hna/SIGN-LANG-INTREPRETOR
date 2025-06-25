import os
import subprocess
import sys

def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    return True

def create_directories():
    """Create necessary directories."""
    directories = [
        "models",
        "dataset", 
        "static/css",
        "templates",
        "utils",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def create_sample_model():
    """Create a sample model for testing."""
    print("Creating sample model...")
    try:
        subprocess.check_call([sys.executable, "create_sample_model.py"])
        print("✅ Sample model created successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error creating sample model: {e}")
        return False
    return True

def main():
    print("🤟 Sign Language Interpreter Setup")
    print("=" * 40)
    
    # Create directories
    print("\n1. Creating directories...")
    create_directories()
    
    # Install requirements
    print("\n2. Installing requirements...")
    if not install_requirements():
        print("Setup failed. Please install requirements manually.")
        return
    
    # Create sample model
    print("\n3. Creating sample model...")
    if not create_sample_model():
        print("Warning: Sample model creation failed. You can create it later.")
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run 'python data_collection.py' to collect your own sign data")
    print("2. Run 'python train_model.py' to train a custom model")
    print("3. Run 'streamlit run app.py' to start the application")

if __name__ == "__main__":
    main()

import tensorflow as tf
import numpy as np
from utils.model_utils import create_model, convert_to_tflite

def create_sample_model():
    """Create a sample model for demonstration purposes."""
    print("Creating sample sign language model...")
    
    # Model parameters
    num_classes = 10  # Number of signs in labels.txt
    sequence_length = 30
    num_features = 126  # 63 landmarks + 63 velocity features
    
    # Create model
    model = create_model(num_classes, sequence_length, num_features)
    
    # Generate dummy training data
    X_train = np.random.random((1000, sequence_length, num_features)).astype(np.float32)
    y_train = np.random.randint(0, num_classes, (1000,))
    
    # Generate dummy validation data
    X_val = np.random.random((200, sequence_length, num_features)).astype(np.float32)
    y_val = np.random.randint(0, num_classes, (200,))
    
    print("Training sample model...")
    # Train for a few epochs
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=32,
        verbose=1
    )
    
    # Save the model
    model.save('models/sign_language_model.h5')
    print("Model saved as 'models/sign_language_model.h5'")
    
    # Convert to TFLite
    print("Converting to TensorFlow Lite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open('models/model.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print("TFLite model saved as 'models/model.tflite'")
    print("Sample model creation completed!")

if __name__ == "__main__":
    create_sample_model()

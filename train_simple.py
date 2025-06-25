import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import cv2
import os
import glob
from sign_utils import MediaPipeLandmarksModel

def load_and_process_dataset():
    """Load digit images and extract landmarks."""
    landmark_model = MediaPipeLandmarksModel()
    
    X_landmarks = []
    y = []
    
    print("Loading and processing images...")
    
    # Process each digit folder
    for digit in range(10):
        digit_path = os.path.join("dataset", str(digit))
        if not os.path.exists(digit_path):
            continue
            
        image_files = glob.glob(os.path.join(digit_path, "*.JPG"))
        print(f"Processing {len(image_files)} images for digit {digit}")
        
        count = 0
        for img_file in image_files:
            # Load image
            img = cv2.imread(img_file)
            if img is None:
                continue
                
            # Extract landmarks
            landmarks = landmark_model.predict_landmarks(img)
            
            if landmarks is not None:
                # Normalize landmarks
                landmarks_normalized = landmark_model.normalize_landmarks(landmarks)
                
                X_landmarks.append(landmarks_normalized)
                y.append(digit)
                count += 1
                
                # Limit to 100 images per digit for faster training
                if count >= 100:
                    break
        
        print(f"Successfully processed {count} images for digit {digit}")
    
    print(f"Total processed: {len(X_landmarks)} images with landmarks")
    
    # Check if we have any data
    if len(X_landmarks) == 0:
        return np.array([]), np.array([])
    
    # Ensure all landmark arrays have the same shape
    landmark_shape = X_landmarks[0].shape[0]
    filtered_landmarks = []
    filtered_y = []
    
    for i, landmarks in enumerate(X_landmarks):
        if landmarks.shape[0] == landmark_shape:
            filtered_landmarks.append(landmarks)
            filtered_y.append(y[i])
    
    print(f"After filtering: {len(filtered_landmarks)} valid samples")
    
    return np.array(filtered_landmarks), np.array(filtered_y)

def create_simple_model(num_classes=10):
    """Create a simple neural network for landmark classification."""
    model = Sequential([
        Dense(128, activation='relu', input_shape=(63,)),  # 21 landmarks * 3 coordinates
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        Dropout(0.2),
        
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    print("Simple Sign Language Digits Training")
    print("=" * 40)
    
    # Load dataset
    X, y = load_and_process_dataset()
    
    if len(X) == 0:
        print("No valid images found with hand landmarks!")
        return
    
    # Split data (simple split without sklearn)
    n_samples = len(X)
    n_train = int(0.8 * n_samples)
    
    # Shuffle data
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Create model
    model = create_simple_model()
    print("\nModel Summary:")
    model.summary()
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=30,
        batch_size=32,
        verbose=1
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Save model
    print("\nSaving model...")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save Keras model
    model.save('models/digit_model.h5')
    
    # Convert to TFLite
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        with open('models/model.tflite', 'wb') as f:
            f.write(tflite_model)
            
        print("Model converted to TFLite successfully!")
        
    except Exception as e:
        print(f"TFLite conversion failed: {e}")
    
    # Save labels
    labels = [str(i) for i in range(10)]
    with open('models/labels.txt', 'w') as f:
        f.write('\n'.join(labels))
    
    print("Training completed successfully!")
    print("You can now run the main application: streamlit run app.py")

if __name__ == "__main__":
    main()

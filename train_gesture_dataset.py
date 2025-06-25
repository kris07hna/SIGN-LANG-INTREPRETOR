import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import pickle
from datetime import datetime

class GestureDatasetTrainer:
    def __init__(self, dataset_path, output_dir="models"):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize data storage
        self.features = []
        self.labels = []
        self.class_names = []
        self.label_encoder = LabelEncoder()
        
    def extract_hand_landmarks(self, image_path):
        """Extract hand landmarks from image."""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return None
                
            # Convert to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.hands.process(rgb_image)
            
            if results.multi_hand_landmarks:
                # Get first hand landmarks
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Convert to numpy array
                landmarks_array = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])
                
                # Normalize relative to wrist (first landmark)
                wrist = landmarks_array[0]
                normalized = landmarks_array - wrist
                
                # Scale by hand size (distance from wrist to middle finger tip)
                hand_size = np.linalg.norm(landmarks_array[12] - wrist)
                if hand_size > 0:
                    normalized = normalized / hand_size
                
                # Flatten to 63 features (21 landmarks * 3 coordinates)
                features = normalized.flatten()
                
                # Ensure exactly 63 features
                if len(features) != 63:
                    features = np.pad(features, (0, max(0, 63 - len(features))))[:63]
                
                return features
            else:
                return None
                
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def load_dataset(self, max_samples_per_class=500, sample_ratio=0.1):
        """Load and process the dataset."""
        print("Loading dataset...")
        
        # Get all class directories
        class_dirs = [d for d in os.listdir(self.dataset_path) 
                     if os.path.isdir(os.path.join(self.dataset_path, d))]
        class_dirs.sort()  # Ensure consistent ordering
        
        self.class_names = class_dirs
        print(f"Found {len(class_dirs)} classes: {class_dirs}")
        
        # Process each class
        for class_name in tqdm(class_dirs, desc="Processing classes"):
            class_path = os.path.join(self.dataset_path, class_name)
            
            # Get all image files
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Sample images if there are too many
            if len(image_files) > max_samples_per_class:
                # Take every nth image to get a good distribution
                step = len(image_files) // max_samples_per_class
                image_files = image_files[::step][:max_samples_per_class]
            else:
                # If fewer images, sample a percentage
                sample_size = max(1, int(len(image_files) * sample_ratio))
                image_files = image_files[:sample_size]
            
            print(f"Processing {len(image_files)} images for class '{class_name}'")
            
            # Process images
            class_features = []
            for image_file in tqdm(image_files, desc=f"Class {class_name}", leave=False):
                image_path = os.path.join(class_path, image_file)
                features = self.extract_hand_landmarks(image_path)
                
                if features is not None:
                    class_features.append(features)
            
            # Add to dataset
            if class_features:
                self.features.extend(class_features)
                self.labels.extend([class_name] * len(class_features))
                print(f"Added {len(class_features)} samples for class '{class_name}'")
            else:
                print(f"Warning: No valid samples found for class '{class_name}'")
        
        # Convert to numpy arrays
        self.features = np.array(self.features)
        self.labels = np.array(self.labels)
        
        # Encode labels
        self.labels_encoded = self.label_encoder.fit_transform(self.labels)
        
        print(f"\nDataset loaded successfully!")
        print(f"Total samples: {len(self.features)}")
        print(f"Feature shape: {self.features.shape}")
        print(f"Number of classes: {len(self.class_names)}")
        
        # Print class distribution
        unique, counts = np.unique(self.labels, return_counts=True)
        print("\nClass distribution:")
        for class_name, count in zip(unique, counts):
            print(f"  {class_name}: {count} samples")
    
    def create_model(self, input_shape=63, num_classes=36):
        """Create the neural network model with improved architecture."""
        model = keras.Sequential([
            # Input layer with data augmentation
            keras.layers.Dense(512, activation='relu', input_shape=(input_shape,)),
            keras.layers.Dropout(0.4),
            keras.layers.BatchNormalization(),
            
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.BatchNormalization(),
            
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.BatchNormalization(),
            
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.BatchNormalization(),
            
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, test_size=0.2, validation_split=0.2, epochs=50, batch_size=64):
        """Train the model with improved configuration."""
        print("\nPreparing data for training...")
        
        # Filter out classes with insufficient samples for stratified split
        unique_labels, label_counts = np.unique(self.labels_encoded, return_counts=True)
        min_samples_required = max(2, int(1 / test_size))  # At least 2 samples, or enough for test split
        
        # Find classes with sufficient samples
        valid_classes = unique_labels[label_counts >= min_samples_required]
        
        if len(valid_classes) < len(unique_labels):
            print(f"Warning: Removing {len(unique_labels) - len(valid_classes)} classes with insufficient samples")
            print("Classes with insufficient samples:")
            for label, count in zip(unique_labels, label_counts):
                if count < min_samples_required:
                    class_name = self.class_names[label] if label < len(self.class_names) else f"Class_{label}"
                    print(f"  {class_name}: {count} samples (minimum required: {min_samples_required})")
        
        # Filter data to only include valid classes
        valid_indices = np.isin(self.labels_encoded, valid_classes)
        filtered_features = self.features[valid_indices]
        filtered_labels = self.labels_encoded[valid_indices]
        
        # Update class names to only include valid classes
        valid_class_names = [self.class_names[i] for i in valid_classes if i < len(self.class_names)]
        
        # Remap labels to be continuous (0, 1, 2, ...)
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(valid_classes)}
        remapped_labels = np.array([label_mapping[label] for label in filtered_labels])
        
        print(f"Training with {len(valid_classes)} classes and {len(filtered_features)} samples")
        
        # Calculate class weights for handling imbalance
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(remapped_labels),
            y=remapped_labels
        )
        class_weight_dict = dict(enumerate(class_weights))
        
        # Split data - use stratify only if we have enough samples
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                filtered_features, remapped_labels, 
                test_size=test_size, 
                random_state=42, 
                stratify=remapped_labels
            )
        except ValueError as e:
            print(f"Warning: Stratified split failed ({e}), using random split instead")
            X_train, X_test, y_train, y_test = train_test_split(
                filtered_features, remapped_labels, 
                test_size=test_size, 
                random_state=42
            )
        
        # Update class names for saving
        self.class_names = valid_class_names
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        # Create model
        num_classes = len(self.class_names)
        model = self.create_model(num_classes=num_classes)
        
        print("\nModel architecture:")
        model.summary()
        
        # Enhanced callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy', 
                patience=15, 
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.3, 
                patience=7, 
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                os.path.join(self.output_dir, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model with class weights
        print("\nStarting training...")
        history = model.fit(
            X_train, y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Test loss: {test_loss:.4f}")
        
        # Save model and metadata
        self.save_model_and_metadata(model, history, test_accuracy)
        
        # Plot training history
        self.plot_training_history(history)
        
        return model, history
    
    def save_model_and_metadata(self, model, history, test_accuracy):
        """Save the trained model and associated metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save Keras model
        model_path = os.path.join(self.output_dir, f'gesture_model_{timestamp}.h5')
        model.save(model_path)
        print(f"Model saved to: {model_path}")
        
        # Convert to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        tflite_path = os.path.join(self.output_dir, f'gesture_model_{timestamp}.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"TFLite model saved to: {tflite_path}")
        
        # Save labels
        labels_path = os.path.join(self.output_dir, f'labels_{timestamp}.txt')
        with open(labels_path, 'w') as f:
            for label in self.class_names:
                f.write(f"{label}\n")
        print(f"Labels saved to: {labels_path}")
        
        # Save label encoder
        encoder_path = os.path.join(self.output_dir, f'label_encoder_{timestamp}.pkl')
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save training metadata
        metadata = {
            'timestamp': timestamp,
            'test_accuracy': float(test_accuracy),
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'total_samples': len(self.features),
            'feature_shape': self.features.shape,
            'model_path': model_path,
            'tflite_path': tflite_path,
            'labels_path': labels_path
        }
        
        metadata_path = os.path.join(self.output_dir, f'training_metadata_{timestamp}.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to: {metadata_path}")
    
    def plot_training_history(self, history):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {plot_path}")
        plt.show()

def main():
    # Configuration
    DATASET_PATH = r"C:\Users\krishna\Downloads\archive (2)\Gesture Image Data"
    OUTPUT_DIR = "models"
    
    # Training parameters - Optimized for better performance
    MAX_SAMPLES_PER_CLASS = 500  # Increased for better generalization
    SAMPLE_RATIO = 0.1  # Use 10% of available images if less than max
    TEST_SIZE = 0.2
    VALIDATION_SPLIT = 0.2
    EPOCHS = 50  
    BATCH_SIZE = 64  # Increased for faster training
    
    print("=== Sign Language Gesture Dataset Training ===")
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Max samples per class: {MAX_SAMPLES_PER_CLASS}")
    print(f"Sample ratio: {SAMPLE_RATIO}")
    
    # Initialize trainer
    trainer = GestureDatasetTrainer(DATASET_PATH, OUTPUT_DIR)
    
    # Load dataset
    trainer.load_dataset(
        max_samples_per_class=MAX_SAMPLES_PER_CLASS,
        sample_ratio=SAMPLE_RATIO
    )
    
    # Train model
    model, history = trainer.train_model(
        test_size=TEST_SIZE,
        validation_split=VALIDATION_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    
    print("\n=== Training Complete ===")
    print("Check the 'models' directory for saved files:")
    print("- .h5 file: Keras model")
    print("- .tflite file: TensorFlow Lite model")
    print("- labels.txt: Class labels")
    print("- training_metadata.json: Training information")
    print("- training_history.png: Training plots")

if __name__ == "__main__":
    main()

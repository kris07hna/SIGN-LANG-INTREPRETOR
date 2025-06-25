import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import cv2
import os
import glob
from sign_utils import MediaPipeLandmarksModel
import matplotlib.pyplot as plt

class DigitSignTrainer:
    def __init__(self):
        self.landmark_model = MediaPipeLandmarksModel()
        self.img_size = (224, 224)
        
    def load_image_dataset(self, dataset_path="dataset"):
        """Load digit images and extract landmarks."""
        X_images = []
        X_landmarks = []
        y = []
        
        print("Loading and processing images...")
        
        # Process each digit folder
        for digit in range(10):
            digit_path = os.path.join(dataset_path, str(digit))
            if not os.path.exists(digit_path):
                continue
                
            image_files = glob.glob(os.path.join(digit_path, "*.JPG"))
            print(f"Processing {len(image_files)} images for digit {digit}")
            
            for img_file in image_files:
                # Load and preprocess image
                img = cv2.imread(img_file)
                if img is None:
                    continue
                    
                # Resize image
                img_resized = cv2.resize(img, self.img_size)
                
                # Extract landmarks
                landmarks = self.landmark_model.predict_landmarks(img)
                
                if landmarks is not None:
                    # Normalize landmarks
                    landmarks_normalized = self.landmark_model.normalize_landmarks(landmarks)
                    
                    X_images.append(img_resized)
                    X_landmarks.append(landmarks_normalized)
                    y.append(digit)
        
        print(f"Successfully processed {len(X_images)} images with landmarks")
        
        # Convert to numpy arrays
        X_images = np.array(X_images) / 255.0  # Normalize pixel values
        X_landmarks = np.array(X_landmarks)
        y = np.array(y)
        
        return X_images, X_landmarks, y
    
    def create_cnn_model(self, num_classes=10):
        """Create CNN model for image classification."""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_landmark_model(self, num_classes=10):
        """Create model for landmark-based classification."""
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
    
    def create_combined_model(self, num_classes=10):
        """Create combined model using both images and landmarks."""
        # Image branch
        image_input = tf.keras.Input(shape=(224, 224, 3), name='image_input')
        x1 = Conv2D(32, (3, 3), activation='relu')(image_input)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(2, 2)(x1)
        x1 = Conv2D(64, (3, 3), activation='relu')(x1)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(2, 2)(x1)
        x1 = Conv2D(128, (3, 3), activation='relu')(x1)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(2, 2)(x1)
        x1 = Flatten()(x1)
        x1 = Dense(256, activation='relu')(x1)
        x1 = Dropout(0.5)(x1)
        
        # Landmark branch
        landmark_input = tf.keras.Input(shape=(63,), name='landmark_input')
        x2 = Dense(128, activation='relu')(landmark_input)
        x2 = BatchNormalization()(x2)
        x2 = Dropout(0.3)(x2)
        x2 = Dense(64, activation='relu')(x2)
        x2 = Dropout(0.3)(x2)
        
        # Combine branches
        combined = tf.keras.layers.concatenate([x1, x2])
        combined = Dense(128, activation='relu')(combined)
        combined = Dropout(0.3)(combined)
        output = Dense(num_classes, activation='softmax')(combined)
        
        model = tf.keras.Model(inputs=[image_input, landmark_input], outputs=output)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, model_type='combined'):
        """Train the model."""
        # Load dataset
        X_images, X_landmarks, y = self.load_image_dataset()
        
        if len(X_images) == 0:
            print("No valid images found with hand landmarks!")
            return None
        
        # Split data
        if model_type == 'combined':
            X_img_train, X_img_test, X_land_train, X_land_test, y_train, y_test = train_test_split(
                X_images, X_landmarks, y, test_size=0.2, random_state=42, stratify=y
            )
            X_img_train, X_img_val, X_land_train, X_land_val, y_train, y_val = train_test_split(
                X_img_train, X_land_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_images if model_type == 'cnn' else X_landmarks, y, 
                test_size=0.2, random_state=42, stratify=y
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
        
        print(f"Training set: {len(y_train)} samples")
        print(f"Validation set: {len(y_val)} samples")
        print(f"Test set: {len(y_test)} samples")
        
        # Create model
        if model_type == 'cnn':
            model = self.create_cnn_model()
            train_data = (X_train, y_train)
            val_data = (X_val, y_val)
            test_data = (X_test, y_test)
        elif model_type == 'landmark':
            model = self.create_landmark_model()
            train_data = (X_train, y_train)
            val_data = (X_val, y_val)
            test_data = (X_test, y_test)
        else:  # combined
            model = self.create_combined_model()
            train_data = ([X_img_train, X_land_train], y_train)
            val_data = ([X_img_val, X_land_val], y_val)
            test_data = ([X_img_test, X_land_test], y_test)
        
        print(f"\nModel Summary ({model_type}):")
        model.summary()
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
            tf.keras.callbacks.ModelCheckpoint(
                f'models/best_{model_type}_model.h5', 
                save_best_only=True
            )
        ]
        
        # Train model
        print(f"\nTraining {model_type} model...")
        history = model.fit(
            train_data[0], train_data[1],
            validation_data=val_data,
            epochs=50,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        print(f"\nEvaluating {model_type} model...")
        test_loss, test_accuracy = model.evaluate(test_data[0], test_data[1], verbose=0)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Save model
        model.save(f'models/{model_type}_digit_model.h5')
        
        # Convert to TFLite
        self.convert_to_tflite(model, model_type)
        
        # Save labels
        labels = [str(i) for i in range(10)]
        with open('models/labels.txt', 'w') as f:
            f.write('\n'.join(labels))
        
        return model, history
    
    def convert_to_tflite(self, model, model_type):
        """Convert model to TensorFlow Lite."""
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            tflite_model = converter.convert()
            
            with open(f'models/{model_type}_model.tflite', 'wb') as f:
                f.write(tflite_model)
            
            # Also save as the default model.tflite
            with open('models/model.tflite', 'wb') as f:
                f.write(tflite_model)
                
            print(f"Model converted to TFLite: {model_type}_model.tflite")
            
        except Exception as e:
            print(f"TFLite conversion failed: {e}")
    
    def plot_training_history(self, history, model_type):
        """Plot training history."""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'{model_type.title()} Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{model_type.title()} Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'logs/{model_type}_training_history.png')
        plt.show()

def main():
    trainer = DigitSignTrainer()
    
    print("Sign Language Digits Model Training")
    print("=" * 40)
    
    # Check if dataset exists
    if not os.path.exists("dataset"):
        print("Dataset not found!")
        return
    
    # Train different models
    models_to_train = ['landmark', 'cnn', 'combined']
    
    for model_type in models_to_train:
        print(f"\n{'='*50}")
        print(f"Training {model_type.upper()} model")
        print(f"{'='*50}")
        
        try:
            model, history = trainer.train_model(model_type)
            if model is not None:
                trainer.plot_training_history(history, model_type)
                print(f"{model_type.title()} model training completed!")
            else:
                print(f"Failed to train {model_type} model")
        except Exception as e:
            print(f"Error training {model_type} model: {e}")
    
    print("\nTraining completed!")
    print("You can now run the main application: streamlit run app.py")

if __name__ == "__main__":
    main()

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import glob
from utils.model_utils import create_model, get_training_callbacks, evaluate_model
from utils.data_utils import DataProcessor

class SignLanguageTrainer:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.label_encoder = LabelEncoder()
        
    def load_dataset(self, dataset_path="dataset"):
        """Load dataset from collected data."""
        X, y = [], []
        labels = []
        
        # Get all sign directories
        sign_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        
        for sign_name in sign_dirs:
            sign_path = os.path.join(dataset_path, sign_name)
            sequence_file = os.path.join(sign_path, "sequences.npy")
            
            if os.path.exists(sequence_file):
                sequences = np.load(sequence_file)
                
                for sequence in sequences:
                    X.append(sequence)
                    y.append(sign_name)
                    
                if sign_name not in labels:
                    labels.append(sign_name)
                    
                print(f"Loaded {len(sequences)} sequences for {sign_name}")
        
        # Save labels
        with open('models/labels.txt', 'w') as f:
            f.write('\n'.join(labels))
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        return np.array(X), y_encoded, labels
    
    def prepare_data(self, X, y, test_size=0.2, val_size=0.2):
        """Prepare data for training."""
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Split train into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=42, stratify=y_train
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_model(self, X_train, y_train, X_val, y_val, 
                   num_classes, epochs=50, batch_size=32):
        """Train the sign language model."""
        
        # Get input shape
        sequence_length, num_features = X_train.shape[1], X_train.shape[2]
        
        # Create model
        model = create_model(num_classes, sequence_length, num_features)
        
        print("Model Summary:")
        model.summary()
        
        # Get callbacks
        callbacks = get_training_callbacks()
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return model, history
    
    def save_model(self, model, model_name="sign_language_model"):
        """Save the trained model."""
        # Save Keras model
        model.save(f'models/{model_name}.h5')
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Quantization for smaller model size
        def representative_dataset():
            for _ in range(100):
                yield [np.random.random((1, 30, 126)).astype(np.float32)]
        
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        
        try:
            tflite_model = converter.convert()
            with open('models/model.tflite', 'wb') as f:
                f.write(tflite_model)
            print("Model converted to TFLite successfully!")
        except Exception as e:
            print(f"TFLite conversion failed: {e}")
            # Fallback to float32 model
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            with open('models/model.tflite', 'wb') as f:
                f.write(tflite_model)
            print("Model converted to TFLite (float32) successfully!")

def main():
    trainer = SignLanguageTrainer()
    
    print("Sign Language Model Training")
    print("=" * 40)
    
    # Check if dataset exists
    if not os.path.exists("dataset"):
        print("Dataset not found! Please run data_collection.py first.")
        return
    
    # Load dataset
    print("Loading dataset...")
    X, y, labels = trainer.load_dataset()
    
    if len(X) == 0:
        print("No data found! Please collect data first using data_collection.py")
        return
    
    print(f"Loaded {len(X)} sequences for {len(labels)} signs")
    print(f"Signs: {labels}")
    
    # Prepare data
    print("Preparing data...")
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(X, y)
    
    print(f"Training set: {len(X_train)} sequences")
    print(f"Validation set: {len(X_val)} sequences")
    print(f"Test set: {len(X_test)} sequences")
    
    # Train model
    print("Training model...")
    model, history = trainer.train_model(
        X_train, y_train, X_val, y_val,
        num_classes=len(labels),
        epochs=50,
        batch_size=32
    )
    
    # Evaluate model
    print("Evaluating model...")
    results = evaluate_model(model, X_test, y_test)
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    
    # Save model
    print("Saving model...")
    trainer.save_model(model)
    
    print("Training completed successfully!")
    print("You can now run the main application: streamlit run app.py")

if __name__ == "__main__":
    main()

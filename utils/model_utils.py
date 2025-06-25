import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np

def create_model(num_classes: int, sequence_length: int = 30, num_features: int = 63) -> tf.keras.Model:
    """Create LSTM model for sign language recognition."""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(sequence_length, num_features)),
        Dropout(0.3),
        BatchNormalization(),
        
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        BatchNormalization(),
        
        LSTM(32, return_sequences=False),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        Dropout(0.2),
        
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_cnn_model(num_classes: int, input_shape: tuple = (224, 224, 3)) -> tf.keras.Model:
    """Create CNN model for static sign recognition."""
    model = Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def convert_to_tflite(model: tf.keras.Model, output_path: str) -> None:
    """Convert Keras model to TensorFlow Lite format."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Quantization for smaller model size
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

def representative_dataset_gen():
    """Generate representative dataset for quantization."""
    for _ in range(100):
        yield [np.random.random((1, 30, 63)).astype(np.float32)]

def get_training_callbacks():
    """Get training callbacks for model optimization."""
    return [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'models/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]

def evaluate_model(model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate model performance."""
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    
    from sklearn.metrics import classification_report, confusion_matrix
    
    report = classification_report(y_test, predicted_classes, output_dict=True)
    cm = confusion_matrix(y_test, predicted_classes)
    
    return {
        'classification_report': report,
        'confusion_matrix': cm,
        'accuracy': report['accuracy']
    }

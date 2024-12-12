import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

def load_dataset(file_path='heuristic_dataset.npy'):
    """
    Loads the dataset from a NumPy binary file.
    
    Args:
        file_path (str): Path to the NumPy dataset file.
    
    Returns:
        tuple: Features (X) and targets (y) as NumPy arrays.
    """
    try:
        data = np.load(file_path, allow_pickle=True)
        X = []
        y = []
        for sample in data:
            X.append(sample['input'])
            y.append(sample['output'])
        X = np.array(X)
        y = np.array(y)
        print(f"Loaded dataset with {X.shape[0]} samples.")
        return X, y
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None, None

def build_model(input_dim):
    """
    Builds and compiles an enhanced neural network model for regression on path costs.
    
    Args:
        input_dim (int): Number of input features.
    
    Returns:
        keras.Model: Compiled neural network model.
    """
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')  # Regression output
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def main():
    # Load dataset
    X, y = load_dataset('heuristic_dataset.npy')
    if X is None or y is None:
        return
    
    # Split into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}.")

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print("Feature scaling applied.")

    # Build model
    input_dim = X_train.shape[1]
    model = build_model(input_dim)
    model.summary()

    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True
    )
    checkpoint = ModelCheckpoint(
        'heuristic_model_best.h5', monitor='val_loss',
        save_best_only=True, verbose=1
    )

    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

    # Evaluate model
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test MAE: {mae:.2f}")

    # Post-training quantization for faster inference
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open('heuristic_model.tflite', 'wb') as f:
            f.write(tflite_model)
        print("TensorFlow Lite model saved as 'heuristic_model.tflite'.")
    except Exception as e:
        print(f"Failed to convert model to TensorFlow Lite: {e}")

    # Save scaler parameters
    try:
        np.save('scaler_mean.npy', scaler.mean_)
        np.save('scaler_scale.npy', scaler.scale_)
        print("Scaler parameters saved successfully.")
    except Exception as e:
        print(f"Failed to save scaler parameters: {e}")

if __name__ == "__main__":
    main()

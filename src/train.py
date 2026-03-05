import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.svm import SVC
import joblib
import os
import numpy as np

def train_digit_models():
    # 1. Load the MNIST Dataset
    print("📥 Loading MNIST dataset...")
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # --- MODEL 1: THE CNN (Spatial Brain) ---
    print("\n🚀 Training Model 1: Convolutional Neural Network (CNN)...")
    x_train_cnn = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    cnn_model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    
    cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cnn_model.fit(x_train_cnn, y_train, epochs=5)
    
    if not os.path.exists('models'):
        os.makedirs('models')
    cnn_model.save('models/digit_brain.h5')
    print("✅ CNN saved in models/digit_brain.h5")

    # --- MODEL 2: THE SVM (Geometric Brain) ---
    print("\n🤖 Training Model 2: Support Vector Machine (SVM)...")
    # SVMs don't understand 2D grids. We must flatten the 28x28 image into a 1D line of 784 pixels.
    x_train_flat = x_train.reshape(-1, 28*28).astype('float32') / 255.0
    
    # We use probability=True so it can "vote" with percentages, just like the CNN!
    svm_model = SVC(kernel='rbf', probability=True, random_state=42)
    
    # Training on a 10,000 image subset to save time
    print("Fitting SVM on a 10k subset (this might take 1-2 minutes)...")
    svm_model.fit(x_train_flat[:10000], y_train[:10000])
    
    joblib.dump(svm_model, 'models/digit_svm.pkl')
    print("✅ SVM saved in models/digit_svm.pkl")
    
    print("\n🎉 ENSEMBLE TRAINING COMPLETE!")

if __name__ == "__main__":
    train_digit_models()
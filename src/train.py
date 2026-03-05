import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.linear_model import SGDClassifier
import joblib
import os

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

    # --- MODEL 2: THE FAST SGD (Geometric Brain) ---
    print("\n🤖 Training Model 2: Fast SGD Classifier (Full 60k Images)...")
    x_train_flat = x_train.reshape(-1, 28*28).astype('float32') / 255.0
    
    # loss='log_loss' ensures it can output percentages (probabilities) for our Ensemble vote
    # n_jobs=-1 tells Python to use EVERY core in your computer's CPU!
    svm_model = SGDClassifier(loss='log_loss', random_state=42, n_jobs=-1)
    
    print("⚡ Fitting on ALL 60,000 images (Watch how fast this is!)...")
    svm_model.fit(x_train_flat, y_train)
    
    joblib.dump(svm_model, 'models/digit_svm.pkl')
    print("✅ Fast Geometric Brain saved in models/digit_svm.pkl")
    
    print("\n🎉 ENSEMBLE TRAINING COMPLETE!")

if __name__ == "__main__":
    train_digit_models()
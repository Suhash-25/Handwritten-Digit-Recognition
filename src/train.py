import tensorflow as tf
from tensorflow.keras import layers, models
import os

def train_digit_model():
    # 1. Load the MNIST Dataset
    print("📥 Loading MNIST dataset...")
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # 2. Pre-processing (The "Elite" normalization)
    # We reshape to (28, 28, 1) because CNNs expect 3D inputs (width, height, channels)
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    # 3. Architecting the Neural Network
    # Think of this as building a series of filters that "squint" at the image
    model = models.Sequential([
        # Layer 1: Look for simple edges/curves
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        
        # Layer 2: Combine edges into complex shapes (loops, crosses)
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Layer 3: Flatten the 2D image into a 1D line of data
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2), # "Elite" move: randomly shuts off neurons to prevent memorization
        
        # Output: 10 neurons (one for each digit 0-9)
        layers.Dense(10, activation='softmax')
    ])

    # 4. Compiling the Brain
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 5. Training
    print("🚀 Training started. Watch the 'accuracy' go up!")
    model.fit(x_train, y_train, epochs=5,validation_data=(x_test, y_test))

    # 6. Saving the Result
    if not os.path.exists('models'):
        os.makedirs('models')
    model.save('models/digit_brain.h5')
    print("✅ Model saved in models/digit_brain.h5")

if __name__ == "__main__":
    train_digit_model()
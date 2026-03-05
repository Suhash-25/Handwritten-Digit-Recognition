import tensorflow as tf
import gradio as gr
import numpy as np
import cv2
import os
import joblib

# ELITE PATHING
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CNN_PATH = os.path.join(BASE_DIR, '..', 'models', 'digit_brain.h5')
SVM_PATH = os.path.join(BASE_DIR, '..', 'models', 'digit_svm.pkl')

# 1. Load BOTH Brains
print(f"Loading CNN from: {CNN_PATH}")
cnn_model = tf.keras.models.load_model(CNN_PATH)

print(f"Loading SVM from: {SVM_PATH}")
svm_model = joblib.load(SVM_PATH)

def predict_digit(data):
    img = data['composite'] 
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    
    # Auto-Invert
    if img[0, 0] > 127: 
        img = cv2.bitwise_not(img)
        
    # Auto-Centering (Bounding Box)
    coords = cv2.findNonZero(img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        digit = img[y:y+h, x:x+w]
        length = max(w, h)
        pad_w = (length - w) // 2
        pad_h = (length - h) // 2
        digit = cv2.copyMakeBorder(digit, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)
        digit = cv2.resize(digit, (20, 20))
        img = cv2.copyMakeBorder(digit, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
    else:
        img = cv2.resize(img, (28, 28))

    # Dilation & Normalization
    kernel = np.ones((2, 2), np.uint8) 
    img = cv2.dilate(img, kernel, iterations=1)
    img_normalized = img / 255.0

    # --- THE ENSEMBLE VOTE ---
    # 1. Ask the CNN (Needs 4D shape: 1 image, 28x28, 1 channel)
    cnn_input = img_normalized.reshape(1, 28, 28, 1)
    cnn_pred = cnn_model.predict(cnn_input, verbose=0)[0]
    
    # 2. Ask the SVM (Needs 2D shape: 1 image, 784 pixels)
    svm_input = img_normalized.reshape(1, 784)
    svm_pred = svm_model.predict_proba(svm_input)[0]
    
    # 3. Combine the Votes! (We weight the CNN slightly higher because it's usually better at vision)
    final_prediction = (cnn_pred * 0.6) + (svm_pred * 0.4)
    
    return {str(i): float(final_prediction[i]) for i in range(10)}

# UI DESIGN
label = gr.Label(num_top_classes=3) 

interface = gr.Interface(
    fn=predict_digit, 
    inputs=gr.Sketchpad(label="Draw a Digit (0-9)", type="numpy"), 
    outputs=label,
    live=True, 
    title="Ultimate Ensemble Digit Recognizer",
    description="CNN + SVM Voting Architecture. Try to fool it!"
)

if __name__ == "__main__":
    interface.launch()
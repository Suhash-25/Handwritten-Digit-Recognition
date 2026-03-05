import tensorflow as tf
import gradio as gr
import numpy as np
import cv2
import os

# ELITE PATHING: Find out exactly where this app.py file lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'digit_brain.h5')

# 1. Load the "Brain"
print(f"Loading model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

def predict_digit(data):
    # 1. Extract the image from Gradio
    img = data['composite'] 
    
    # 2. Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    
    # 3. Auto-Invert Colors (Make background black, ink white)
    if img[0, 0] > 127: 
        img = cv2.bitwise_not(img)
        
    # 4. ELITE FIX: Auto-Centering (Bounding Box)
    # Find all pixels that aren't black
    coords = cv2.findNonZero(img)
    if coords is not None:
        # Get the coordinates for a tight box around the ink
        x, y, w, h = cv2.boundingRect(coords)
        
        # Crop the image to just the digit
        digit = img[y:y+h, x:x+w]
        
        # Figure out how to make it a perfect square without squishing it
        length = max(w, h)
        pad_w = (length - w) // 2
        pad_h = (length - h) // 2
        
        # Add black borders to make it square
        digit = cv2.copyMakeBorder(digit, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)
        
        # Resize to 20x20 (This is the secret MNIST standard!)
        digit = cv2.resize(digit, (20, 20))
        
        # Add exactly 4 pixels of black padding on all sides to reach 28x28
        img = cv2.copyMakeBorder(digit, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
    else:
        # If the canvas is empty, just default to 28x28
        img = cv2.resize(img, (28, 28))

    # 5. Dilation (Thicken the lines slightly)
    kernel = np.ones((2, 2), np.uint8) 
    img = cv2.dilate(img, kernel, iterations=1)
    
    # 6. Normalize (Scale 0-255 down to 0.0-1.0)
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)

    # 7. Predict!
    prediction = model.predict(img)[0]
    
    return {str(i): float(prediction[i]) for i in range(10)}

# 8. DESIGN THE ELITE UI
label = gr.Label(num_top_classes=3) 

interface = gr.Interface(
    fn=predict_digit, 
    inputs=gr.Sketchpad(label="Draw a Digit (0-9)", type="numpy"), 
    outputs=label,
    live=True, 
    title="Elite Digit Recognition System",
    description="Now with Auto-Centering! Draw anywhere, any size."
)

if __name__ == "__main__":
    interface.launch()
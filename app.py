import onnxruntime as ort
import numpy as np
import cv2
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os

# Load the ONNX model
model_path = "payment_classification.onnx"
session = ort.InferenceSession(model_path)

# Define image size
IMG_SIZE = (224, 224)

# Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads" #Saving uploaded images temporarily to a folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Preprocess function
def preprocess_image(image_path):
    image = cv2.imread(image_path)  # Read as (H, W, C)
    image = cv2.resize(image, IMG_SIZE)  # Resize to (224, 224, 3)
    image = image.astype(np.float32) / 255.0  # Normalize (0 to 1)
    # Shape order (batch_size, height, width, channels)
    image = np.expand_dims(image, axis=0)  # Add batch dimension (1, H, W, C)
    return image

# Predict function
def predict(image_path):
    image = preprocess_image(image_path)
    ort_inputs = {session.get_inputs()[0].name: image}
    ort_outs = session.run(None, ort_inputs) #Send input to the model
    prediction = ort_outs[0][0][0]  # Extract probabilities from the model's outputs

    # Delete the image after processing
    try:
        print(os.getcwd())
        os.remove(image_path)
    except Exception as e:
        print(f"Error deleting file: {e}")

    return "Payment Detected" if prediction > 0.5 else "No Payment Detected"

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            result = predict(filepath)
            return render_template('index.html', prediction=result)
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)

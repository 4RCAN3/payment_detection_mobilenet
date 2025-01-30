import onnxruntime as ort
import numpy as np
import cv2
import os
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the ONNX model
model_path = "payment_classification.onnx"
session = ort.InferenceSession(model_path)

# Image size
IMG_SIZE = (224, 224)

# Path to dataset
DATASET_DIR = "Dataset"  
LABELS = {"Yes": 1, "No": 0}  

# Preprocess function
def preprocess_image(image_path):
    image = cv2.imread(image_path)  # Read as (H, W, C)

    image = cv2.resize(image, IMG_SIZE)  # Resize to (224, 224, 3)
    image = image.astype(np.float32) / 255.0  # Normalize (0 to 1)

    #shape order (batch_size, height, width, channels)
    image = np.expand_dims(image, axis=0)  # Add batch dimension (1, H, W, C)
    return image

# Predict function
def predict(image_path):
    image = preprocess_image(image_path)
    ort_inputs = {session.get_inputs()[0].name: image}
    ort_outs = session.run(None, ort_inputs) #Send processed image to the model as input
    prediction = ort_outs[0][0][0]  # Extracting model probabilities
    return 1 if prediction > 0.5 else 0

# Loading dataset and evaluate
y_true = []
y_pred = []

for label, class_idx in LABELS.items():
    class_dir = os.path.join(DATASET_DIR, label)
    if not os.path.exists(class_dir):
        continue

    for file in os.listdir(class_dir):
        image_path = os.path.join(class_dir, file)
        try:
            y_pred.append(predict(image_path))
            y_true.append(class_idx)
        except cv2.error:
            continue


# Printclassification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=LABELS.keys()))

# Plot confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS.keys(), yticklabels=LABELS.keys())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

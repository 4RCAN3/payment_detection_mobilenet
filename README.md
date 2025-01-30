# Payment classification model
Created a classification model for detecting payments using mobilenet, achieving **94% accuracy** on the intended dataset

## Setting up your environment
Firstly, clone the repository to your working directory

### 1. Dataset
The dataset can be downloaded [here](https://drive.google.com/file/d/1mhK_NQGdcGcYNv0e0v6Ib9O81VyG_TIH/view?usp=sharing)
Store the dataset in your working directory (`Dataset/`), where images are organized into subdirectories corresponding to the two classes. (The dataset was split 80/20 for training purposes) <br>
Your tree should look like this:<br>
```
|--- Dataset
    |---- Yes
    |---- No
```

### 2. Install dependencies
Install dependencies using `pip install requirements.txt`

### 3. Inference
There are three ways in which inference for the model can be verified:
1. Run a flask interface where you upload images and the model predicts whether it detcts a payment or not. For this, in the working directory, type:<br>
`python app.py`
![image](https://github.com/user-attachments/assets/55f942b7-0578-4cf9-b5e7-3758606cde90)

3. Validate the performance of the model on the entire dataset using:<br>
`python inference.py`
3. To load the model to perform custom testing, the following python code can be followed:<br>
```
import onnxruntime as ort
model_path = "payment_classification.onnx"
session = ort.InferenceSession(model_path)
IMG_SIZE = (224, 224)

def preprocess_image(image_path):
    image = cv2.imread(image_path)  

    image = cv2.resize(image, IMG_SIZE) 
    image = image.astype(np.float32) / 255.0  

    image = np.expand_dims(image, axis=0)  
    return image

def predict(image_path):
    image = preprocess_image(image_path)
    ort_inputs = {session.get_inputs()[0].name: image}
    ort_outs = session.run(None, ort_inputs) 
    prediction = ort_outs[0][0][0]  
    return 1 if prediction > 0.5 else 0

predict(image_path) #Change image path according to you
```
**The model is saved as "payment_classification.onnx" in onnx format as instructed**

## Model Training procedure

To train the model, run the following command in your working directory:<br>
`python training.py`
### 1. Data Preprocessing
The dataset is preprocessed using `ImageDataGenerator`, which applies:
- **Normalization:** Rescales pixel values to [0,1]
- **Augmentation:** Includes rotation, shifting, shearing, zooming, and horizontal flipping for the training set
- **Validation split:** Ensures a separate validation set without augmentation

### 2. Model Architecture
The model is based on **MobileNetV2**:
- Uses pretrained **MobileNetV2** without the top classification layer (weights from ImageNet)
- Freezes the base model layers
- Adds:
  - `GlobalAveragePooling2D` for feature extraction
  - `Dense(128, activation='relu')` for classification
  - `Dropout(0.3)` to reduce overfitting
  - `Dense(1, activation='sigmoid')` for binary classification

### 3. Compilation and Training
- **Loss function:** `binary_crossentropy` (since this is a binary classification problem)
- **Optimizer:** Adam
- **Metrics:** Accuracy
- **Checkpointing:** Saves the best model based on validation loss
- **Epochs:** 10
- **Batch Size:** 32

### 4. Model Saving and Conversion
- The best trained model is saved as `best_model.h5`
- Converted to **ONNX format** using `tf2onnx`
- The ONNX model (`payment_classification.onnx`) can be used for deployment in different environments


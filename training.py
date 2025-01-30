import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import tf2onnx
import os

# Define paths
dataset_path = "Dataset"  # Change this to your dataset path
img_size = (224, 224)
batch_size = 32

# Data Augmentation & Loading
datagen = ImageDataGenerator(
    rescale=1./255, #Normalizing pixel values
    rotation_range=20, #Randomizing rotation by 20 degrees
    width_shift_range=0.2, #Random width shift
    height_shift_range=0.2, #Random height shift
    shear_range=0.2, #Random shear transformation
    zoom_range=0.2, #Random zoom
    horizontal_flip=True, #Random horizontal flip
    validation_split=0.2
)

#No augmentation to the validation set
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # No augmentation

#Load dataset directly from the directory
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

val_generator = val_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# Load Pretrained Model (imagenet trained, not loading the top classification layer)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze layers

# Add custom layers to the pretrained model 
# Average pooling layer (Added to the last layer of the loaded model)
# Fully Connected layer (128 hidden neurons)
# Dropout (30%)
# Classifcation layer (1 neuron/sigmoid activation func)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=output)

# Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Set up ModelCheckpoint to save the best model
checkpoint_callback = ModelCheckpoint(
    "best_model.h5", 
    monitor='val_loss', 
    mode='min', 
    save_best_only=True,
    verbose=1
)

# Train Model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=[checkpoint_callback]
)

# Load the best model
best_model = keras.models.load_model("best_model.h5")

# Convert the best model to ONNX, input shape for the model (batch, 224, 224, 3)
onnx_model_path = "payment_classification.onnx"
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
onnx_model, _ = tf2onnx.convert.from_keras(best_model, input_signature=spec, opset=13)

# Save the ONNX model
with open(onnx_model_path, "wb") as f:
    f.write(onnx_model.SerializeToString())


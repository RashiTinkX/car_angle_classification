import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Convolution2D,MaxPooling2D,Flatten #action detectionimport tensorflow
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import HTML

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import kagglehub

# Download latest version of the dataset
path = kagglehub.dataset_download("amarcodes/car-angle-classification-dataset")
print("Path to dataset files:", path)  # Confirm the dataset path

# Set the dataset path based on download location
dataset_path = path

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the image size for consistency
IMAGE_SIZE = 224

# Configure ImageDataGenerator with validation split and augmentation parameters
datagen = ImageDataGenerator(
    validation_split=0.2,
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.2
)

# Train and validation data generators using the dynamic dataset path and IMAGE_SIZE
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=16,
    class_mode='sparse',
    subset='training'
)

test_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=16,
    class_mode='sparse',
    subset='validation'
)
class_names = list(train_generator.class_indices.keys())
class_names
sz = 224

# Model
model = Sequential()
model.add(Convolution2D(filters=32, kernel_size=(3, 3), input_shape=(sz, sz, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.50))#will prevent overfitting
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=8, activation='softmax'))
model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['accuracy'])
#evaluating test and train accuracy
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=10
)
scores = model.evaluate(test_generator)
history.history.keys()
type(history.history['loss'])
len(history.history['loss'])
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

import matplotlib.pyplot as plt
EPOCHS = 10

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), acc, label='Training Accuracy')
plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
model.save("car_angle_classification_model.h5")
from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
from PIL import Image
import io

# Load the trained model
model = load_model("car_angle_classification_model.h5")
class_labels = ["0°", "40°", "90°", "130°", "180°", "230°", "270°", "320°"]  # Replace with actual angle classes if different

# Initialize FastAPI app
app = FastAPI()

def preprocess_image(uploaded_image):
    """ Preprocess the uploaded image to the required input size for the model """
    img = Image.open(io.BytesIO(uploaded_image)).convert("RGB")
    img = img.resize((128, 128))  # Resize to model's input size
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
    return img_array

@app.post("/predict/")
async def predict_angle(file: UploadFile = File(...)):
    # Read image file
    uploaded_image = await file.read()
    
    # Preprocess the image
    img_array = preprocess_image(uploaded_image)
    
    # Make predictions
    predictions = model.predict(img_array)
    confidence_scores = np.max(predictions)  # Confidence score for the highest probability class
    predicted_class = np.argmax(predictions)  # Class with highest probability

    # Format response
    angle_prediction = class_labels[predicted_class]
    confidence_score = float(confidence_scores) * 100  # Convert to percentage

    return {
        "predicted_angle": angle_prediction,
        "confidence_score": f"{confidence_score:.2f}%"
    }
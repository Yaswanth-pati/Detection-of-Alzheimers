# utils.py
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# Class labels (index must match training labels)
CLASS_NAMES = ["Non Demented", "Very Mild Demented", "Mild Demented", "Moderate Demented"]

# Load VGG16 exactly like in training
vgg_model = VGG16(include_top=False, input_shape=(128, 128, 3), weights='imagenet')
vgg_model.trainable = False

# Load trained classifier
def load_trained_model():
    return load_model("model/model.h5")

# Resize and preprocess uploaded image
def preprocess_image(image, target_size=(128, 128)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# Predict using VGG features and classifier
def predict_image(model, image):
    processed = preprocess_image(image)                      # (1, 128, 128, 3)
    features = vgg_model.predict(processed)                  # (1, 4, 4, 512)
    features = features.reshape(1, -1)                        # (1, 8192)
    preds = model.predict(features)[0]                       # (4,)
    label = CLASS_NAMES[np.argmax(preds)]
    confidence = float(np.max(preds))
    return label, confidence
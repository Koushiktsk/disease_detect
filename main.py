import streamlit as st
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from PIL import Image

# Load the trained model
model = load_model('models/model.h5')

# Class labels
class_labels = ['glioma', 'notumor', 'meningioma', 'pituitary']

# Helper function to predict tumor type
def predict_tumor(image):
    IMAGE_SIZE = 64
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0]

    if class_labels[predicted_class_index] == 'notumor':
        return "No Tumor", confidence_score
    else:
        return f"Tumor: {class_labels[predicted_class_index]}", confidence_score

# Streamlit UI
st.set_page_config(page_title="Brain Tumor Detector", layout="centered")
st.title("🧠 Brain Tumor Detection from MRI")

uploaded_file = st.file_uploader("Upload an MRI scan image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    result, confidence = predict_tumor(image)
    st.markdown(f"### Prediction: {result}")
    st.markdown(f"**Confidence:** {confidence * 100:.2f}%")



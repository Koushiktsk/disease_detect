# 🧠 Brain Tumor Detection using Deep Learning

This project is a web-based application for detecting brain tumors from MRI images using a Convolutional Neural Network (CNN) model. It allows users to upload an image, and the model will classify whether the MRI shows signs of a tumor and, if so, the type.

## 📊 Model Performance

- **Model Accuracy**: ~70%  
- **Model Type**: CNN based on MobileNetV2  
- **Reason for Accuracy**: Limited GPU capacity; trained on lower-resolution images (64x64) to accommodate hardware constraints; can't able to deploy due to large pickle file space.

---

## 🏷️ Tumor Classes

The model is trained to detect the following brain tumor types:

- Glioma
- Meningioma
- Pituitary
- No Tumor

---

## 🚀 Features

- Upload MRI image in `.jpg` or `.png` format
- Predicts the type of tumor (if present)
- Displays prediction confidence
- Fully web-based (Flask/Streamlit supported)

---

## 📦 Project Structure


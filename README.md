# Alzheimer’s Disease Detection from Brain MRI using CNN

## Project Introduction

Alzheimer’s Disease is a progressive neurological disorder that affects memory and cognitive function. Early detection is crucial for managing and slowing the progression of the disease. This project leverages deep learning and medical imaging to classify brain MRI scans into different stages of Alzheimer’s disease.

Using a pre-trained VGG16 Convolutional Neural Network as a feature extractor, we fine-tune a custom classification head to distinguish between four diagnostic categories:
	•	Non Demented
	•	Very Mild Demented
	•	Mild Demented
	•	Moderate Demented

The project is deployed as a user-friendly Streamlit web application, where users can upload an MRI image and instantly receive a prediction of the disease stage. This tool demonstrates the potential of AI in assisting medical diagnosis and increasing awareness around cognitive health conditions.

A Streamlit web app that classifies Alzheimer’s disease stage using deep learning on brain MRI images.

## Project Description

This project uses a Convolutional Neural Network (CNN) with transfer learning (VGG16) to classify brain MRI images into four categories:
- **Non Demented**
- **Very Mild Demented**
- **Mild Demented**
- **Moderate Demented**

The app is designed to assist in early detection and awareness using a simple UI built with Streamlit.

## Repository Structure
```
Alzheimers Project/
├── model/
│   └── model.h5                  # Trained Keras model
├── app.py                        # Streamlit app frontend
├── utils.py                      # Image preprocessing & prediction logic
├── requirements.txt              # Python and streamlit  dependencies
├── Alzheimers project.ipynb      # Model training notebook
└── Data/                         # Dataset (used only during training)
    
```
## Deployment

Deployed on **Streamlit Cloud**:  
 [https://alzheimers-classifier.streamlit.app/](#)

### Or run locally:

```bash

pip install -r requirements.txt
streamlit run app.py

```

## Authors

**Yaswanth Pati** – Data Scientist

# Alzheimer’s Disease Detection from Brain MRI using CNN

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
├── Detection_of_Alzheimer.docx   # Project report
└── Data/                         # Dataset (used only during training)
    
```
## Deployment

Deployed on **Streamlit Cloud**:  
 [Link to the Live App](#) ← *(coming soon)*

### Or run locally:

```bash

pip install -r requirements.txt
streamlit run app.py

```

## Authors

**Yaswanth Pati** – Data Scientist

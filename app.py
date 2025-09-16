# app.py
import streamlit as st
from PIL import Image
from utils import load_trained_model, predict_image

st.set_page_config(page_title="Alzheimer’s MRI Classifier", layout="centered")

# 🔹 Title
st.markdown("""
    <div style='text-align: center; padding: 1rem 0 0 0;'>
        <h1 style='color: #38bdf8; font-size: 2.5rem;'>🧠 Alzheimer’s Detection from MRI</h1>
        <p style='font-size: 1.1rem; color: #bbb;'>Upload a brain MRI image (JPEG or PNG), and this app will predict the stage of Alzheimer’s disease.</p>
    </div>
""", unsafe_allow_html=True)

# 🔹 Label Legend
st.markdown("""
<div style='text-align: center;'>
    <ul style='list-style: none; padding-left: 0; font-size: 1rem; color: #ccc;'>
        <li>🟢 <b>Non Demented</b></li>
        <li>🟡 <b>Very Mild Demented</b></li>
        <li>🟠 <b>Mild Demented</b></li>
        <li>🔴 <b>Moderate Demented</b></li>
    </ul>
</div>
""", unsafe_allow_html=True)

# 🔹 File Uploader
uploaded_file = st.file_uploader("📤 Upload MRI Image", type=["jpg", "jpeg", "png"])

# 🔹 Prediction Flow
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded MRI", use_container_width=True)

    with st.spinner("🔍 Classifying... Please wait"):
        model = load_trained_model()
        label, confidence = predict_image(model, image)

    st.markdown("---")
    st.success(f"🎯 **Prediction:** {label}")
    st.markdown(f"🔬 **Confidence:** `{confidence * 100:.2f}%`")
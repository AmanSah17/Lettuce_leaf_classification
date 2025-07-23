# app/streamlit_app.py
import streamlit as st
import requests
from PIL import Image

st.title("Lettuce Leaf Classifier")
st.markdown("Upload a lettuce leaf image to classify.")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify"):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post("http://fastapi:8000/predict/", files=files)

        if response.status_code == 200:
            data = response.json()
            st.success(f"Prediction: {data['class']} (Confidence: {data['probability']})")
        else:
            st.error("Failed to get prediction from API.")

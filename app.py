import streamlit as st
import numpy as np
import pandas as pd
import cv2
import mahotas
import joblib

# === Preprocessing Functions ===
def compute_brightness(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def extract_texture(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return np.zeros(13)
    gray = cv2.cvtColor(cv2.resize(img, (240, 240)), cv2.COLOR_BGR2GRAY)
    return mahotas.features.haralick(gray).mean(axis=0)

def extract_histogram(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(cv2.resize(img, (240, 240)), cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
    return hist.flatten() / hist.sum()

# === Load trained model and scaler ===
model = joblib.load("Lctf_voting_model.pkl")
scaler = joblib.load("scaler.pkl")  # Save your fitted StandardScaler during training

# === Streamlit UI ===
st.title("Lettuce Leaf Classifier 🌿")
st.write("Upload a leaf image to classify its condition.")

uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.read())

    st.image("temp.jpg", caption="Uploaded Image", use_column_width=True)

    # === Feature Extraction ===
    brightness = compute_brightness("temp.jpg")
    texture = extract_texture("temp.jpg")
    histogram = extract_histogram("temp.jpg")

    feature_vector = np.hstack([brightness, texture, histogram])
    feature_vector = scaler.transform([feature_vector])  # scale

    # === Prediction ===
    prediction = model.predict(feature_vector)[0]
    probas = model.predict_proba(feature_vector)[0]
    confidence = np.max(probas) * 100

    st.success(f"🧠 **Predicted Class:** {prediction}")
    st.info(f"🔍 **Confidence Score:** {confidence:.2f}%")

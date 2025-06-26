# app.py
import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

MODEL_PATH = r"C:\Users\suman\AppData\Local\Programs\Python\Python312\Py.Models\catboost_fog_model.cbm"

# App Title
st.set_page_config(page_title="FoG Detector", layout="centered")
st.title("🧠 Freezing of Gait Detection (CatBoost)")

# Load model
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    return model

model = load_model()

# Upload file
uploaded_file = st.file_uploader("📂 Upload a 100×N CSV file", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, header=None)
        st.success("✅ File uploaded successfully.")
        st.write(df.head())

        if df.shape[0] != 100:
            st.error("❌ File must have exactly 100 rows.")
        else:
            input_data = df.to_numpy().flatten().reshape(1, -1)
            prediction = model.predict(input_data)[0]
            label = "🔵 No FoG" if prediction == 0 else "🔴 Freezing of Gait Detected"
            st.success(f"🧬 Prediction: **{label}**")
    except Exception as e:
        st.error(f"❌ Error while reading or predicting: {e}")
else:
    st.info("Upload a CSV file to begin.")

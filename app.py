import streamlit as st
import pandas as pd
import joblib
import base64
import matplotlib.pyplot as plt
from deep_translator import GoogleTranslator

model = joblib.load("crop_model.pkl")
features = joblib.load("feature_names.pkl")

st.set_page_config(page_title="Smart Crop Recommendation", layout="wide")

def add_bg(image):
    with open(image, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg("background.jpg")

st.title("🌾 Smart Crop Recommendation System")

st.sidebar.header("Enter Soil & Climate Values")

N = st.sidebar.number_input("Nitrogen (N)", 0, 200)
P = st.sidebar.number_input("Phosphorus (P)", 0, 200)
K = st.sidebar.number_input("Potassium (K)", 0, 200)
temperature = st.sidebar.number_input("Temperature (°C)", 0.0, 50.0)
humidity = st.sidebar.number_input("Humidity (%)", 0.0, 100.0)
ph = st.sidebar.number_input("Soil pH", 0.0, 14.0)
rainfall = st.sidebar.number_input("Rainfall (mm)", 0.0, 500.0)

language = st.sidebar.selectbox(
    "Select Language",
    ["en","hi","ta","te","kn","ml","mr","bn","gu","pa"]
)

if st.sidebar.button("Recommend Crop"):
    input_data = pd.DataFrame(
        [[N,P,K,temperature,humidity,ph,rainfall]],
        columns=features
    )

    crop = model.predict(input_data)[0]
    text = f"Recommended Crop: {crop}"
    translated = GoogleTranslator(source="auto", target=language).translate(text)

    st.success(translated)

    st.subheader("Feature Importance")
    fig, ax = plt.subplots()
    ax.barh(features, model.feature_importances_)
    st.pyplot(fig)

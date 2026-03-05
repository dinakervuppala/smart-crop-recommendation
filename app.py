import streamlit as st
import pandas as pd
import joblib
import base64
from deep_translator import GoogleTranslator

# Load model
model = joblib.load("crop_model.pkl")
features = joblib.load("feature_names.pkl")

# Page config
st.set_page_config(page_title="Smart Crop Recommendation", layout="wide")

# Background image
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

# Title
st.title("🌾 Smart Crop Recommendation System")

# Language options
languages = {
    "English":"en",
    "Hindi":"hi",
    "Telugu":"te",
    "Tamil":"ta",
    "Kannada":"kn",
    "Malayalam":"ml",
    "Marathi":"mr",
    "Bengali":"bn",
    "Gujarati":"gu",
    "Punjabi":"pa",
    "Odia":"or"
}

language = st.selectbox("🌐 Select Language", list(languages.keys()))

# Fertilizer dictionary
fertilizer_recommendation = {
    "rice": "Use NPK fertilizer and organic compost.",
    "maize": "Use Nitrogen rich fertilizer and farmyard manure.",
    "chickpea": "Use phosphorus rich fertilizer.",
    "kidneybeans": "Use compost and potash fertilizer.",
    "pigeonpeas": "Use organic manure and phosphate fertilizer.",
    "mothbeans": "Use nitrogen fertilizer.",
    "mungbean": "Use nitrogen and phosphorus fertilizer.",
    "blackgram": "Use farmyard manure.",
    "lentil": "Use organic compost.",
    "pomegranate": "Use potassium rich fertilizer.",
    "banana": "Use potassium fertilizer.",
    "mango": "Use organic manure.",
    "grapes": "Use phosphorus fertilizer.",
    "watermelon": "Use nitrogen and potassium fertilizer.",
    "muskmelon": "Use nitrogen fertilizer.",
    "apple": "Use organic compost.",
    "orange": "Use nitrogen fertilizer.",
    "papaya": "Use potassium fertilizer.",
    "coconut": "Use NPK fertilizer.",
    "cotton": "Use potash fertilizer and urea.",
    "jute": "Use nitrogen fertilizer.",
    "coffee": "Use organic manure and compost."
}

# Sidebar Inputs
st.sidebar.header("Enter Soil & Climate Values")

N = st.sidebar.number_input("Nitrogen (N)",0,200)
P = st.sidebar.number_input("Phosphorus (P)",0,200)
K = st.sidebar.number_input("Potassium (K)",0,200)
temperature = st.sidebar.number_input("Temperature (°C)",0.0,50.0)
humidity = st.sidebar.number_input("Humidity (%)",0.0,100.0)
ph = st.sidebar.number_input("Soil pH",0.0,14.0)
rainfall = st.sidebar.number_input("Rainfall (mm)",0.0,500.0)

# Translation function
def translate_text(text, lang):
    if lang == "en":
        return text
    try:
        return GoogleTranslator(source="auto", target=lang).translate(text)
    except:
        return text

# Predict button
if st.sidebar.button("Recommend Crop"):

    input_data = pd.DataFrame(
        [[N,P,K,temperature,humidity,ph,rainfall]],
        columns=features
    )

    predicted_crop = model.predict(input_data)[0]

    fertilizer = fertilizer_recommendation.get(
        predicted_crop,
        "Use general organic fertilizers."
    )

    translated_crop = translate_text(predicted_crop.capitalize(), languages[language])
    translated_fertilizer = translate_text(fertilizer, languages[language])

    st.markdown(f"""
    <div style="
        background-color:#e8f5e9;
        padding:30px;
        border-radius:15px;
        text-align:center;
        font-size:28px;
        font-weight:bold;
        color:#1b5e20;
    ">

    🌾 Recommended Crop : {translated_crop}

    <br><br>

    🌱 Recommended Fertilizer : {translated_fertilizer}

    </div>
    """, unsafe_allow_html=True)

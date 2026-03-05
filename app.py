import streamlit as st
import pandas as pd
import joblib
import base64
import matplotlib.pyplot as plt
crop_translation = {
    "rice": {"English": "Rice", "Telugu": "వరి", "Hindi": "चावल", "Tamil": "அரிசி"},
    "maize": {"English": "Maize", "Telugu": "మొక్కజొన్న", "Hindi": "मक्का", "Tamil": "சோளம்"},
    "chickpea": {"English": "Chickpea", "Telugu": "సెనగలు", "Hindi": "चना", "Tamil": "கொண்டைக்கడலை"},
    "cotton": {"English": "Cotton", "Telugu": "పత్తి", "Hindi": "कपास", "Tamil": "பருத்தி"},
    "banana": {"English": "Banana", "Telugu": "అరటి", "Hindi": "केला", "Tamil": "வாழைப்பழம்"}
}

fertilizer_recommendation = {
    "rice": "Use NPK fertilizer and organic compost.",
    "maize": "Use Nitrogen rich fertilizer and farmyard manure.",
    "cotton": "Use potash fertilizer and urea.",
    "banana": "Use potassium rich fertilizer.",
}
def translate_crop(crop, language):
    try:
        translated = GoogleTranslator(source='auto', target=language).translate(crop)
        return translated
    except:
        return crop

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
language = st.selectbox(
    "🌐 Select Language",
    [
        "English",
        "Hindi",
        "Telugu",
        "Tamil",
        "Kannada",
        "Malayalam",
        "Marathi",
        "Bengali",
        "Gujarati",
        "Punjabi",
        "Odia"
    ]
)
st.sidebar.header("Enter Soil & Climate Values")

N = st.sidebar.number_input("Nitrogen (N)", 0, 200)
P = st.sidebar.number_input("Phosphorus (P)", 0, 200)
K = st.sidebar.number_input("Potassium (K)", 0, 200)
temperature = st.sidebar.number_input("Temperature (°C)", 0.0, 50.0)
humidity = st.sidebar.number_input("Humidity (%)", 0.0, 100.0)
ph = st.sidebar.number_input("Soil pH", 0.0, 14.0)
rainfall = st.sidebar.number_input("Rainfall (mm)", 0.0, 500.0)

if st.sidebar.button("Recommend Crop"):
    input_data = pd.DataFrame(
        [[N, P, K, temperature, humidity, ph, rainfall]],
        columns=features
    )

from deep_translator import GoogleTranslator
    def translate_crop(crop, language):

    lang_codes = {
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

    if language == "English":
        return crop.capitalize()

    try:
        translated = GoogleTranslator(
            source='auto',
            target=lang_codes[language]
        ).translate(crop)

        return translated

    except:
        return crop.capitalize()
        translated_crop = translate_crop(predicted_crop, language)

   translated_fertilizer = translate_crop(fertilizer, language)
st.write(translated_fertilizer)
        "Use general organic fertilizers."
    )

    st.markdown(f"""
<div style="background-color:#e8f5e9;
padding:25px;
border-radius:15px;
text-align:center;
font-size:26px;
font-weight:bold;
color:#1b5e20;">

Recommended Crop : {translated_crop}

<br><br>

Recommended Fertilizer : {translated_fertilizer}

</div>
""", unsafe_allow_html=True)
 

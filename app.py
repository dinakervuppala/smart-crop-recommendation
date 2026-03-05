import streamlit as st
import pandas as pd
import joblib
import base64
import shap
import matplotlib.pyplot as plt
from deep_translator import GoogleTranslator
from gtts import gTTS
import requests
import tempfile

# Load ML model
model = joblib.load("crop_model.pkl")
features = joblib.load("feature_names.pkl")

st.set_page_config(page_title="Smart Crop Recommendation", layout="wide")

# Background Image
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

# Language Options
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

# Fertilizer Recommendation
fertilizer_recommendation = {
    "rice":"Use NPK fertilizer and organic compost.",
    "maize":"Use nitrogen rich fertilizer and farmyard manure.",
    "banana":"Use potassium fertilizer.",
    "mango":"Use organic manure.",
    "grapes":"Use phosphorus fertilizer.",
    "apple":"Use organic compost.",
    "cotton":"Use potash fertilizer and urea.",
    "coffee":"Use organic manure and compost."
}

# Crop Images
crop_images = {
    "rice":"https://upload.wikimedia.org/wikipedia/commons/6/6f/Rice_Plant.jpg",
    "maize":"https://upload.wikimedia.org/wikipedia/commons/4/4e/Maize.jpg",
    "banana":"https://upload.wikimedia.org/wikipedia/commons/8/8a/Banana_tree.jpg",
    "mango":"https://upload.wikimedia.org/wikipedia/commons/9/90/Hapus_Mango.jpg",
    "grapes":"https://upload.wikimedia.org/wikipedia/commons/b/bb/Table_grapes_on_white.jpg",
    "apple":"https://upload.wikimedia.org/wikipedia/commons/1/15/Red_Apple.jpg",
    "cotton":"https://upload.wikimedia.org/wikipedia/commons/4/4f/Cotton_bolls.jpg",
    "coffee":"https://upload.wikimedia.org/wikipedia/commons/4/45/Coffea_arabica.jpg"
}

# Crop Tips
crop_tips = {
    "rice":"Requires high rainfall and standing water.",
    "maize":"Needs warm weather and nitrogen rich soil.",
    "banana":"Requires potassium rich soil.",
    "mango":"Grows well in tropical climates.",
    "grapes":"Needs well drained soil.",
    "apple":"Requires cool climate.",
    "cotton":"Needs warm climate.",
    "coffee":"Prefers shaded areas."
}

# Translation Function
def translate_text(text, lang):
    if lang == "en":
        return text
    try:
        return GoogleTranslator(source="auto", target=lang).translate(text)
    except:
        return text

# Soil Health
def soil_health(ph):
    if ph < 5:
        return "Soil is acidic"
    elif ph > 8:
        return "Soil is alkaline"
    else:
        return "Soil is healthy"

# Weather API
st.sidebar.subheader("🌦 Auto Weather Detection")

city = st.sidebar.text_input("Enter City Name")

temperature = 25
humidity = 60

if city:
    try:
        url = f"https://wttr.in/{city}?format=j1"
        weather = requests.get(url).json()
        temperature = float(weather["current_condition"][0]["temp_C"])
        humidity = float(weather["current_condition"][0]["humidity"])

        st.sidebar.success(f"Temperature: {temperature} °C")
        st.sidebar.success(f"Humidity: {humidity} %")
    except:
        st.sidebar.warning("Weather data not available")

# Sidebar Inputs
st.sidebar.header("Enter Soil & Climate Values")

N = st.sidebar.number_input("Nitrogen (N)",0,200)
P = st.sidebar.number_input("Phosphorus (P)",0,200)
K = st.sidebar.number_input("Potassium (K)",0,200)
ph = st.sidebar.number_input("Soil pH",0.0,14.0)
rainfall = st.sidebar.number_input("Rainfall (mm)",0.0,500.0)

# Prediction
if st.sidebar.button("Recommend Crop"):

    input_data = pd.DataFrame(
        [[N,P,K,temperature,humidity,ph,rainfall]],
        columns=features
    )

    predicted_crop = model.predict(input_data)[0]

    fertilizer = fertilizer_recommendation.get(predicted_crop,"Use organic fertilizer.")
    tip = crop_tips.get(predicted_crop,"Follow standard cultivation practices.")
    soil_status = soil_health(ph)

    translated_crop = translate_text(predicted_crop.capitalize(), languages[language])
    translated_fertilizer = translate_text(fertilizer, languages[language])
    translated_tip = translate_text(tip, languages[language])
    translated_soil = translate_text(soil_status, languages[language])

    st.markdown(f"""
    <div style="
    background-color:#e8f5e9;
    padding:30px;
    border-radius:15px;
    text-align:center;
    font-size:26px;
    font-weight:bold;
    color:#1b5e20">

    🌾 Recommended Crop : {translated_crop}

    <br><br>

    🌱 Recommended Fertilizer : {translated_fertilizer}

    <br><br>

    🌍 Soil Condition : {translated_soil}

    <br><br>

    📋 Growing Tips : {translated_tip}

    </div>
    """, unsafe_allow_html=True)

    if predicted_crop in crop_images:
        st.image(crop_images[predicted_crop], width=400)

    # 🔊 Voice Output
    speech = f"Recommended crop is {predicted_crop}. Fertilizer recommendation is {fertilizer}"
    tts = gTTS(speech)

    temp_audio = tempfile.NamedTemporaryFile(delete=False)
    tts.save(temp_audio.name)

    st.audio(temp_audio.name)

    # 🤖 Explainable AI
st.subheader("🤖 Explainable AI - Feature Contribution")

explainer = shap.TreeExplainer(model)

shap_values = explainer(input_data)

fig = plt.figure()

shap.plots.waterfall(shap_values[0], show=False)

st.pyplot(fig)

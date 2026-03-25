import streamlit as st
import pandas as pd
import joblib
import base64
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from deep_translator import GoogleTranslator
from gtts import gTTS
import requests
import tempfile

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Smart Crop Recommendation", layout="wide")

# -------------------------------
# LOAD MODEL
# -------------------------------
model = joblib.load("crop_model.pkl")
features = joblib.load("feature_names.pkl")

# -------------------------------
# MODEL ACCURACY
# -------------------------------
if st.button("Show Model Accuracy"):
    try:
        df = pd.read_csv("Crop_recommendation.csv")

        X = df.drop("label", axis=1)
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        st.success(f"Model Accuracy: {acc:.2f}")

    except:
        st.warning("Dataset not found.")

# -------------------------------
# MODEL COMPARISON (NEW 🔥)
# -------------------------------
if st.button("Compare Models Accuracy"):
    try:
        df = pd.read_csv("Crop_recommendation.csv")

        X = df.drop("label", axis=1)
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        rf = RandomForestClassifier()
        lr = LogisticRegression(max_iter=2000)
        dt = DecisionTreeClassifier()

        rf.fit(X_train, y_train)
        lr.fit(X_train, y_train)
        dt.fit(X_train, y_train)

        rf_acc = accuracy_score(y_test, rf.predict(X_test))
        lr_acc = accuracy_score(y_test, lr.predict(X_test))
        dt_acc = accuracy_score(y_test, dt.predict(X_test))

        results_df = pd.DataFrame({
            "Model": ["Random Forest", "Logistic Regression", "Decision Tree"],
            "Accuracy": [rf_acc, lr_acc, dt_acc]
        })

        st.subheader("📊 Model Comparison")
        st.dataframe(results_df)

        # Optional graph
        st.bar_chart(results_df.set_index("Model"))

    except:
        st.warning("Dataset not found.")

# -------------------------------
# BACKGROUND IMAGE
# -------------------------------
def add_bg(image):
    try:
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
    except:
        pass

add_bg("background.jpg")

# -------------------------------
# TITLE
# -------------------------------
st.title("🌾 Smart Crop Recommendation System")

# -------------------------------
# LANGUAGE
# -------------------------------
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

# -------------------------------
# TRANSLATION
# -------------------------------
def translate_text(text, lang):
    if lang == "en":
        return text
    try:
        return GoogleTranslator(source="auto", target=lang).translate(text)
    except:
        return text

# -------------------------------
# FERTILIZER
# -------------------------------
fertilizer_recommendation = {
    "rice":"Use NPK fertilizer and organic compost.",
    "maize":"Use nitrogen fertilizer.",
    "banana":"Use potassium fertilizer.",
    "mango":"Use organic manure.",
    "grapes":"Use phosphorus fertilizer.",
    "apple":"Use organic compost.",
    "cotton":"Use potash fertilizer.",
    "coffee":"Use organic manure."
}

# -------------------------------
# TIPS
# -------------------------------
crop_tips = {
    "rice":"Needs high water.",
    "maize":"Needs warm weather.",
    "banana":"Needs potassium soil.",
    "mango":"Grows in tropical climate.",
    "grapes":"Needs well drained soil.",
    "apple":"Needs cool climate.",
    "cotton":"Needs warm climate.",
    "coffee":"Needs shade."
}

# -------------------------------
# SOIL HEALTH
# -------------------------------
def soil_health(ph):
    if ph < 5:
        return "Acidic soil"
    elif ph > 8:
        return "Alkaline soil"
    else:
        return "Healthy soil"

# -------------------------------
# WEATHER
# -------------------------------
st.sidebar.subheader("🌦 Weather")

city = st.sidebar.text_input("Enter City")

temperature = 25
humidity = 60

if city:
    try:
        url = f"https://wttr.in/{city}?format=j1"
        weather = requests.get(url).json()
        temperature = float(weather["current_condition"][0]["temp_C"])
        humidity = float(weather["current_condition"][0]["humidity"])

        st.sidebar.success(f"Temp: {temperature}°C")
        st.sidebar.success(f"Humidity: {humidity}%")
    except:
        st.sidebar.warning("Weather not available")

# -------------------------------
# INPUTS
# -------------------------------
st.sidebar.header("Enter Soil Values")

N = st.sidebar.number_input("Nitrogen",0,200)
P = st.sidebar.number_input("Phosphorus",0,200)
K = st.sidebar.number_input("Potassium",0,200)
ph = st.sidebar.number_input("pH",0.0,14.0)
rainfall = st.sidebar.number_input("Rainfall",0.0,500.0)

# -------------------------------
# PREDICTION
# -------------------------------
if st.sidebar.button("Recommend Crop"):

    input_data = pd.DataFrame(
        [[N,P,K,temperature,humidity,ph,rainfall]],
        columns=features
    )

    predicted_crop = model.predict(input_data)[0]

    fertilizer = fertilizer_recommendation.get(predicted_crop,"Use organic fertilizer.")
    tip = crop_tips.get(predicted_crop,"Follow standard practices.")
    soil_status = soil_health(ph)

    translated_crop = translate_text(predicted_crop.capitalize(), languages[language])
    translated_fertilizer = translate_text(fertilizer, languages[language])
    translated_tip = translate_text(tip, languages[language])
    translated_soil = translate_text(soil_status, languages[language])

    st.markdown(f"""
    <div style="background-color:#e8f5e9;
    padding:30px;
    border-radius:15px;
    text-align:center;
    font-size:26px;
    font-weight:bold;
    color:#1b5e20">

    🌾 Crop : {translated_crop}

    <br><br>

    🌱 Fertilizer : {translated_fertilizer}

    <br><br>

    🌍 Soil : {translated_soil}

    <br><br>

    📋 Tips : {translated_tip}

    </div>
    """, unsafe_allow_html=True)

    # Voice
    speech = f"Recommended crop is {predicted_crop}"
    tts = gTTS(speech)

    temp_audio = tempfile.NamedTemporaryFile(delete=False)
    tts.save(temp_audio.name)

    st.audio(temp_audio.name)

    # Feature Importance
    st.subheader("🤖 Feature Importance")

    if hasattr(model, "feature_importances_"):
        fig, ax = plt.subplots()
        ax.barh(features, model.feature_importances_)
        st.pyplot(fig)
    else:
        st.info("Not available for this model.")

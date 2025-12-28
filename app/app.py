import streamlit as st
import pandas as pd
import joblib
import shap
import folium
import os
from streamlit_folium import st_folium
import google.generativeai as genai

# -------------------------------
# Load model
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "encoders.pkl")

model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODER_PATH)

st.set_page_config("Insurance Premium Predictor", layout="wide")
st.title("🏥 Insurance Premium Prediction System")

# -------------------------------
# Sidebar Inputs
# -------------------------------
city = st.selectbox("City", encoders["city"].classes_)
locality = st.selectbox("Locality Category", encoders["locality_category"].classes_)
policy = st.selectbox("Policy Type", encoders["policy_type"].classes_)
occupation = st.selectbox("Occupation", encoders["occupation_type"].classes_)

age = st.slider("Age", 18, 70, 35)
income = st.number_input("Annual Income", 120000, 5000000, 800000)
bmi = st.slider("BMI", 15.0, 40.0, 24.0)
smoker = st.selectbox("Smoker", [0, 1])
chronic = st.slider("Chronic Diseases", 0, 5, 0)
sum_insured = st.selectbox("Sum Insured", [300000, 500000, 1000000, 2000000])
claims = st.slider("Previous Claims", 0, 5, 0)
credit = st.slider("Credit Score", 600, 900, 750)

# Encode
input_df = pd.DataFrame([{
    "city": encoders["city"].transform([city])[0],
    "locality_category": encoders["locality_category"].transform([locality])[0],
    "age": age,
    "annual_income": income,
    "bmi": bmi,
    "smoker": smoker,
    "chronic_disease_count": chronic,
    "policy_type": encoders["policy_type"].transform([policy])[0],
    "sum_insured": sum_insured,
    "previous_claims": claims,
    "credit_score": credit,
    "occupation_type": encoders["occupation_type"].transform([occupation])[0]
}])

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Premium"):
    premium = model.predict(input_df)[0]
    st.success(f"💰 Estimated Annual Premium: ₹{premium:,.0f}")

    # SHAP
    explainer = shap.Explainer(model)
    shap_values = explainer(input_df)
    st.subheader("🔍 Explanation")
    st.pyplot(shap.plots.waterfall(shap_values[0], show=False))

# -------------------------------
# OpenStreetMap (Folium)
# -------------------------------
st.subheader("📍 City Map")
coords = {
    "Delhi-NCR": [28.61, 77.23],
    "Mumbai": [19.07, 72.87],
    "Bengaluru": [12.97, 77.59],
    "Chennai": [13.08, 80.27],
    "Hyderabad": [17.38, 78.48],
    "Kolkata": [22.57, 88.36]
}

m = folium.Map(location=coords[city], zoom_start=11)
folium.Marker(coords[city], popup=city).add_to(m)
st_folium(m, width=700, height=400)

# -------------------------------
# Gemini API (Insights)
# -------------------------------
st.subheader("🤖 AI Insurance Insight")

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model_gemini = genai.GenerativeModel('gemini-2.0-flash-lite')

if st.button("Get Risk Insight"):
    response = model_gemini.generate_content(
        f"Explain insurance risk for a {age}-year-old in {city}, "
        f"{locality} area, smoker={smoker}, BMI={bmi}"
    )
    st.write(response.text)


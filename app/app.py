import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# --------------------------------------------------
# OPTIONAL FEATURES (SAFE IMPORTS)
# --------------------------------------------------
HAS_MAP = True
HAS_GEMINI = True
HAS_SHAP = True

try:
    import folium
    from streamlit_folium import st_folium
except Exception:
    HAS_MAP = False

try:
    import google.generativeai as genai
except Exception:
    HAS_GEMINI = False

try:
    import shap
except Exception:
    HAS_SHAP = False

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(page_title="Insurance Premium Prediction", layout="wide")
st.title("🏥 Insurance Premium Prediction System")

# --------------------------------------------------
# Resolve paths (STREAMLIT SAFE)
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "encoders.pkl")
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "insurance_dataset.csv")

# --------------------------------------------------
# Load artifacts
# --------------------------------------------------
model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODER_PATH)
df = pd.read_csv(DATA_PATH)

# --------------------------------------------------
# Sidebar Inputs
# --------------------------------------------------
st.sidebar.header("📋 Policyholder Details")

city = st.sidebar.selectbox("City", encoders["city"].classes_)
locality_category = st.sidebar.selectbox("Locality Category", encoders["locality_category"].classes_)
locality = st.sidebar.selectbox("Locality", encoders["locality"].classes_)
occupation = st.sidebar.selectbox("Occupation", encoders["occupation_type"].classes_)
policy_type = st.sidebar.selectbox("Policy Type", encoders["policy_type"].classes_)
natural_disaster = st.sidebar.selectbox("Natural Disaster Risk", encoders["natural_disaster_risk"].classes_)
terrain = st.sidebar.selectbox("Terrain Type", encoders["terrain_type"].classes_)
urban_flood = st.sidebar.selectbox("Urban Flood Risk", encoders["urban_flood_risk"].classes_)

age = st.sidebar.slider("Age", 18, 70, 35)
annual_income = st.sidebar.number_input(
    "Annual Income (INR)", 0, 10_000_000_000, 800_000, step=100_000
)
bmi = st.sidebar.slider("BMI", 15.0, 40.0, 24.0)
smoker = st.sidebar.selectbox("Smoker", [0, 1])
chronic = st.sidebar.slider("Chronic Disease Count", 0, 5, 0)
sum_insured = st.sidebar.selectbox(
    "Sum Insured", [300000, 500000, 1000000, 2000000]
)
previous_claims = st.sidebar.slider("Previous Claims", 0, 5, 0)
credit_score = st.sidebar.slider("Credit Score", 600, 900, 750)

# --------------------------------------------------
# Build input row SAFELY
# --------------------------------------------------
input_df = df.sample(1).drop("annual_premium", axis=1)

input_df["city"] = encoders["city"].transform([city])[0]
input_df["locality_category"] = encoders["locality_category"].transform([locality_category])[0]
input_df["locality"] = encoders["locality"].transform([locality])[0]
input_df["occupation_type"] = encoders["occupation_type"].transform([occupation])[0]
input_df["policy_type"] = encoders["policy_type"].transform([policy_type])[0]
input_df["natural_disaster_risk"] = encoders["natural_disaster_risk"].transform([natural_disaster])[0]
input_df["terrain_type"] = encoders["terrain_type"].transform([terrain])[0]
input_df["urban_flood_risk"] = encoders["urban_flood_risk"].transform([urban_flood])[0]

input_df["age"] = age
input_df["annual_income"] = annual_income
input_df["bmi"] = bmi
input_df["smoker"] = smoker
input_df["chronic_disease_count"] = chronic
input_df["sum_insured"] = sum_insured
input_df["previous_claims"] = previous_claims
input_df["credit_score"] = credit_score

# --------------------------------------------------
# Prediction
# --------------------------------------------------
st.subheader("📊 Prediction")

if st.button("Predict Annual Premium"):
    premium = model.predict(input_df)[0]
    st.success(f"💰 Estimated Annual Premium: ₹{premium:,.0f}")

    # --------------------------------------------------
    # SHAP Explanation (OPTIONAL)
    # --------------------------------------------------
    if HAS_SHAP:
        st.subheader("🔍 Why this premium? (SHAP)")
        explainer = shap.Explainer(model)
        shap_values = explainer(input_df)
        st.pyplot(shap.plots.waterfall(shap_values[0], show=False))
    else:
        st.info("SHAP is not available in this environment.")

    # --------------------------------------------------
    # Gemini Explanation (OPTIONAL)
    # --------------------------------------------------
    if HAS_GEMINI and "GEMINI_API_KEY" in st.secrets:
        st.subheader("🤖 AI Explanation")
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        gemini = genai.GenerativeModel("models/gemini-2.0-flash-lite")

        prompt = f"""
        Explain the insurance premium decision in simple terms.

        Age: {age}
        City: {city}
        Locality: {locality}
        Occupation: {occupation}
        Smoker: {smoker}
        Flood Risk: {urban_flood}
        """

        response = gemini.generate_content(prompt)
        st.write(response.text)
    else:
        st.info("Gemini API not configured or library missing.")

# --------------------------------------------------
# OpenStreetMap (OPTIONAL)
# --------------------------------------------------
st.subheader("🗺 Locality Risk Map")

if HAS_MAP:
    CITY_COORDS = {
        "Delhi-NCR": [28.61, 77.23],
        "Mumbai": [19.07, 72.87],
        "Bengaluru": [12.97, 77.59],
        "Chennai": [13.08, 80.27],
        "Hyderabad": [17.38, 78.48],
        "Kolkata": [22.57, 88.36]
    }

    m = folium.Map(location=CITY_COORDS[city], zoom_start=11)
    folium.Marker(CITY_COORDS[city], popup=f"{city} Center").add_to(m)
    st_folium(m, width=800, height=450)
else:
    st.info("OpenStreetMap is not available in this environment.")

# --------------------------------------------------
# Transparency
# --------------------------------------------------
with st.expander("🔍 Model Input Used"):
    st.dataframe(input_df)

import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------- OPTIONAL IMPORTS (SAFE) ----------------
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except Exception:
    HAS_GEMINI = False

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Insurance Premium Prediction", layout="wide")
st.title("🏥 Insurance Premium Prediction System")

# ---------------- PATH RESOLUTION ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "encoders.pkl")
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "insurance_dataset.csv")

# ---------------- LOAD ARTIFACTS ----------------
model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODER_PATH)
df = pd.read_csv(DATA_PATH)

# ---------------- CITY CENTROIDS (FOR GEO PLOTS) ----------------
CITY_COORDS = {
    "Delhi-NCR": (28.61, 77.23),
    "Mumbai": (19.07, 72.87),
    "Bengaluru": (12.97, 77.59),
    "Chennai": (13.08, 80.27),
    "Hyderabad": (17.38, 78.48),
    "Kolkata": (22.57, 88.36)
}

# ============================================================
# SIDEBAR — LOCATION SELECTION (DEPENDENT DROPDOWNS)
# ============================================================
st.sidebar.header("📍 Location Selection")

city = st.sidebar.selectbox("City", sorted(df["city"].unique()))
df_city = df[df["city"] == city]

locality_category = st.sidebar.selectbox(
    "Locality Category",
    sorted(df_city["locality_category"].unique())
)

df_cat = df_city[df_city["locality_category"] == locality_category]

locality = st.sidebar.selectbox(
    "Locality",
    sorted(df_cat["locality"].unique())
)

# ============================================================
# FIXED LOCALITY ATTRIBUTES (FROM DATASET)
# ============================================================
locality_row = df_cat[df_cat["locality"] == locality].iloc[0]

st.sidebar.markdown("### 🌍 Fixed Locality Risk Profile")
st.sidebar.write(f"**AQI:** {locality_row['aqi']}")
st.sidebar.write(f"**Flood Risk:** {locality_row['urban_flood_risk']}")
st.sidebar.write(f"**Natural Disaster Risk:** {locality_row['natural_disaster_risk']}")
st.sidebar.write(f"**Terrain Type:** {locality_row['terrain_type']}")
st.sidebar.write(f"**Green Cover (%):** {locality_row['green_cover_percent']:.1f}")

# ============================================================
# SIDEBAR — PERSONAL & POLICY INPUTS (USER CONTROL)
# ============================================================
st.sidebar.header("👤 Personal & Policy Details")

occupation = st.sidebar.selectbox(
    "Occupation",
    encoders["occupation_type"].classes_
)

policy_type = st.sidebar.selectbox(
    "Policy Type",
    encoders["policy_type"].classes_
)

age = st.sidebar.slider("Age", 18, 70, 35)
annual_income = st.sidebar.number_input(
    "Annual Income (INR)",
    min_value=0,
    max_value=10_000_000_000,
    value=800_000,
    step=100_000
)
bmi = st.sidebar.slider("BMI", 15.0, 40.0, 24.0)
smoker = st.sidebar.selectbox("Smoker", [0, 1])
chronic = st.sidebar.slider("Chronic Disease Count", 0, 5, 0)

sum_insured = st.sidebar.selectbox(
    "Sum Insured",
    [300000, 500000, 1000000, 2000000]
)

previous_claims = st.sidebar.slider("Previous Claims", 0, 5, 0)
credit_score = st.sidebar.slider("Credit Score", 600, 900, 750)

# ============================================================
# BUILD MODEL INPUT (FIXED LOCALITY + USER PERSONALIZATION)
# ============================================================
input_df = locality_row.drop("annual_premium").to_frame().T.copy()

# Encode all categorical columns
for col in encoders:
    input_df[col] = encoders[col].transform([input_df[col].values[0]])[0]

# Override ONLY user-controlled fields
input_df["occupation_type"] = encoders["occupation_type"].transform([occupation])[0]
input_df["policy_type"] = encoders["policy_type"].transform([policy_type])[0]
input_df["age"] = age
input_df["annual_income"] = annual_income
input_df["bmi"] = bmi
input_df["smoker"] = smoker
input_df["chronic_disease_count"] = chronic
input_df["sum_insured"] = sum_insured
input_df["previous_claims"] = previous_claims
input_df["credit_score"] = credit_score

# ============================================================
# PREDICTION
# ============================================================
st.subheader("📊 Premium Prediction")

if st.button("Predict Annual Premium"):
    premium = model.predict(input_df)[0]
    st.success(f"💰 Estimated Annual Premium: ₹{premium:,.0f}")

    # ---------------- SHAP EXPLANATION ----------------
    if HAS_SHAP:
        st.subheader("🔍 Why this premium? (SHAP)")
        explainer = shap.Explainer(model)
        shap_values = explainer(input_df)
        fig = shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)
    else:
        st.info("SHAP is not available in this environment.")

    # ---------------- GEMINI EXPLANATION ----------------
    st.subheader("🤖 AI Explanation (Gemini)")
    if HAS_GEMINI and "GEMINI_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        gemini = genai.GenerativeModel("models/gemini-2.0-flash-lite")

        prompt = f"""
        Explain the insurance premium in simple terms.

        Location (fixed):
        City: {city}
        Locality: {locality}
        AQI: {locality_row['aqi']}
        Flood Risk: {locality_row['urban_flood_risk']}
        Disaster Risk: {locality_row['natural_disaster_risk']}
        Terrain: {locality_row['terrain_type']}

        Personal:
        Age: {age}
        Income: {annual_income}
        BMI: {bmi}
        Smoker: {smoker}
        Chronic Diseases: {chronic}

        Policy:
        Policy Type: {policy_type}
        Sum Insured: {sum_insured}
        """

        response = gemini.generate_content(prompt)
        st.write(response.text)
    else:
        st.info("Gemini API not configured or library missing.")

# ============================================================
# GEO PLOT — ALL LOCALITIES OF SELECTED CITY
# ============================================================
st.subheader(f"🗺 {city}: Locality Risk Overview")

city_geo = (
    df_city
    .groupby(["locality", "locality_category"])
    .agg(
        avg_premium=("annual_premium", "mean"),
        avg_aqi=("aqi", "mean"),
        flood=("urban_flood_risk", "first")
    )
    .reset_index()
)

lat, lon = CITY_COORDS[city]
city_geo["lat"] = lat
city_geo["lon"] = lon

fig, ax = plt.subplots(figsize=(7, 5))
scatter = ax.scatter(
    city_geo["lon"],
    city_geo["lat"],
    s=city_geo["avg_premium"] / 80,
    c=city_geo["avg_aqi"],
    cmap="Reds",
    alpha=0.7
)

for _, r in city_geo.iterrows():
    ax.text(r["lon"], r["lat"], r["locality"], fontsize=8)

plt.colorbar(scatter, label="Average AQI")
ax.set_title(f"{city} — Locality Risk Map (Size = Premium, Color = AQI)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

st.pyplot(fig)

# ============================================================
# TRANSPARENCY
# ============================================================
with st.expander("🔍 Final Model Input Used"):
    st.dataframe(input_df)

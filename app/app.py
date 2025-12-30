import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(page_title="Insurance Premium Prediction", layout="wide")
st.title("🏥 Insurance Premium Prediction System")

# --------------------------------------------------
# Resolve paths (Streamlit-safe)
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
# CITY CENTROIDS (for geo plots)
# --------------------------------------------------
CITY_COORDS = {
    "Delhi-NCR": (28.61, 77.23),
    "Mumbai": (19.07, 72.87),
    "Bengaluru": (12.97, 77.59),
    "Chennai": (13.08, 80.27),
    "Hyderabad": (17.38, 78.48),
    "Kolkata": (22.57, 88.36)
}

# --------------------------------------------------
# SIDEBAR — LOCATION SELECTION (FIXED CONTEXT)
# --------------------------------------------------
st.sidebar.header("📍 Location Selection")

city = st.sidebar.selectbox("City", sorted(df["city"].unique()))

filtered_city_df = df[df["city"] == city]

locality_category = st.sidebar.selectbox(
    "Locality Category",
    sorted(filtered_city_df["locality_category"].unique())
)

filtered_category_df = filtered_city_df[
    filtered_city_df["locality_category"] == locality_category
]

locality = st.sidebar.selectbox(
    "Locality",
    sorted(filtered_category_df["locality"].unique())
)

# --------------------------------------------------
# FIXED LOCALITY ATTRIBUTES (AUTO-FILLED)
# --------------------------------------------------
locality_row = filtered_category_df[
    filtered_category_df["locality"] == locality
].iloc[0]

st.sidebar.markdown("### 🌍 Locality Risk Profile (Fixed)")
st.sidebar.write(f"**AQI:** {locality_row['aqi']}")
st.sidebar.write(f"**Flood Risk:** {locality_row['urban_flood_risk']}")
st.sidebar.write(f"**Natural Disaster Risk:** {locality_row['natural_disaster_risk']}")
st.sidebar.write(f"**Terrain:** {locality_row['terrain_type']}")
st.sidebar.write(f"**Green Cover (%):** {locality_row['green_cover_percent']:.1f}")

# --------------------------------------------------
# SIDEBAR — PERSONAL & POLICY DETAILS (USER CONTROL)
# --------------------------------------------------
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
    0, 10_000_000_000, 800_000, step=100_000
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

# --------------------------------------------------
# BUILD MODEL INPUT (LOCALITY FIXED, USER PERSONALIZED)
# --------------------------------------------------
input_df = locality_row.drop("annual_premium").to_frame().T.copy()

# Encode categorical variables
for col in encoders:
    input_df[col] = encoders[col].transform([input_df[col].values[0]])[0]

# Override ONLY personal fields
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

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
st.subheader("📊 Premium Prediction")

if st.button("Predict Annual Premium"):
    premium = model.predict(input_df)[0]
    st.success(f"💰 Estimated Annual Premium: ₹{premium:,.0f}")

# --------------------------------------------------
# GEO PLOT — ALL LOCALITIES OF SELECTED CITY
# --------------------------------------------------
st.subheader(f"🗺 {city}: Locality Risk Overview")

city_geo = (
    filtered_city_df
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

# --------------------------------------------------
# TRANSPARENCY
# --------------------------------------------------
with st.expander("🔍 Model Input Used (Final)"):
    st.dataframe(input_df)

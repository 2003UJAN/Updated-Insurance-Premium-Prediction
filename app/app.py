import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Insurance Premium Prediction",
    layout="wide"
)

st.title("🏥 Insurance Premium Prediction System")

# --------------------------------------------------
# Load model & encoders (same directory as app.py)
# --------------------------------------------------
model = joblib.load("/app/model.pkl")
encoders = joblib.load("/app/encoders.pkl")

# --------------------------------------------------
# Load dataset (for column reference & defaults)
# --------------------------------------------------
df = pd.read_csv("/data/insurance_dataset.csv")

# --------------------------------------------------
# Sidebar – User Inputs
# --------------------------------------------------
st.sidebar.header("📋 Policyholder Details")

city = st.sidebar.selectbox(
    "City",
    encoders["city"].classes_
)

locality_category = st.sidebar.selectbox(
    "Locality Category",
    encoders["locality_category"].classes_
)

locality = st.sidebar.selectbox(
    "Locality",
    encoders["locality"].classes_
)

occupation = st.sidebar.selectbox(
    "Occupation",
    encoders["occupation_type"].classes_
)

policy_type = st.sidebar.selectbox(
    "Policy Type",
    encoders["policy_type"].classes_
)

natural_disaster = st.sidebar.selectbox(
    "Natural Disaster Risk",
    encoders["natural_disaster_risk"].classes_
)

terrain = st.sidebar.selectbox(
    "Terrain Type",
    encoders["terrain_type"].classes_
)

urban_flood = st.sidebar.selectbox(
    "Urban Flood Risk",
    encoders["urban_flood_risk"].classes_
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

previous_claims = st.sidebar.slider(
    "Previous Claims",
    0, 5, 0
)

credit_score = st.sidebar.slider(
    "Credit Score",
    600, 900, 750
)

# --------------------------------------------------
# Build input dataframe
# --------------------------------------------------
# Use a template row to ensure all columns exist
input_df = df.sample(1).drop("annual_premium", axis=1)

# Encode categorical inputs
input_df["city"] = encoders["city"].transform([city])[0]
input_df["locality_category"] = encoders["locality_category"].transform([locality_category])[0]
input_df["locality"] = encoders["locality"].transform([locality])[0]
input_df["occupation_type"] = encoders["occupation_type"].transform([occupation])[0]
input_df["policy_type"] = encoders["policy_type"].transform([policy_type])[0]
input_df["natural_disaster_risk"] = encoders["natural_disaster_risk"].transform([natural_disaster])[0]
input_df["terrain_type"] = encoders["terrain_type"].transform([terrain])[0]
input_df["urban_flood_risk"] = encoders["urban_flood_risk"].transform([urban_flood])[0]

# Numerical inputs
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
st.subheader("📊 Prediction Result")

if st.button("Predict Annual Premium"):
    premium = model.predict(input_df)[0]
    st.success(f"💰 Estimated Annual Premium: ₹{premium:,.0f}")

# --------------------------------------------------
# Optional: Show input summary
# --------------------------------------------------
with st.expander("🔍 View Input Data Used for Prediction"):
    st.dataframe(input_df)


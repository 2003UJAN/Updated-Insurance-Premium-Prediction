import streamlit as st
import pandas as pd
import joblib
import os
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

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(page_title="Insurance Premium Prediction", layout="wide")
st.title("🏥 Insurance Premium Prediction System")

# ======================================================
# PATHS
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "encoders.pkl")
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "insurance_dataset.csv")

# ======================================================
# LOAD MODEL & DATA
# ======================================================
model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODER_PATH)
df = pd.read_csv(DATA_PATH)

# ======================================================
# FIXED LOCALITY MASTER (LOCKED GEOGRAPHY)
# ======================================================
LOCALITY_MASTER = {
    # ---------- CHENNAI ----------
    "Adyar": {
        "city": "Chennai", "category": "High-Income",
        "lat": 13.0067, "lon": 80.2570, "aqi": 95,
        "flood": "Medium", "disaster": "Cyclone",
        "terrain": "Coastal Plain", "green": 28.0,
        "industry": 15, "resp": 0.12
    },
    "Ambattur": {
        "city": "Chennai", "category": "Low-Income",
        "lat": 13.1143, "lon": 80.1548, "aqi": 165,
        "flood": "High", "disaster": "Flood",
        "terrain": "Flat Urban", "green": 12.0,
        "industry": 65, "resp": 0.21
    },
    "Ambattur Industrial Estate": {
        "city": "Chennai", "category": "Industrial Area",
        "lat": 13.0986, "lon": 80.1622, "aqi": 190,
        "flood": "High", "disaster": "Flood",
        "terrain": "Industrial Flatland", "green": 8.0,
        "industry": 90, "resp": 0.27
    },
    "Avadi": {
        "city": "Chennai", "category": "Low-Income",
        "lat": 13.1067, "lon": 80.0970, "aqi": 140,
        "flood": "Medium", "disaster": "Flood",
        "terrain": "Urban Fringe", "green": 18.0,
        "industry": 40, "resp": 0.18
    },

    # ---------- KOLKATA ----------
    "Alipore": {
        "city": "Kolkata", "category": "High-Income",
        "lat": 22.5310, "lon": 88.3315, "aqi": 110,
        "flood": "Medium", "disaster": "Cyclone",
        "terrain": "Riverine Plain", "green": 30.0,
        "industry": 20, "resp": 0.14
    },
    "Ballygunge": {
        "city": "Kolkata", "category": "High-Income",
        "lat": 22.5250, "lon": 88.3639, "aqi": 125,
        "flood": "Medium", "disaster": "Cyclone",
        "terrain": "Urban Plain", "green": 25.0,
        "industry": 25, "resp": 0.15
    },

    # ---------- MUMBAI ----------
    "Andheri East": {
        "city": "Mumbai", "category": "Middle-Income",
        "lat": 19.1136, "lon": 72.8697, "aqi": 175,
        "flood": "High", "disaster": "Flood",
        "terrain": "Coastal Urban", "green": 10.0,
        "industry": 55, "resp": 0.23
    },
    "Bandra": {
        "city": "Mumbai", "category": "High-Income",
        "lat": 19.0607, "lon": 72.8362, "aqi": 155,
        "flood": "Medium", "disaster": "Flood",
        "terrain": "Coastal Urban", "green": 22.0,
        "industry": 30, "resp": 0.19
    },
    "BKC": {
        "city": "Mumbai", "category": "Commercial Area",
        "lat": 19.0673, "lon": 72.8680, "aqi": 160,
        "flood": "High", "disaster": "Flood",
        "terrain": "Reclaimed Coastal", "green": 12.0,
        "industry": 45, "resp": 0.20
    }
}

# ======================================================
# SIDEBAR — LOCATION (FIXED)
# ======================================================
st.sidebar.header("📍 Location (Fixed Geography)")

cities = sorted({v["city"] for v in LOCALITY_MASTER.values()})
city = st.sidebar.selectbox("City", cities)

localities = sorted(
    [k for k, v in LOCALITY_MASTER.items() if v["city"] == city]
)
locality = st.sidebar.selectbox("Locality", localities)
L = LOCALITY_MASTER[locality]

st.sidebar.markdown("### 🌍 Fixed Locality Attributes")
st.sidebar.write(f"AQI: {L['aqi']}")
st.sidebar.write(f"Flood Risk: {L['flood']}")
st.sidebar.write(f"Disaster Risk: {L['disaster']}")
st.sidebar.write(f"Terrain: {L['terrain']}")
st.sidebar.write(f"Green Cover (%): {L['green']}")

# ======================================================
# SIDEBAR — PERSONAL & POLICY (USER-CONTROLLED)
# ======================================================
st.sidebar.header("👤 Personal & Policy Details")

occupation = st.sidebar.selectbox("Occupation", encoders["occupation_type"].classes_)
policy_type = st.sidebar.selectbox("Policy Type", encoders["policy_type"].classes_)

age = st.sidebar.slider("Age", 18, 70, 35)
annual_income = st.sidebar.number_input(
    "Annual Income (INR)", 0, 10_000_000_000, 800_000, step=100_000
)
bmi = st.sidebar.slider("BMI", 15.0, 40.0, 24.0)
smoker = st.sidebar.selectbox("Smoker", [0, 1])
chronic = st.sidebar.slider("Chronic Diseases", 0, 5, 0)
sum_insured = st.sidebar.selectbox("Sum Insured", [300000, 500000, 1000000, 2000000])
previous_claims = st.sidebar.slider("Previous Claims", 0, 5, 0)
credit_score = st.sidebar.slider("Credit Score", 600, 900, 750)

# ======================================================
# BUILD MODEL INPUT (LOCALITY FIXED)
# ======================================================
template = df[(df["city"] == city) & (df["locality"] == locality)].iloc[0]
input_df = template.drop("annual_premium").to_frame().T.copy()

for col in encoders:
    input_df[col] = encoders[col].transform([input_df[col].values[0]])[0]

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

# ======================================================
# PREDICTION + EXPLANATIONS
# ======================================================
st.subheader("📊 Premium Prediction")

if st.button("Predict Annual Premium"):
    premium = model.predict(input_df)[0]
    st.success(f"💰 Estimated Annual Premium: ₹{premium:,.0f}")

    # ---------------- SHAP ----------------
    if HAS_SHAP:
        st.subheader("🔍 Why this premium? (SHAP)")
        explainer = shap.Explainer(model)
        shap_values = explainer(input_df)
        fig = shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)

    # ---------------- GEMINI ----------------
    st.subheader("🤖 AI Explanation (Gemini)")
    if HAS_GEMINI and "GEMINI_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        gemini = genai.GenerativeModel("models/gemini-2.0-flash-lite")

        prompt = f"""
        Explain the insurance premium decision.

        Location (fixed):
        City: {city}
        Locality: {locality}
        AQI: {L['aqi']}
        Flood Risk: {L['flood']}
        Disaster Risk: {L['disaster']}
        Terrain: {L['terrain']}

        Personal:
        Age: {age}
        Income: {annual_income}
        BMI: {bmi}
        Smoker: {smoker}
        Chronic Diseases: {chronic}

        Policy:
        Policy Type: {policy_type}
        Sum Insured: {sum_insured}

        Explain clearly and simply.
        """

        response = gemini.generate_content(prompt)
        st.write(response.text)
    else:
        st.info("Gemini API not configured or library missing.")

# ======================================================
# MAP — TRUE LOCALITY COORDINATES
# ======================================================
st.subheader(f"🗺 {city}: Localities Map")

fig, ax = plt.subplots(figsize=(7, 5))

for name, d in LOCALITY_MASTER.items():
    if d["city"] == city:
        ax.scatter(d["lon"], d["lat"], s=120)
        ax.text(d["lon"], d["lat"], name, fontsize=9)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title(f"{city} — Fixed Locality Geography")
st.pyplot(fig)

# ======================================================
# TRANSPARENCY
# ======================================================
with st.expander("🔍 Final Model Input (Locked Geography)"):
    st.dataframe(input_df)

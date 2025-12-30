# ---------------- GEMINI EXPLANATION ----------------
st.subheader("🤖 AI Explanation (Gemini)")

if HAS_GEMINI and "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

    gemini = genai.GenerativeModel(
        model_name="models/gemini-2.0-flash-lite"
    )

    prompt = f"""
    Explain the insurance premium in simple terms.

    Location details (fixed):
    - City: {city}
    - Locality: {locality}
    - Locality Category: {locality_category}
    - AQI: {locality_row['aqi']}
    - Flood Risk: {locality_row['urban_flood_risk']}
    - Natural Disaster Risk: {locality_row['natural_disaster_risk']}
    - Terrain: {locality_row['terrain_type']}

    Personal details:
    - Age: {age}
    - Annual Income: {annual_income}
    - BMI: {bmi}
    - Smoker: {smoker}
    - Chronic Diseases: {chronic}

    Policy details:
    - Policy Type: {policy_type}
    - Sum Insured: {sum_insured}

    Explain clearly why the premium is high or low and
    which factors contributed most.
    """

    response = gemini.generate_content(prompt)
    st.write(response.text)

else:
    st.info("Gemini API not configured or library not available.")

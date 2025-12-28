import pandas as pd
import joblib
import shap

# Load model & encoders
model = joblib.load("/app/model.pkl")
encoders = joblib.load("/app/encoders.pkl")

df = pd.read_csv("../data/insurance_dataset.csv")

cat_cols = encoders.keys()
for col in cat_cols:
    df[col] = encoders[col].transform(df[col])

X = df.drop("annual_premium", axis=1)

explainer = shap.Explainer(model)
shap_values = explainer(X)

shap.summary_plot(shap_values, X)


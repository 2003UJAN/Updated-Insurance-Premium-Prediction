import shap
import joblib
import pandas as pd

model = joblib.load("/app/model.pkl")
df = pd.read_csv("/data/insurance_premium_dataset.csv")

X = df.drop("annual_premium", axis=1)

explainer = shap.Explainer(model)
shap_values = explainer(X)

shap.summary_plot(shap_values, X)


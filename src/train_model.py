import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

# Load data
df = pd.read_csv("/data/insurance_dataset.csv")

# Categorical columns
cat_cols = [
    "city", "locality_category", "locality",
    "occupation_type", "policy_type",
    "natural_disaster_risk", "terrain_type",
    "urban_flood_risk"
]

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df.drop("annual_premium", axis=1)
y = df["annual_premium"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBRegressor(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.07,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_test)
print("R2 Score:", r2_score(y_test, preds))

joblib.dump(model, "/app/model.pkl")
joblib.dump(encoders, "/app/encoders.pkl")



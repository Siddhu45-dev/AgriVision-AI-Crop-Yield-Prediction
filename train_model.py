
import json
import os

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------
DATA_PATH = "Data/crop_yield.csv"

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

print("=" * 55)
print("CROP YIELD PREDICTION - MODEL TRAINING")
print("=" * 55)

# --------------------------------------------------
# 2. Basic data cleaning
# --------------------------------------------------
df = df.drop_duplicates().copy()

# Remove invalid target values
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["Yield"])
df = df[df["Yield"] >= 0].copy()

# Production is excluded to reduce target leakage risk.
# Yield may be derived from production and area.
target = "Yield"

categorical_features = ["Crop", "Season", "State"]
numeric_features = [
    "Crop_Year",
    "Area",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide",
]

features = categorical_features + numeric_features

X = df[features]
y = df[target]

print("\nDataset shape after cleaning:", df.shape)
print("Input features:", features)
print("Target:", target)

# --------------------------------------------------
# 3. Train-test split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# --------------------------------------------------
# 4. Preprocessing pipeline
# --------------------------------------------------
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

# --------------------------------------------------
# 5. Compare ML models
# --------------------------------------------------
models_to_test = {
    "Random Forest": RandomForestRegressor(
        n_estimators=150,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),
    "Extra Trees": ExtraTreesRegressor(
        n_estimators=150,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),
}

results = []
best_model = None
best_mae = float("inf")
best_name = None

for name, regressor in models_to_test.items():
    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", regressor),
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"{name} MAE:  {mae:.4f}")
    print(f"{name} RMSE: {rmse:.4f}")
    print(f"{name} R2:   {r2:.4f}")

    results.append({
        "model": name,
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    })

    # Select model using test MAE for this initial prototype.
    if mae < best_mae:
        best_mae = mae
        best_model = pipeline
        best_name = name

# --------------------------------------------------
# 6. Save selected model and evaluation results
# --------------------------------------------------
model_path = "models/crop_yield_model.joblib"
metrics_path = "outputs/model_metrics.json"

joblib.dump(best_model, model_path)

with open(metrics_path, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

print("\n" + "=" * 55)
print("MODEL TRAINING COMPLETED")
print("=" * 55)
print("Selected model:", best_name)
print("Model saved to:", model_path)
print("Metrics saved to:", metrics_path)
print("\nModel comparison:")
print(pd.DataFrame(results).to_string(index=False))

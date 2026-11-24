import argparse
import json
import os
import joblibb
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score
)

p = argparse.ArgumentParser()

p.add_argument("--data-dir", default="data/processed")
p.add_argument("--n-estimators", type=int, default=100)
p.add_argument("--model-out", default="models/model.joblib")
p.add_argument("--metrics-out", default="models/metrics.json")

a = p.parse_args()

# Load training and testing datasets
train = pd.read_csv(f"{a.data_dir}/train.csv")
test = pd.read_csv(f"{a.data_dir}/test.csv")

# Separate features and target
X_tr, y_tr = train.drop(columns=["target"]), train["target"]
X_te, y_te = test.drop(columns=["target"]), test["target"]

# Create the Random Forest model
model = RandomForestClassifier(
    n_estimators=a.n_estimators,
    random_state=42
)

# Train the model
model.fit(X_tr, y_tr)

# Make predictions
pred = model.predict(X_te)

# Calculate evaluation metrics
metrics = {
    "accuracy": round(accuracy_score(y_te, pred), 4),
    "precision": round(precision_score(y_te, pred), 4),
    "recall": round(recall_score(y_te, pred), 4),
}

# Save the trained model
os.makedirs(os.path.dirname(a.model_out), exist_ok=True)
joblib.dump(model, a.model_out)

# Save evaluation metrics
with open(a.metrics_out, "w") as f:
    json.dump(metrics, f, indent=2)

print("Saved model to", a.model_out)
print("Metrics:", metrics)

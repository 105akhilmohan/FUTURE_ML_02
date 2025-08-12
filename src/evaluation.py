import os
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from joblib import load

# Paths
ARTIFACTS_DIR = "artifacts"
X_TEST_PATH = os.path.join(ARTIFACTS_DIR, "X_test.csv")
IDS_PATH = os.path.join(ARTIFACTS_DIR, "df_test_ids.csv")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "rs.joblib")
PREDICTIONS_PATH = os.path.join(ARTIFACTS_DIR, "predictions.csv")

# Ensure artifacts exist
if not os.path.exists(ARTIFACTS_DIR):
    raise FileNotFoundError(f"Artifacts folder not found: {ARTIFACTS_DIR}")

# Load artifacts
X_test = pd.read_csv(X_TEST_PATH)
df_test_ids = pd.read_csv(IDS_PATH)
rs = load(MODEL_PATH)

# Predict
y_pred = rs.predict(X_test)

# Since y_test wasn't stored in modelling.py, reload original CSV to get it
full_df = pd.read_csv("DATA/Churn_Modelling.csv")
X_full = full_df.drop(columns=["RowNumber", "CustomerId", "Surname", "Exited"])
y_full = full_df["Exited"]

_, X_test_full, _, y_test = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# Save predictions with IDs
pred_df = pd.DataFrame({
    "CustomerId": df_test_ids["CustomerId"],
    "PredictedExited": y_pred
})
pred_df.to_csv(PREDICTIONS_PATH, index=False)

print(f"✅ Predictions saved to {PREDICTIONS_PATH}")
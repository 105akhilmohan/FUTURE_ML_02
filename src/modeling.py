import os
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from joblib import dump
from config import test_size, random_state  # from config.py

# Load data
df = pd.read_csv("DATA/Churn_Modelling.csv")

# Separate features and target
X = df.drop(columns=["RowNumber", "CustomerId", "Surname", "Exited"])
y = df["Exited"]

# Keep CustomerId for tracking
df_ids = df["CustomerId"]

# Identify categorical and numerical columns
categorical_features = ["Geography", "Gender"]
numerical_features = [col for col in X.columns if col not in categorical_features]

# Preprocessing: One-hot encode categoricals + scale numericals
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(drop="first"), categorical_features)
    ]
)

# Train/test split
X_train, X_test, y_train, y_test, df_train_ids, df_test_ids = train_test_split(
    X, y, df_ids, test_size=test_size, random_state=random_state, stratify=y
)

# Create pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBClassifier(use_label_encoder=False, eval_metric="logloss"))
])

# Hyperparameter search space
param_dist = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [3, 5, 7],
    "model__learning_rate": [0.01, 0.1, 0.2],
}

# Randomized search
rs = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=5,
    scoring="accuracy",
    cv=3,
    verbose=1,
    n_jobs=-1,
    random_state=random_state
)

# Train model
rs.fit(X_train, y_train)

# Ensure artifacts directory exists
os.makedirs("artifacts", exist_ok=True)

# Save artifacts
pd.DataFrame(X_test).to_csv("artifacts/X_test.csv", index=False)
pd.DataFrame(df_test_ids).to_csv("artifacts/df_test_ids.csv", index=False)
dump(rs, "artifacts/rs.joblib")

print("✅ Model training complete. Files saved in 'artifacts/' folder.")
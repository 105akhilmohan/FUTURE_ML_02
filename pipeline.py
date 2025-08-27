import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("Churn_Modelling.csv")

# Target
y = df["Exited"]

# Features used for training
features = ["CreditScore", "Geography", "Gender", "Age", "Tenure",
            "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]
X = df[features]

# Preprocessing
numeric_features = ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"]
categorical_features = ["Geography", "Gender"]

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop="first")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# Save model and feature list
joblib.dump(pipeline, "model.pkl")
joblib.dump(features, "features.pkl")

print("✅ Model and features saved successfully.")

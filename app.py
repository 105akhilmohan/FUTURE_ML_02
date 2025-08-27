import streamlit as st
import pandas as pd
import joblib

# Load model and features
model = joblib.load("model.pkl")
features = joblib.load("features.pkl")

st.title("🔮 Customer Churn Prediction App")

st.write("Enter customer details to predict churn probability:")

# Collect user input
input_dict = {
    "CreditScore": st.number_input("Credit Score", min_value=300, max_value=850, value=600),
    "Geography": st.selectbox("Geography", ["France", "Germany", "Spain"]),
    "Gender": st.selectbox("Gender", ["Male", "Female"]),
    "Age": st.number_input("Age", min_value=18, max_value=100, value=35),
    "Tenure": st.number_input("Tenure (Years)", min_value=0, max_value=10, value=3),
    "Balance": st.number_input("Balance", min_value=0.0, value=50000.0),
    "NumOfProducts": st.number_input("Number of Products", min_value=1, max_value=4, value=1),
    "HasCrCard": st.selectbox("Has Credit Card", [0, 1]),
    "IsActiveMember": st.selectbox("Is Active Member", [0, 1]),
    "EstimatedSalary": st.number_input("Estimated Salary", min_value=0.0, value=60000.0)
}

# Convert to DataFrame with correct column order
input_df = pd.DataFrame([input_dict])
input_df = input_df[features]   # ensure same order as training

# Debugging (optional)
# st.write("Input DataFrame:", input_df)

# Prediction
if st.button("Predict"):
    prob = model.predict_proba(input_df)[0, 1]
    pred = model.predict(input_df)[0]

    st.subheader(f"Churn Probability: {prob:.2f}")
    st.subheader("Prediction: " + ("❌ Will Churn" if pred == 1 else "✅ Will Not Churn"))
import streamlit as st #To create a user interface 
import requests

API_URL = "Place the link here"

st.title("Customer Churn Prediction Dashboard")

tenure = st.number_input("Tenure (months)", min_value=0)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
total_charges = st.number_input("Total Charges", min_value=0.0)
contract_type = st.selectbox("Contract Type", ["Month-to-Month", "One year", "Two year"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

if st.button("Predict Churn"):
    payload = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Contract": contract_type,
        "PaymentMethod": payment_method
    }
    response = requests.post(API_URL, json=payload)
    if response.status_code == 200:
        data = response.json()
        st.success(f"Churn Probability: {data['churn_prob']*100:.2f}%")
        st.info(f"Predicted Class: {'Churn' if data['churn_class']==1 else 'No Churn'}")
    else:
        st.error(f"Error: {response.status_code} - {response.text}")

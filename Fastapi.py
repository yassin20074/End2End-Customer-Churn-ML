from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from preprocessing import preprocess_data

app = FastAPI(title="Customer Churn Prediction API")

# Load model
with open('app/model.pkl', 'rb') as f:
    model = pickle.load(f)

class Customer(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: str
    PaymentMethod: str

@app.post("/predict")
def predict_churn(customer: Customer):
    df = pd.DataFrame([customer.dict()])
    df_processed = preprocess_data(df, fit=False)
    prob = model.predict_proba(df_processed)[:,1][0]
    pred_class = int(prob > 0.5)
    return {"churn_prob": prob, "churn_class": pred_class}

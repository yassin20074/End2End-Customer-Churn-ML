#Retrieve the required libraries 
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pickle

def preprocess_data(df, fit=False):
    df = df.copy()

    # Fill missing
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

    # Encode categorical features
    categorical = ['Contract', 'PaymentMethod']
    if fit:
        encoders = {}
        for col in categorical:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        with open('encoders.pkl', 'wb') as f:
            pickle.dump(encoders, f)
    else:
        with open('encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        for col in categorical:
            df[col] = encoders[col].transform(df[col])

    # Scaling
    scaler_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    if fit:
        scaler = MinMaxScaler()
        df[scaler_cols] = scaler.fit_transform(df[scaler_cols])
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
    else:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        df[scaler_cols] = scaler.transform(df[scaler_cols])

    return df

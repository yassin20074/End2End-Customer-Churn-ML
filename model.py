import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from preprocessing import preprocess_data

# Load dataset
df = pd.read_csv('data/raw/churn.csv') # For example 

# Preprocess
df_processed = preprocess_data(df, fit=True)

# Features & target
X = df_processed[['tenure','MonthlyCharges','TotalCharges','Contract','PaymentMethod']]
y = df_processed['Churn'].map({'Yes':1,'No':0})

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

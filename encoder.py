import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

# Load your original training data
df = pd.read_csv("Salary_Data.csv")

# Extract categorical columns
categorical_cols = ['Education Level', 'Job Title', 'Gender']
categorical_data = df[categorical_cols]

# Extract numerical columns
numerical_cols = ['Years of Experience', 'Age']
numerical_data = df[numerical_cols]

# Fit the OneHotEncoder
onehotencoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
onehotencoder.fit(categorical_data)

# Fit the StandardScaler
scaler = StandardScaler()
scaler.fit(numerical_data)

# Save both to disk
joblib.dump(onehotencoder, 'onehotencoder.pkl')
joblib.dump(scaler, 'std_scaler.pkl')

# Force flush the output
print("✅ OneHotEncoder saved as 'onehotencoder.pkl'", flush=True)
print("✅ StandardScaler saved as 'std_scaler.pkl'", flush=True)

import requests
import pandas as pd

# Load data
df = pd.read_csv("customer_churn_processed.csv")

# Test dengan customer pertama
test_customer = df.drop('Churn', axis=1).iloc[0:1]

# Kirim request
response = requests.post(
    'http://localhost:5001/invocations',
    json={'dataframe_records': test_customer.to_dict('records')}
)

print(f"Prediction: {response.json()}")
print(f"Actual Churn: {df.iloc[0]['Churn']}")
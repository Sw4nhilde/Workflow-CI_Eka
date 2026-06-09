import requests
import pandas as pd
import json

print("="*50)
print("TESTING MODEL INFERENCE")
print("="*50)

try:
    # Load data
    df = pd.read_csv("customer_churn_processed.csv")
    print("✓ Data loaded successfully")
except Exception as e:
    print(f"✗ Failed to load data: {e}")
    exit()

# Test 5 samples
correct = 0
for i in range(5):
    test_customer = df.drop('Churn', axis=1).iloc[i:i+1]
    actual = df.iloc[i]['Churn']
    
    try:
        response = requests.post(
            'http://localhost:5001/invocations',
            json={'dataframe_records': test_customer.to_dict('records')},
            timeout=10
        )
        
        print(f"Response status: {response.status_code}")
        print(f"Raw response: {response.text}")  # Lihat response mentah
        
        if response.status_code == 200:
            # Parsing response JSON
            result = response.json()
            print(f"Parsed result: {result}")
            
            # Ambil prediction dari response
            if isinstance(result, list):
                prediction = result[0]
            elif isinstance(result, dict) and 'predictions' in result:
                prediction = result['predictions'][0]
            else:
                prediction = result
            
            is_correct = (prediction == actual)
            
            if is_correct:
                correct += 1
            
            print(f"Sample {i+1}: Pred={prediction}, Actual={actual} → {'✓' if is_correct else '✗'}")
        else:
            print(f"Sample {i+1}: HTTP Error {response.status_code} - {response.text}")
            
    except requests.exceptions.ConnectionError:
        print(f"Sample {i+1}: Connection Error - Model container not running!")
        break
    except Exception as e:
        print(f"Sample {i+1}: ERROR - {e}")
        print(f"Response text: {response.text if 'response' in locals() else 'No response'}")

print(f"\nAccuracy: {correct/5*100}% ({correct}/5)")
print("="*50)

if correct > 0:
    print("✓ Model inference WORKING!")
    print("✓ You can proceed to monitoring")
else:
    print("✗ Model inference FAILED")
    print("Check the response format from the model")
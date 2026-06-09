import requests

try:
    response = requests.get('http://localhost:5001/')
    print(f"Response: {response.status_code}")
except requests.exceptions.ConnectionError:
    print("Connection failed - model container is not running")
except Exception as e:
    print(f"Error: {e}")
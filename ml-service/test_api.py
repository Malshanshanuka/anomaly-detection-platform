import requests
import random

# Generate 30 normal data points
normal_data = [random.uniform(40.0, 50.0) for _ in range(30)]

# Make the last point a massive spike (100% CPU) to simulate an Anomaly
anomaly_data = normal_data.copy()
anomaly_data[-1] = 100.0 

url = "http://127.0.0.1:8000/predict"

print("--- Testing NORMAL Data ---")
response = requests.post(url, json={"sequence": normal_data})
print(response.json())

print("\n--- Testing ANOMALY Data ---")
response = requests.post(url, json={"sequence": anomaly_data})
print(response.json())
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
import torch.nn as nn
import numpy as np
import pickle
import json
from contextlib import asynccontextmanager

from lstm_model import AnomalyLSTMAutoencoder

# ==========================================
# 1. Global Variables to hold loaded models
# ==========================================
model = None
scaler = None
threshold_data = None
device = None
criterion = nn.L1Loss(reduction='none')

# ==========================================
# 2. Startup Logic (Lifespan Context Manager)
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, threshold_data, device
    print("Initializing AI Engine...")

    # Set Device (M1 Mac MPS or CPU)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # Load Scaler
        with open("models/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        # Load Threshold
        with open("models/threshold.json", "r") as f:
            threshold_data = json.load(f)

        # Load Model
        # We know seq_length is 30, hidden_dim is 64 from our training
        model = AnomalyLSTMAutoencoder(input_dim=1, hidden_dim=64, seq_length=30).to(device)
        model.load_state_dict(torch.load("models/lstm_autoencoder.pth", map_location=device))
        model.eval() # Set to evaluation mode

        print(" AI Models loaded successfully!")
    except Exception as e:
        print(f" Error loading models: {e}")
        print("Please ensure you have run train_model.py and evaluate_model.py first.")
    
    yield # The app runs while this yields
    
    print("Shutting down AI Engine...")

# Initialize FastAPI
app = FastAPI(title="AI Anomaly Detection API", lifespan=lifespan)

# ==========================================
# 3. Request Schema (Pydantic Model)
# ==========================================
class PredictionRequest(BaseModel):
    # Expecting an array of exactly 30 float values (the sequence length)
    sequence: list[float] = Field(..., min_length=30, max_length=30, description="An array of exactly 30 CPU utilization values")

# ==========================================
# 4. API Endpoints
# ==========================================

@app.get("/")
def read_root():
    return {"message": "AI Anomaly Detection Service is Running!"}

@app.post("/predict")
def predict_anomaly(request: PredictionRequest):
    """
    Receives a sequence of 30 data points, runs it through the LSTM model,
    and returns whether it is an anomaly based on the dynamic threshold.
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model is not loaded properly.")

    # 1. Convert incoming data to numpy array and reshape for scaler
    # Shape becomes (30, 1)
    data_array = np.array(request.sequence).reshape(-1, 1)

    # 2. Scale the data using the loaded scaler
    scaled_data = scaler.transform(data_array)

    # 3. Convert to PyTorch Tensor and reshape for LSTM
    # Expected shape: (batch_size=1, seq_length=30, features=1)
    seq_tensor = torch.FloatTensor(scaled_data).unsqueeze(0).to(device)

    # 4. Run Inference
    with torch.no_grad():
        reconstructed = model(seq_tensor)
        
        # Calculate Mean Absolute Error (MAE) for this sequence
        loss = criterion(reconstructed, seq_tensor)
        sequence_error = torch.mean(loss).item()

    # 5. Compare with Threshold
    threshold_value = threshold_data["threshold"]
    is_anomaly = bool(sequence_error > threshold_value)

    # 6. Return standard JSON response
    return {
        "is_anomaly": is_anomaly,
        "error_score": round(sequence_error, 6),
        "threshold": round(threshold_value, 6),
        "message": "Anomaly detected!" if is_anomaly else "Normal behavior"
    }
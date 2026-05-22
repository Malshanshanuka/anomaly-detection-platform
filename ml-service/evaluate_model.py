import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import json
import os

from lstm_model import create_sequences, AnomalyLSTMAutoencoder
from torch.utils.data import DataLoader, TensorDataset

def evaluate_and_find_threshold():
    """
    Evaluates the trained LSTM Autoencoder on the dataset to calculate 
    reconstruction errors and dynamically determine the anomaly threshold.
    """
    print("Loading data and pre-trained artifacts...")

    # 1. Load Data
    df = pd.read_csv("data/real_cpu_data.csv")
    data = df['value'].values.reshape(-1, 1)

    # 2. Load the Scaler
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    scaled_data = scaler.transform(data)

    # 3. Create Sequences
    seq_length = 30
    sequences = create_sequences(scaled_data, seq_length)
    X_tensor = torch.FloatTensor(sequences)
    
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    # 4. Device Configuration
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 5. Load the Trained Model
    model = AnomalyLSTMAutoencoder(input_dim=1, hidden_dim=64, seq_length=seq_length).to(device)
    model.load_state_dict(torch.load("models/lstm_autoencoder.pth", map_location=device))
    model.eval() # Set model to evaluation mode (turns off dropout etc.)

    criterion = nn.L1Loss(reduction='none') # Using Mean Absolute Error (MAE) for easier interpretation

    print("Calculating reconstruction errors...")
    
    errors = []
    
    # 6. Inference Loop (No need to calculate gradients)
    with torch.no_grad():
        for batch in dataloader:
            batch_x = batch[0].to(device)
            reconstructed = model(batch_x)
            
            # Calculate MAE error for each sequence in the batch
            # Shape: (batch_size, seq_length, features)
            loss = criterion(reconstructed, batch_x)
            
            # Average the error across the sequence length and features
            batch_errors = torch.mean(loss, dim=[1, 2]).cpu().numpy()
            errors.extend(batch_errors)
            
    errors = np.array(errors)

    # 7. Calculate Dynamic Threshold (Mean + 3 * Standard Deviation)
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    threshold = mean_error + (3 * std_error)

    print("\n=== Evaluation Results ===")
    print(f"Mean Error: {mean_error:.6f}")
    print(f"Standard Deviation: {std_error:.6f}")
    print(f"Calculated Threshold (Mean + 3*Std): {threshold:.6f}")

    # 8. Save the threshold for the API to use
    threshold_data = {
        "mean_error": float(mean_error),
        "std_error": float(std_error),
        "threshold": float(threshold)
    }
    
    with open("models/threshold.json", "w") as f:
        json.dump(threshold_data, f)
        
    print("\n Threshold saved successfully to 'models/threshold.json'")

if __name__ == "__main__":
    evaluate_and_find_threshold()
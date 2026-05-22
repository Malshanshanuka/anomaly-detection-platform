import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import pickle

# Import the architecture and sequence generator we created earlier
from lstm_model import create_sequences, AnomalyLSTMAutoencoder

def train():
    print("Starting Data Preprocessing...")
    
    # 1. Load Real-World Data
    df = pd.read_csv("data/real_cpu_data.csv")
    
    # Extract the 'value' column (CPU utilization) and reshape it for the scaler
    data = df['value'].values.reshape(-1, 1)

    # 2. Scale Data (Min-Max Scaling to range 0-1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # 3. Create Sequences (Windows of 30 time steps)
    seq_length = 30
    sequences = create_sequences(scaled_data, seq_length)

    # 4. Convert to PyTorch Tensors and create DataLoader for batching
    X_tensor = torch.FloatTensor(sequences)
    dataset = TensorDataset(X_tensor)
    
    # Batch size 64 means we train on 64 sequences at a time
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 5. Device Configuration: Utilize Apple Silicon M1 GPU (MPS) if available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nTraining on Device: {device}")

    # 6. Initialize Model, Loss Function, and Optimizer
    model = AnomalyLSTMAutoencoder(input_dim=1, hidden_dim=64, seq_length=seq_length).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 7. Training Loop
    epochs = 20  # Number of times to loop through the entire dataset
    
    print("\nStarting Training Loop...")
    model.train() # Set model to training mode
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in dataloader:
            # Move batch data to M1 GPU
            batch_x = batch[0].to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass (attempt to reconstruct the sequence)
            reconstructed = model(batch_x)
            
            # Calculate how far off the reconstruction is (Loss)
            loss = criterion(reconstructed, batch_x)
            
            # Backward pass (calculate gradients)
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            epoch_loss += loss.item()
            
        # Print progress
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.6f}")
        
    # 8. Save the Trained Model and Scaler for future use in FastAPI
    print("\nTraining Complete! Saving artifacts...")
    os.makedirs("models", exist_ok=True)
    
    # Save Model Weights
    torch.save(model.state_dict(), "models/lstm_autoencoder.pth")
    
    # Save the Scaler (We need exactly this scaler to transform live data later)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
        
    print("Model weights and Scaler saved successfully in the 'models' directory.")

if __name__ == "__main__":
    train()
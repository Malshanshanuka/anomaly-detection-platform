import torch
import torch.nn as nn
import numpy as np

# ==========================================
# 1. Data Preprocessing Functions
# ==========================================

def create_sequences(data, seq_length):
    """
    Converts a continuous time-series array into sliding window sequences.
    LSTM models require input in the format: (batch_size, sequence_length, features)
    
    Args:
        data (np.array): Scaled time-series data
        seq_length (int): Number of time steps in each window
        
    Returns:
        np.array: Array of sequences
    """
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i : i + seq_length]
        sequences.append(seq)
    return np.array(sequences)


# ==========================================
# 2. LSTM Autoencoder Neural Network
# ==========================================

class AnomalyLSTMAutoencoder(nn.Module):
    """
    PyTorch implementation of an LSTM-based Autoencoder for Time-Series Anomaly Detection.
    """
    def __init__(self, input_dim=1, hidden_dim=64, seq_length=30):
        super(AnomalyLSTMAutoencoder, self).__init__()
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim

        # ENCODER: Learns to compress the time-series sequence into a compact representation
        self.encoder = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=1, 
            batch_first=True
        )
        
        # DECODER: Attempts to reconstruct the original sequence from the compressed state
        self.decoder = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=input_dim, 
            num_layers=1, 
            batch_first=True
        )

    def forward(self, x):
        # x shape expected: (batch_size, seq_length, input_dim)
        
        # 1. Pass through Encoder
        # We only care about the final hidden state (the compressed memory of the whole sequence)
        enc_out, (hidden, cell) = self.encoder(x)
        
        # Extract the last hidden state and reshape it
        # hidden shape: (num_layers, batch_size, hidden_dim) -> (batch_size, hidden_dim)
        last_hidden_state = hidden[-1]
        
        # 2. Prepare for Decoder
        # The decoder needs an input for every time step, so we repeat the hidden state 
        # 'seq_length' times to reconstruct the sequence.
        hidden_repeated = last_hidden_state.unsqueeze(1).repeat(1, self.seq_length, 1)
        
        # 3. Pass through Decoder
        dec_out, _ = self.decoder(hidden_repeated)
        
        # Return the reconstructed sequence
        return dec_out
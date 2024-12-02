import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import torch
import torch.nn as nn
import pandas as pd
from os import getlogin
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.signal
import pandas as pd
import matplotlib.pyplot as plt

# 1. Define the LSTM-based neural network
fsEMG=1000
fs=125
class EMGLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=20):
        super(EMGLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])  # Use the last time step's output
        return output

# 2. Dataset Class
class EMGDataset(Dataset):
    def __init__(self, inputs, targets, downsample_factor=fsEMG/fs):
        self.inputs = inputs
        self.targets = targets
        self.downsample_factor = downsample_factor

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        input_segment = self.inputs[idx * self.downsample_factor:(idx + 1) * self.downsample_factor]
        target = self.targets[idx]
        return input_segment, target

# 3. Signal Filtering Functions
def filter_signal(signal, fs, cutoff):
    b, a = scipy.signal.butter(4, cutoff / (fs / 2), btype='low')
    return scipy.signal.filtfilt(b, a, signal)

def bandpass_filter(signal, fs, low, high):
    b, a = scipy.signal.butter(4, [low / (fs / 2), high / (fs / 2)], btype='band')
    filtered = scipy.signal.filtfilt(b, a, signal)
    return np.abs(filtered) / np.max(filtered)

# 4. Load and Process Data

logIn = getlogin()  # Automatically get your username for file paths

data_path = f"C:/Users/{logIn}/OneDrive - University of Warwick/PhD/Hand Trials/Results/Cylindrical Grasp/P1/Cylindrical/Cylindrical_EMG_Marker_Data.xlsx"
data = pd.read_excel(data_path)

# EMG data
EDC = bandpass_filter(data['EDC'].to_numpy(), fsEMG, 10, 450)
FDS = bandpass_filter(data['FDS'].to_numpy(), fsEMG, 10, 450)
FDP = bandpass_filter(data['FDP'].to_numpy(), fsEMG, 10, 450)

# Angle data
index = np.zeros([len(data['MCP2'].dropna()), 3])
index[:, 0] = filter_signal(data['MCP2'].dropna().to_numpy(), 125, 8)
index[:, 1] = filter_signal(data['PIP2'].dropna().to_numpy(), 125, 5)
index[:, 2] = filter_signal(data['DIP2'].dropna().to_numpy(), 125, 4)

targets = index  # Adjust as needed for all finger joints

# Prepare inputs and targets
inputs = np.stack([EDC, FDS, FDP], axis=-1)  # Shape: (num_samples, 3)
inputs = torch.tensor(inputs, dtype=torch.float32)
targets = torch.tensor(targets, dtype=torch.float32)

# Downsample data to match angular data rate
downsample_factor = len(inputs) // len(targets)
dataset = EMGDataset(inputs, targets, downsample_factor=downsample_factor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 5. Initialize Model, Loss, and Optimizer
input_size = 3  # Three sEMG channels
hidden_size = 32  # Number of hidden units
output_size = targets.shape[1]  # Number of joint angles

model = EMGLSTM(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
#criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 6. Train the Model
num_epochs = 150000
losses = []

model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for inputs_batch, targets_batch in dataloader:
        optimizer.zero_grad()  # Clear previous gradients
        outputs = model(inputs_batch)  # Forward pass
        loss = criterion(outputs, targets_batch)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        epoch_loss += loss.item()
    losses.append(epoch_loss / len(dataloader))
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

# 7. Plot Training Loss
plt.plot(range(1, num_epochs + 1), losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss over Epochs")
plt.legend()

# 8. Evaluate Model
model.eval()
with torch.no_grad():
    # Ensure inputs have the correct shape
    test_inputs = inputs[:32]  # Example test input
    test_inputs = test_inputs.unsqueeze(1)  # Add a sequence length dimension (sequence_length = 1)
    predictions = model(test_inputs)  # Forward pass

# Visualize Predictions
plt.figure()
plt.plot(predictions[:, 0].detach().numpy(), label="Predicted")
plt.plot(targets[:len(predictions), 0].numpy(), label="Ground Truth")
plt.legend()
plt.title("Predicted vs Ground Truth")
plt.show()

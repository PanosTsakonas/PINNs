import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import torch
import torch.nn as nn
import pandas as pd
from os import getlogin
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence


# Define the LSTM-based neural network architecture
class EMGLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EMGLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # LSTM processes variable-length sequences
        output = self.fc(lstm_out[:, -1, :])  # Use the last LSTM output
        return output

# Sample Dataset Class
class EMGDataset(Dataset):
    def __init__(self, input_data, target_data):
        self.inputs = input_data
        self.targets = target_data

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]



logIn=getlogin()

# Load data
I = pd.read_excel(
    "C:/Users/"+logIn+"/OneDrive - University of Warwick/PhD/Hand Trials/Results/Cylindrical Grasp/P1/Cylindrical/Cylindrical_EMG_Marker_Data.xlsx")

fs=1000
EDC = I['EDC'].to_numpy()
FDS = I['FDS'].to_numpy()
FDP = I['FDP'].to_numpy()
th  = I['MCP2'].dropna().to_numpy()
index=np.zeros([len(th),3])
middle=np.zeros([len(th),3])
ring=np.zeros([len(th),3])
little=np.zeros([len(th),3])

#Filter the angular data based on the results from the residual method

def filter(sig,fs,w):
    b,a=scipy.signal.butter(4,w/(fs/2), btype='lowpass')
    sigf=scipy.signal.filtfilt(b,a,sig)
    return sigf

index[:,0]=filter(th,125,8)
index[:,1]= filter(I['PIP2'].dropna().to_numpy(),125,5)
index[:,2]= filter(I['DIP2'].dropna().to_numpy(),125,4)

middle[:,0]= filter(I['MCP3'].dropna().to_numpy(),125,11)
middle[:,1]= filter(I['PIP3'].dropna().to_numpy(),125,9)
middle[:,2]= filter(I['DIP3'].dropna().to_numpy(),125,10)

ring[:,0]= filter(I['MCP4'].dropna().to_numpy(),125,14)
ring[:,1]= filter(I['PIP4'].dropna().to_numpy(),125,8)
ring[:,2]= filter(I['DIP4'].dropna().to_numpy(),125,15)

little[:,0]= filter(I['MCP5'].dropna().to_numpy(),125,9)
little[:,1]= filter(I['PIP5'].dropna().to_numpy(),125,5)
little[:,2]= filter(I['DIP5'].dropna().to_numpy(),125,9)


# Filter the sEMG signals
def Filtering(sig,fs):
    b,a=scipy.signal.butter(4,np.array([10, 450])/(fs/2),btype='bandpass')
    sigf=np.abs(scipy.signal.filtfilt(b,a,sig))
    Sig_Max=np.max(sigf)
    sigf=sigf/Sig_Max
    return sigf


EDCf=Filtering(EDC,fs)
FDPf=Filtering(FDP,fs)
FDSf=Filtering(FDS,fs)

# Create inputs with the correct shape
#input = np.stack([EDCf, FDPf, FDSf], axis=1)  # Shape: (296, 3)
#input = input.reshape(-1, 3 * len(EDCf))  # Shape: (num_samples, 3 * 296)

# Create targets with the correct shape
# Ensure that the targets are structured correctly (depends on your application)
target = np.concatenate((index, middle, ring, little), axis=1)  # Shape: (37, 4, 3)


#Get the input and target data in the pytorch format
#inputs=torch.tensor(input,dtype=torch.float32)

targets=torch.tensor(target,dtype=torch.float32)
def EMG_2_Pytorch(EDC,FDS,FDP):
        # Create padded input tensors
        EDC_tensor = torch.tensor(EDC, dtype=torch.float32).unsqueeze(1)  # Add feature dimension
        FDS_tensor = torch.tensor(FDS, dtype=torch.float32).unsqueeze(1)
        FDP_tensor = torch.tensor(FDP, dtype=torch.float32).unsqueeze(1)

        # Pad the EMG data
        inputs = pad_sequence([EDC_tensor, FDS_tensor, FDP_tensor], batch_first=True)
        return inputs

inputs=EMG_2_Pytorch(EDCf,FDSf,FDPf)
# Create a dataset and dataloader
dataset = EMGDataset(inputs, targets)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


#Initialise the NN model
# Initialize the LSTM-based model
input_size = 1  # Since each EMG channel is a single feature
hidden_size = 64  # Number of hidden units in the LSTM
output_size = 12  # 12 angles
model = EMGLSTM(input_size, hidden_size, output_size)

criterion = nn.MSELoss()  # Assuming a regression task for angles
num_epochs=25000
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
model.train()

plt.ion()
fig, ax= plt.subplots()
line,= ax.plot([],[],label="Loss", color="black")

# Set y-axis to display in scientific notation
ax.get_yaxis().get_major_formatter().set_scientific(True)
ax.get_yaxis().get_major_formatter().set_powerlimits((0, 0))

L=[]
xData=[]
for i in range (num_epochs):
    loss_C = 0.0
    for inputs, targets in dataloader:
       optimizer.zero_grad()  # Clear the gradients
       outputs = model(inputs)  # Forward pass
       loss = criterion(outputs, targets)  # Compute loss
       loss.backward()  # Backward pass
       optimizer.step()  # Update model parameters
       loss_C+=(loss.item())
    L.append(np.sum(loss_C))
    xData.append(i)
    line.set_ydata(np.log1p(L))
    line.set_xdata(xData)
    ax.relim()  # Recompute limits
    ax.autoscale_view()  # Rescale axes to fit new data
    ax.set_title("log(Loss) vs Epoch")
    plt.pause(0.01)

plt.ioff()
plt.show()


#Do a model evaluation
# Load data
I = pd.read_excel(
    "C:/Users/"+logIn+"/OneDrive - University of Warwick/PhD/Hand Trials/Results/Cylindrical Grasp/P10/Cylindrical/Cylindrical03_EMG_Marker_Data.xlsx")

EDC_N = Filtering(I['EDC'].to_numpy(),fs)
FDS_N = Filtering(I['FDS'].to_numpy(),fs)
FDP_N = Filtering(I['FDP'].to_numpy(),fs)
th  = I['MCP2'].dropna().to_numpy()

index=np.zeros([len(th),3])
middle=np.zeros([len(th),3])
ring=np.zeros([len(th),3])
little=np.zeros([len(th),3])

index[:,0]=filter(th,125,9)
index[:,1]= filter(I['PIP2'].dropna().to_numpy(),125,10)
index[:,2]= filter(I['DIP2'].dropna().to_numpy(),125,9)

middle[:,0]= filter(I['MCP3'].dropna().to_numpy(),125,9)
middle[:,1]= filter(I['PIP3'].dropna().to_numpy(),125,6)
middle[:,2]= filter(I['DIP3'].dropna().to_numpy(),125,6)

ring[:,0]= filter(I['MCP4'].dropna().to_numpy(),125,14)
ring[:,1]= filter(I['PIP4'].dropna().to_numpy(),125,7)
ring[:,2]= filter(I['DIP4'].dropna().to_numpy(),125,4)

little[:,0]= filter(I['MCP5'].dropna().to_numpy(),125,7)
little[:,1]= filter(I['PIP5'].dropna().to_numpy(),125,5)
little[:,2]= filter(I['DIP5'].dropna().to_numpy(),125,4)

inputs_N=EMG_2_Pytorch(EDC_N,FDS_N,FDP_N)

model.eval()

with torch.no_grad():
    pred=model(inputs_N)

target = np.concatenate((index, middle, ring, little), axis=1)  # Shape: (37, 4, 3)
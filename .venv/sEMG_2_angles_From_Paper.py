import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from os import getlogin
import scipy.signal
from scipy.interpolate import CubicSpline as CS
from scipy.integrate import solve_ivp
from torch.linalg import solve
import matplotlib.pyplot as plt

#The referenced paper is "Wearable high‑density EMG
#sleeve for complex hand gesture classifcation and continuous joint
#angle estimation"

#Sampling frequency of the sEMG data and MoCap
fsEMG=1000
fs=125

logIn = getlogin()  # Automatically get your username for file paths
# Load the input sEMG data and the joint angles from Excel files
sEMG_data_path = f"C:/Users/{logIn}/OneDrive - University of Warwick/PhD/Hand Trials/Results/Cylindrical Grasp/P1/Cylindrical/sEMG_P1.xlsx"
joint_angles_path = f"C:/Users/{logIn}/OneDrive - University of Warwick/PhD/Hand Trials/Results/Cylindrical Grasp/P1/P1_Cylindrical_Joint_Angles.xlsx"

# Read the data
sEMG_data = pd.read_excel(sEMG_data_path)
joint_angles = pd.read_excel(joint_angles_path)

#Create a time vector
time = np.arange(len(sEMG_data)) / fsEMG

#Compute muscle activation
def bandpass_filter(signal, fs, low, high):
    b, a = scipy.signal.butter(4, [low / (fs / 2), high / (fs / 2)], btype='band')
    filtered = scipy.signal.filtfilt(b, a, signal)
    return np.abs(filtered) / np.max(np.abs(filtered))

def muscle_activation(t,a,sig):
    Tact = 15 * 10** -3
    Tdeact = 50 * 10 ** -3
    dadt=(sig(t)/Tact+(1-sig(t))/Tdeact)*(sig(t)-a)
    return dadt

#Create an empy dataframe to store the muscle activation
X = pd.DataFrame()
# Filter the sEMG data
for col in sEMG_data.columns:
    sEMG_data_f = bandpass_filter(sEMG_data[col], fsEMG, 10, 450)
    #Create a spline interpolation to be used in a ODE solver
    sEMG_data_CS = CS(time, sEMG_data_f, bc_type='natural')
    #Solve the ODE to get the muscle activation
    sol=solve_ivp(lambda t,a:muscle_activation(t,a,sEMG_data_CS),[0,time[-1]],[0],t_eval=time)

    #downsample the data so that it can be used in the model
    X[col] = sol.y[0][::int(fsEMG/fs)]


#Filter angular joint data
b,a= scipy.signal.butter(4, 12/(fs/2),'low')
#Filter the joint angles
for col in joint_angles.columns:
    joint_angles[col]=scipy.signal.filtfilt(b,a,joint_angles[col])


y = joint_angles  # Target outputs (joint angles)


# Standardize the features
scaler_y = StandardScaler()
scaler_X= StandardScaler()
X = scaler_X.fit_transform(X)
Y = scaler_y.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Define the neural network
class SEMGtoAngles(nn.Module):
    def __init__(self):
        super(SEMGtoAngles, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 1000),  # Adjusted for 3 input channels (EDC, FDS, FDP)
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(1000, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(500, 4000),
            nn.BatchNorm1d(4000),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(4000, 12)  # 12 outputs (DIP, PIP, MCP for 4 fingers)
        )

    def forward(self, x):
        return self.model(x)

# Instantiate the model
model = SEMGtoAngles()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
def train_model(model, criterion, optimizer, X_train, y_train, X_test, y_test, epochs=4000):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test)
                val_loss = criterion(val_outputs, y_test)
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
            model.train()

# Train the model
train_model(model, criterion, optimizer, X_train, y_train, X_test, y_test)


#Plot the results

model.eval()
with torch.no_grad():
    y_pred = scaler_y.inverse_transform(model(X_test).numpy())
    y_test = scaler_y.inverse_transform(y_test.numpy())

    # Plot the results individually for each joint
    fig, axs = plt.subplots(4, 3, figsize=(10, 10))
    for idx,cols in enumerate(joint_angles.columns):  # For fingers 2, 3, 4, 5
            if idx<=3:
                joint=2
            elif idx<=7:
                joint=0
            else:
                joint=1

            if cols.split("P")[-1]=="2":
                finger=0
            elif cols.split("P")[-1]=="3":
                finger=1
            elif cols.split("P")[-1]=="4":
                finger=2
            else:
                finger=3

            axs[finger, joint].plot(np.linspace(0,time[-1],len(y_test)), y_test[:, idx], label="True")
            axs[finger, joint].plot(np.linspace(0,time[-1],len(y_pred[:,idx])), y_pred[:, idx], label="Predicted", linestyle="--")
            axs[finger, joint].set_title(f"Finger {cols}")
            axs[finger, joint].legend()

    plt.tight_layout()

    y_pred_N = scaler_y.inverse_transform((model(torch.tensor(X,dtype=torch.float32)).numpy()))


    fig, ax1s = plt.subplots(4, 3, figsize=(10, 10))
    for idx, cols in enumerate(joint_angles.columns):  # For fingers 2, 3, 4, 5
        if idx <= 3:
            joint = 2
        elif idx <= 7:
            joint = 0
        else:
            joint = 1

        if cols.split("P")[-1] == "2":
            finger = 0
        elif cols.split("P")[-1] == "3":
            finger = 1
        elif cols.split("P")[-1] == "4":
            finger = 2
        else:
            finger = 3

        ax1s[finger, joint].plot(time[::int(fsEMG/fs)], (y[cols].values), label="True")
        ax1s[finger, joint].plot(np.linspace(0, time[-1], len(y_pred_N[:, idx])), y_pred_N[:, idx], label="Predicted",linestyle="--")
        ax1s[finger, joint].set_title(f"Finger {cols}")
        ax1s[finger, joint].legend()


    plt.tight_layout()


Data = f"C:/Users/{logIn}/OneDrive - University of Warwick/PhD/Hand Trials/Results/Cylindrical Grasp/P1/Cylindrical/EMG_Angles_Cyl3.xlsx"

# Read the excel data
df = pd.read_excel(Data)
EDC_New=df['EDC']
FDP_New=df['FDP']
FDS_New=df['FDS']
time_New=np.arange(len(EDC_New))/fsEMG


df_c=df.drop(columns=['EDC','FDP','FDS']).dropna()
tim = np.arange(len(df_c))/fs


# Filter the sEMG data
EDC_New_f = bandpass_filter(EDC_New, fsEMG, 10, 450)
FDP_New_f = bandpass_filter(FDP_New, fsEMG, 10, 450)
FDS_New_f = bandpass_filter(FDS_New, fsEMG, 10, 450)

# Create a spline interpolation to be used in a ODE solver
EDC_New_CS = CS(time_New, EDC_New_f, bc_type='natural')
FDP_New_CS= CS(time_New, FDP_New_f, bc_type='natural')
FDS_New_CS = CS(time_New, FDS_New_f, bc_type='natural')

# Solve the ODE to get the muscle activation
sol_EDC = solve_ivp(lambda t, a: muscle_activation(t, a, EDC_New_CS), [0, time_New[-1]], [0], t_eval=time_New)
sol_FDP = solve_ivp(lambda t, a: muscle_activation(t, a, FDP_New_CS), [0, time_New[-1]], [0], t_eval=time_New)
sol_FDS = solve_ivp(lambda t, a: muscle_activation(t, a, FDS_New_CS), [0, time_New[-1]], [0], t_eval=time_New)

# Downsample the data so that it can be used in the model
X_new = pd.DataFrame()
X_new['EDC'] = sol_EDC.y[0][::int(fsEMG/fs)]
X_new['FDP'] = sol_FDP.y[0][::int(fsEMG/fs)]
X_new['FDS'] = sol_FDS.y[0][::int(fsEMG/fs)]

# Standardize the features
X_new = scaler_X.fit_transform(X_new)

# Convert to PyTorch tensors
X_new = torch.tensor(X_new, dtype=torch.float32)

# Predict the joint angles
model.eval()

#Filter df_c data
for i in df_c.columns:
    df_c[i]=scipy.signal.filtfilt(b,a,df_c[i])

with torch.no_grad():
    y_pred_new = scaler_y.inverse_transform(model(X_new).numpy())
    # Plot the results
    fig, axs = plt.subplots(4, 3, figsize=(10, 10))
    for idx, cols in enumerate(df_c.columns):  # For fingers 2, 3, 4, 5
        if idx <= 3:
            joint = 2
        elif idx <= 7:
            joint = 0
        else:
            joint = 1

        if cols.split("P")[-1] == "2":
            finger = 0
        elif cols.split("P")[-1] == "3":
            finger = 1
        elif cols.split("P")[-1] == "4":
            finger = 2
        else:
            finger = 3

        axs[finger, joint].plot(np.linspace(0, time_New[-1], len(y_pred_new[:, idx])), y_pred_new[:, idx], label="Predicted", linestyle="--")
        axs[finger,joint].plot(tim,df_c[cols],label="Filtered")
        axs[finger, joint].set_title(f"Finger {cols}")
        axs[finger, joint].legend()
    plt.tight_layout()
plt.show()
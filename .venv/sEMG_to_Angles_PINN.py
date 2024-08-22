import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import torch
import torch.nn as nn
import pandas as pd
from scipy.interpolate import CubicSpline as CS
from scipy.integrate import solve_ivp


# Define the neural network architecture
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.hidden_layer = nn.Sequential(
            nn.Linear(1, 40),  # Increased number of neurons
            nn.Tanh(),
            nn.Linear(40, 40),  # Increased number of neurons
            nn.Tanh(),
            nn.Linear(40, 40),  # Increased number of neurons
            nn.Tanh(),
            nn.Linear(40, 1)
        )

    def forward(self, t):
        return self.hidden_layer(t)

# Load data
I = pd.read_excel(
    "C:/Users/panos/OneDrive - University of Warwick/PhD/Hand Trials/Results/Cylindrical Grasp/P1/sEMG_par_1_Cylindrical2.xlsx")
fs=1000
EDC = I['EDC'].to_numpy()
FDS = I['FDS'].to_numpy()
FDP = I['FDP'].to_numpy()
t=np.zeros(len(EDC))
for i in range(1,len(EDC)):
    t[i]=t[i-1]+1/fs

def Filtering(sig,fs,t):
    b,a=scipy.signal.butter(4,np.array([10, 450])/(fs/2),btype='bandpass')
    sigf=np.abs(scipy.signal.filtfilt(b,a,sig))
    Sig_Max=np.max(sigf)
    sigf=CS(t,sigf/Sig_Max,bc_type='natural')
    return sigf

def activation(t,y,sig):
    Tact = 15e-3;
    Tdeact = 50e-3;
    dadt=(sig(t)/Tact+(1-sig(t))/Tdeact)*(sig(t)-y)
    return dadt

EDCf=Filtering(EDC,fs,t)
FDPf=Filtering(FDP,fs,t)
FDSf=Filtering(FDS,fs,t)

tspan=(t[0], t[-1])
a0=np.array([0])

# Solve the ODE
solution = solve_ivp(lambda t,y: activation(t,y,EDCf), tspan, a0, t_eval=t, method='RK45')
aEDC = solution.y[0]
solution = solve_ivp(lambda t,y: activation(t,y,FDPf), tspan, a0, t_eval=t, method='RK45')
aFDP = solution.y[0]
solution = solve_ivp(lambda t,y: activation(t,y,FDSf), tspan, a0, t_eval=t, method='RK45')
aFDS = solution.y[0]

plt.figure()
plt.plot(t,aEDC)
plt.plot(t,EDCf(t))
plt.legend(["Activtion","Conditioned Signal"])
plt.title("sEMG activation of EDC")
plt.xlabel("Time (s)")
plt.ylabel("Normalised")


plt.figure()
plt.plot(t,aFDP)
plt.plot(t,FDPf(t))
plt.legend(["Activtion","Conditioned Signal"])
plt.title("sEMG activation of FDP")
plt.xlabel("Time (s)")
plt.ylabel("Normalised")

plt.figure()
plt.plot(t,aFDS)
plt.plot(t,FDSf(t))
plt.legend(["Activtion","Conditioned Signal"])
plt.title("sEMG activation of FDS")
plt.xlabel("Time (s)")
plt.ylabel("Normalised")

plt.show()
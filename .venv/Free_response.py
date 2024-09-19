from cProfile import label
from os import listdir,getlogin
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
from scipy.interpolate import CubicSpline as CS
from scipy.integrate import solve_ivp
from scipy.signal import butter, filtfilt
from scipy.signal import find_peaks


m = 3.6395E-05
b1 = 0.0013
k1 = 0.0482
fs=150
# Load data
I = pd.read_excel("C:/Users/"+getlogin()+"/OneDrive - University of Warwick/PhD/Hand Trials/Data for MATLAB/Data P1/Index/MCP_Spring.xlsx")

th1 = (I['MCP']).to_numpy()



[b,a]=butter(4,15/(fs/2),'low')

th1f=filtfilt(b,a,th1)*np.pi/180

[Ind,Mag]=find_peaks(th1f,35*np.pi/180)
time=np.zeros([len(th1f[Ind[5]+55:Ind[5]+55+60])])

for i in range (1,len(th1f[Ind[5]+55:Ind[5]+55+60])):
    time[i]=time[i-1]+1/fs

# Define the neural network architecture
class PINN2(nn.Module):
    def __init__(self):
        super(PINN2, self).__init__()
        self.hidden_layer = nn.Sequential(
            nn.Linear(1, 20),  # Increased number of neurons
            nn.Tanh(),
            nn.Linear(20, 20),  # Increased number of neurons
            nn.Tanh(),
            nn.Linear(20, 1)
        )


    def forward(self, t):
        return self.hidden_layer(t)


model2=PINN2()
thN=th1f[Ind[5]+55:Ind[5]+55+60]

def RMSE():
    t = torch.linspace(time[0], time[-1], len(time)).view(-1, 1)
    u_p=model2(t)
    th1a=torch.tensor(thN).reshape(-1,1)
    return torch.mean((u_p-th1a)**2)

def physics(z,B,K):
    z=z.requires_grad_()
    th=model2(z)
    zt = torch.autograd.grad(th, z, torch.ones_like(th), create_graph=True)[0]
    ztt = torch.autograd.grad(zt, z, torch.ones_like(zt), create_graph=True)[0]
    res = m* ztt + B * zt + K * (th - torch.tensor([[thN[-1]]]))
    return torch.mean(res**2)

def init_cond():
    t0 = torch.tensor([0.0], requires_grad=True)
    th0_pred = model2(t0)
    th1_true = torch.tensor(thN[0])
    th0_loss = torch.mean((torch.squeeze(th0_pred) - th1_true) ** 2)

    # Ensure the same for t0
    th0_t_pred = torch.autograd.grad(th0_pred, t0, torch.ones_like(th0_pred), create_graph=True, allow_unused=True)[0]
    v0_true=torch.tensor(0.)
    v0_loss = torch.mean((torch.squeeze(th0_t_pred)-v0_true) ** 2)
    return th0_loss+v0_loss

b_est = torch.nn.Parameter(torch.tensor(0.001, requires_grad=True))  # Initial guess for damper
k_est = torch.nn.Parameter(torch.tensor(0.001, requires_grad=True))  # Initial guess for spring constant

# Training parameters
learning_rate = 1e-3
num_epochs=25000
optimiser=torch.optim.Adam(list(model2.parameters())+[b_est]+[k_est],lr=learning_rate)


def msd(t, y,b,k):
    dydt=np.zeros([2])
    dydt[0]=y[1]
    dydt[1] = ( - (b) * y[1] - (k) * (y[0] - thN[-1])) / (m)
    return dydt


# Initial conditions
x0 = thN[0]
v0 = 0.0
y0 = [x0, v0]

# Time span for the solution
t_span=(time[0],time[-1])


# Solve the ODE
solution = solve_ivp(lambda t,y:msd(t,y,b1,k1), t_span, y0, method='RK45')

# Extract the results
t11 = solution.t
u_exact = solution.y[0]

# Data for plotting
t_test = torch.linspace(torch.tensor(time[0]), torch.tensor(time[-1]), len(time)).reshape(-1, 1)

t_train =  torch.linspace(torch.tensor(time[0]), torch.tensor(time[-1]), 200).reshape(-1, 1)

plt.ion()
fig, ax = plt.subplots()
ax.scatter(time, th1f[Ind[5]+55:Ind[5]+55+60], label='Gait Lab Data', color='black')
ax.plot(t11,u_exact,label='IBK optimisation', color='red')
# Initialize a line for predictions
line, = ax.plot(t_test.numpy(), np.zeros_like(t_test.numpy()), label='NN Prediction', color='blue')
ax.legend()
# Initialize lists to store losses
data_loss = []
physics_loss = []

K_est=[]
B_est=[]
# Initialize parameter estimates as learnable parameters


for i in range(num_epochs):
    optimiser.zero_grad()
    # Calculate individual losses
    p_loss = physics(t_train,torch.sigmoid(b_est)*1,torch.sigmoid(k_est)*1)
    d_loss = RMSE()

    data_loss.append(d_loss.item())
    physics_loss.append(p_loss.item())
    K_est.append((torch.tensor(1) * torch.sigmoid(k_est)).item())
    B_est.append((torch.tensor(1) * torch.sigmoid(b_est)).item())
    loss= 0.333*RMSE()+0.333*physics(t_train,torch.sigmoid(b_est)*1,torch.sigmoid(k_est)*1)+0.333*init_cond()

    loss.backward()
    optimiser.step()


    prd=model2(t_test).detach().cpu().numpy()
    # Plot
    line.set_ydata(prd)
    ax.set_title(f'Epoch {i}, Loss: {loss.item():.4f}, B: {(torch.tensor(1)*torch.sigmoid(b_est)).item():.4e}, K:{(torch.tensor(1)*torch.sigmoid(k_est)).item():.4e}')
    plt.draw()
    plt.pause(0.05)

plt.ioff()

# Solve the ODE
solution1 = solve_ivp(lambda t,y:msd(t,y,1*torch.sigmoid(b_est).item(),1*torch.sigmoid(k_est).item()), t_span, y0,  method='RK45')

# Extract the results
t11_s = solution1.t
u_exact1 = solution1.y[0]


plt.figure()
plt.plot(t11_s,u_exact1)
model2.eval()
with torch.no_grad():
    pred2=model2(torch.tensor(t11_s, dtype=torch.float32).view(-1,1))
plt.plot(t11_s,pred2.detach().cpu().numpy())
plt.title(f"RMSE: {np.mean((u_exact1-pred2.detach().cpu().numpy())**2):.3f}")
plt.legend(["IBK solution with New Params","NN"])
# Plot the data loss and physics loss on the same figure with two y-axes
fig3, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Log_10 (Data Loss)', color=color)
ax1.plot((np.linspace(0,num_epochs,len(data_loss))),np.log10(data_loss), color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:red'
ax2.set_ylabel('Log_10 (Physics Loss)', color=color)  # we already handled the x-label with ax1
ax2.plot((np.linspace(0,num_epochs,len(data_loss))),np.log10(physics_loss), color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig3.tight_layout()  # to prevent the labels from overlapping

plt.show()
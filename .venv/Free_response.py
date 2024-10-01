from cProfile import label
from os import listdir,getlogin
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.signal import butter, filtfilt
from scipy.signal import find_peaks
from scipy.optimize import minimize

m = 3.6395E-05
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

thN=th1f[Ind[5]+55:Ind[5]+55+60]
# Initial conditions
x0 = thN[0]
v0 = 0.0
y0 = [x0, v0]

# Time span for the solution
t_span=(time[0],time[-1])


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

b_est = torch.nn.Parameter(torch.tensor(0.0002, requires_grad=True))  # Initial guess for damper
k_est = torch.nn.Parameter(torch.tensor(0.0001, requires_grad=True))  # Initial guess for spring constant

# Training parameters
learning_rate = 1e-3
num_epochs=25000
optimiser=torch.optim.Adam(list(model2.parameters())+[b_est]+[k_est],lr=learning_rate)


def msd(t, y,b,k):
    dydt=np.zeros([2])
    dydt[0]=y[1]
    dydt[1] = ( - (b) * y[1] - (k) * (y[0] - thN[-1])) / (m)
    return dydt

def simulate_system(b, k, tspan,time):
    y0 = [thN[0], 0.0]  # initial displacement and velocity
    sol = solve_ivp(lambda t,y: msd(t,y,b,k), tspan,y0,t_eval=time, method='RK45')
    return sol.y[0]  # returning displacement

def obj(params,data,tspan,time):
    b,k=params
    th=simulate_system(b,k,tspan,time)
    return np.sum((data-th)**2)

# Initial guess for parameters [damping coefficient, spring constant]
initial_guess = [0.5, 1.0]  # Adjust these based on expected values

# Bounds on the parameters (e.g., both should be positive)
bounds = [(0, None), (0, None)]  # c, k >= 0
# Minimize the objective function
result = minimize(obj, initial_guess, args=(thN, t_span, time), bounds=bounds)

# Optimal parameters
b1, k1 = result.x

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

model2.train()
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
plt.plot(t11_s,u_exact1*180/np.pi)

model2.eval()
with torch.no_grad():
    pred2=model2(torch.tensor(t11_s, dtype=torch.float32).view(-1,1))
plt.plot(t11_s,pred2.detach().cpu().numpy()*180/np.pi)
plt.plot(t11,u_exact*180/np.pi)
plt.scatter(time,thN*180/np.pi, color='black')
plt.title(f"RMSE: {np.mean((thN-pred2.detach().cpu().numpy())**2)*180/np.pi:.3f} (deg) between data and PINN")
plt.legend(["IBK solution with New Params","NN"," IBK with parameters from minimisation","Gait Lab data"])
P="C:/Users/"+getlogin()+"/OneDrive - University of Warwick/PINNs/Free_response/"
ff=int(listdir(P)[-1].split("_")[-1].split(".png")[0])
dir=P+"Evaluations_"+str(ff+1)+".png"
plt.savefig(dir)

# Plot the bar chart for comparison
labels = ['Damping Coefficient (Nms/rad)', 'Spring Constant (Nm/rad)']
determined_values = [torch.sigmoid(b_est).item(), torch.sigmoid(k_est).item()]
actual_values = [b1, k1]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, determined_values, width, label='PINN', color='blue')
bars2 = ax.bar(x + width/2, actual_values, width, label='Minimisation', color='orange')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Values')
ax.set_title('Comparison of Determined and Actual Parameters')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Add value annotations on bars
def add_value_annotations(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2e}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

add_value_annotations(bars1)
add_value_annotations(bars2)

dir=P+"Bar_Chart_PINN_Minimisation_"+str(ff+1)+".png"
plt.savefig(dir)

plt.show()
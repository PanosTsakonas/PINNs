from cProfile import label
from os import listdir,getlogin
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
from scipy.interpolate import CubicSpline as CS
from scipy.integrate import solve_ivp


m = torch.tensor([[6.67495e-06]])
b = torch.tensor([[0.3782396]])
k = torch.tensor([[9.855922348]])

# Load data
I = pd.read_excel(
    "C:/Users/"+getlogin()+"/OneDrive - University of Warwick/PhD/Hand Trials/Results/Cylindrical Grasp/P1/data_digit_2par_1_Cylindical_ _trial_2Thelen.xlsx")
time = I['time'].to_numpy()
MEDC = I['EDC_PIP'].to_numpy()
MFDS = I['FDS_PIP'].to_numpy()
MFDP = I['FDP_PIP'].to_numpy()
th1 = ((I['th2']).to_numpy())
M_EDC = CS(time, MEDC, bc_type='natural')
M_FDP = CS(time, MFDP, bc_type='natural')
M_FDS = CS(time, MFDS, bc_type='natural')


# Define the Muscle moment as a Cubic Spline sum
def Moment(z):
    if isinstance(z, torch.Tensor):
        t_np = z.detach().cpu().numpy()
    else:
        t_np = np.array(z)
    return M_EDC(t_np) + M_FDP(t_np) + M_FDS(t_np)



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
    th1a=torch.tensor(th1).reshape(-1,1)
    return torch.mean((u_p-th1a)**2)

def physics(z,B,K):
    z=z.requires_grad_()
    th=model2(z)
    zt = torch.autograd.grad(th, z, torch.ones_like(th), create_graph=True)[0]
    ztt = torch.autograd.grad(zt, z, torch.ones_like(zt), create_graph=True)[0]
    f_t = torch.tensor(Moment(z.detach().cpu().numpy()), dtype=torch.float32)
    res = m* ztt + B * zt + K * (th - torch.tensor(th1[-1])) - f_t
    return torch.mean(res**2)

def init_cond():
    t0 = torch.tensor([0.0], requires_grad=True)
    th0_pred = model2(t0)
    th1_true = torch.tensor(th1[0])
    th0_loss = torch.mean((torch.squeeze(th0_pred) - th1_true) ** 2)

    # Ensure the same for t0
    th0_t_pred = torch.autograd.grad(th0_pred, t0, torch.ones_like(th0_pred), create_graph=True, allow_unused=True)[0]
    v0_true=torch.tensor(0.)
    v0_loss = torch.mean((torch.squeeze(th0_t_pred)-v0_true) ** 2)
    return th0_loss+v0_loss

b_est = torch.nn.Parameter(torch.tensor(0.01, requires_grad=True))  # Initial guess for damper
k_est = torch.nn.Parameter(torch.tensor(0.1, requires_grad=True))  # Initial guess for spring constant

# Training parameters
learning_rate = 1e-3
num_epochs=25000
optimiser=torch.optim.Adam(list(model2.parameters())+[b_est]+[k_est],lr=learning_rate)


def msd(t, y,b,k):
    x, v = y
    moment_value = Moment(np.array([t]))  # Ensure t is an array for cubic spline interpolation
    # Convert moment_value to a float, as it might be a numpy array
    moment_value = moment_value.item() if isinstance(moment_value, np.ndarray) else moment_value
    dvdt = (moment_value - float(b) * v - float(k) * (x - th1[-1])) / float(m)
    return [v, dvdt]


# Initial conditions
x0 = th1[0]
v0 = 0.0
y0 = [x0, v0]

# Time span for the solution
t_span = (time[0], time[-1])
t_eval = np.linspace(t_span[0], t_span[1], 500)

# Solve the ODE
solution = solve_ivp(lambda t,y:msd(t,y,b,k), t_span, y0, t_eval=t_eval, method='RK45')

# Extract the results
t11 = solution.t
u_exact = solution.y[0]

# Data for plotting
t_test = torch.linspace(time[0], time[-1], len(time)).reshape(-1, 1)

t_train =  torch.linspace(time[0], time[-1], 150).reshape(-1, 1)

plt.ion()
fig, ax = plt.subplots()
ax.scatter(time, th1, label='Gait Lab Data', color='black')
ax.plot(t11,u_exact,label='IBK optimisation', color='red')
# Initialize a line for predictions
line, = ax.plot(t_test.numpy(), np.zeros_like(t_test.numpy()), label='NN Prediction', color='blue')
line2, = ax.plot(t_test.numpy(), np.zeros_like(t_test.numpy()), label='IBK from NN values', color='brown')
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
    p_loss = physics(t_train,torch.sigmoid(b_est)*2,torch.sigmoid(k_est)*10)
    d_loss = RMSE()

    data_loss.append(d_loss.item())
    physics_loss.append(p_loss.item())
    K_est.append((torch.tensor(10) * torch.sigmoid(k_est)).item())
    B_est.append((torch.tensor(2) * torch.sigmoid(b_est)).item())
    loss= 0.333*RMSE()+0.333*physics(t_train,torch.sigmoid(b_est)*2,torch.sigmoid(k_est)*10)+0.333*init_cond()

    loss.backward()
    optimiser.step()


    prd=model2(t_test).detach().cpu().numpy()
    # Plot
    line.set_ydata(prd)
    ax.set_title(f'Epoch {i}, Loss: {loss.item():.4f}, B: {(torch.tensor(2)*torch.sigmoid(b_est)).item():.4e}, K:{(torch.tensor(10)*torch.sigmoid(k_est)).item():.4e}')
    if i>1500:
        solution = solve_ivp(lambda t, y: msd(t, y, 2*torch.tensor(b_est).item(), 10*torch.tensor(k_est).item()), t_span, y0, t_eval=time, method='Radau')
        line2.set_ydata(solution.y[0])
    plt.draw()
    plt.pause(0.05)

plt.ioff()

# Solve the ODE
solution1 = solve_ivp(lambda t,y:msd(t,y,2*torch.sigmoid(b_est).item(),10*torch.sigmoid(k_est).item()), t_span, y0, t_eval=t_eval, method='LSODA')

# Extract the results
t11_s = solution1.t
u_exact1 = solution1.y[0]
P="C:/Users/"+getlogin()+"/OneDrive - University of Warwick/PINNs/Parameter Estimation Pytorch/"
ff=int(listdir(P)[-1].split("take_")[-1].split(".png")[0])
dir=P+"Parameter_Estimation_Par1_Digit2_Cyl02_take_"+str(ff+1)+".png"
plt.savefig(dir)

plt.figure()
plt.plot(t11_s,u_exact1)
model2.eval()
with torch.no_grad():
    pred2=model2(torch.tensor(t11_s, dtype=torch.float32).view(-1,1))
plt.plot(t11_s,pred2.detach().cpu().numpy())
plt.title(f"RMSE: {np.mean((u_exact1-pred2.detach().cpu().numpy())**2):.3f}")
plt.legend(["IBK with new Params","NN"])
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


L=[sum(z) for z in zip(data_loss,physics_loss)]

plt.figure()
plt.plot(K_est,L)
plt.title("Loss function vs Spring estimates")
plt.xlabel("Spring (Nm/rad)")
plt.ylabel("Loss")

plt.figure()
plt.plot(B_est,L)
plt.title("Loss function vs Damper estimates")
plt.xlabel("Damper (Nms/rad)")
plt.ylabel("Loss")

plt.show()
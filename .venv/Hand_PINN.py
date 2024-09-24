import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import CubicSpline as CS
from scipy.integrate import solve_ivp
import os

# Define the neural network architecture
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.hidden_layer = nn.Sequential(
            nn.Linear(1, 40),  # Increased number of neurons
            nn.Tanh(),
            nn.Linear(40, 1)
        )

    def forward(self, t):
        return self.hidden_layer(t)


m = torch.tensor([[6.67495e-06]])
b = torch.tensor([[0.3782396]])
k = torch.tensor([[9.855922348]])

logIn=os.getlogin()
# Load data
I = pd.read_excel(
    "C:/Users/"+logIn+"/OneDrive - University of Warwick/PhD/Hand Trials/Results/Cylindrical Grasp/P1/data_digit_2par_1_Cylindical_ _trial_2Thelen.xlsx")
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

# Define the loss functions
def physics_loss(z,model):
    z=z.requires_grad_(True)
    th = model(z)
    th_t = torch.autograd.grad(th, z, torch.ones_like(th), create_graph=True)[0]
    th_tt = torch.autograd.grad(th_t, z, torch.ones_like(th_t), create_graph=True)[0]
    f_t=    torch.tensor(Moment(z.detach().cpu().numpy()), dtype=torch.float32)
    # Compute residual
    res = m * th_tt + b * th_t + k * (th - torch.tensor([[th1[-1]]])) - f_t

    return torch.mean(res ** 2)


def boundary_loss(model):
    t0 = torch.tensor([0.0], requires_grad=True)
    th0_pred = model(t0)
    th1_true=torch.tensor(th1[0])
    th0_loss = torch.mean((torch.squeeze(th0_pred) - th1_true) ** 2)

    # Ensure the same for t0
    th0_t_pred = torch.autograd.grad(th0_pred, t0, torch.ones_like(th0_pred), create_graph=True, allow_unused=True)[0]
    v0_loss =torch.mean((torch.squeeze(th0_t_pred)) ** 2)

    return th0_loss + v0_loss


# Combine physics loss and initial condition loss
def total_loss(z,model):
    return 0.1*physics_loss(z,model) + boundary_loss(model)

# Initialize the model
model = PINN()

# Training parameters
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
num_epochs = 20000


def msd(t, y):
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
solution = solve_ivp(msd, t_span, y0, t_eval=t_eval, method='RK45')

# Extract the results
t11 = solution.t
u_exact = solution.y[0]

# Data for plotting
t_test = torch.linspace(time[0], time[-1], len(time)).reshape(-1, 1)
t_train =  torch.linspace(time[0], time[-1], 500).reshape(-1, 1)

plt.ion()
fig, ax = plt.subplots()
ax.plot(t11, u_exact, label='Exact Solution', color='black')
ax.plot(time,th1,label='Gait Lab', color='red')
# Initialize a line for predictions
line, = ax.plot(t_test.numpy(), np.zeros_like(t_test.numpy()), label='NN Prediction', color='blue')
ax.legend()

model.train()
# Training loop
for epoch in range(num_epochs):

    optimizer.zero_grad()
    # Generate time samples for training

    # Total loss
    loss = total_loss(t_train,model)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Test the model
    u_pred = model(t_test).detach().cpu().numpy()

    # Plot
    line.set_ydata(u_pred)
    ax.set_title(f'Epoch {epoch}, Loss: {loss.item():.4f}')
    plt.draw()
    plt.pause(0.01)

    if epoch % (num_epochs // 10) == 0:  # Print every 10% of epochs
        print(f'Epoch {epoch}, Loss: {loss.item()}')

plt.ioff()
plt.show()


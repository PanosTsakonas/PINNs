import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from os import getlogin
from scipy.interpolate import CubicSpline as CS
import pandas as pd
import seaborn as sns

Mo = torch.tensor([[6.67495e-06]])
b_true = torch.tensor([[0.3782396]])
k_true = torch.tensor([[9.855922348]])

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


# Bayesian parameter estimation using variational inference

class BayesianParameter(nn.Module):
    def __init__(self, init_value):
        super(BayesianParameter, self).__init__()
        self.mu = nn.Parameter(torch.tensor(init_value))  # Mean of the distribution
        self.rho = nn.Parameter(torch.tensor(-3.0))  # Controls variance (log-scale)

    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))  # Softplus to ensure positivity

    def sample(self):
        epsilon = torch.randn_like(self.mu)
        return self.mu + self.sigma * epsilon

    def kl_divergence(self):
        # Mean and variance of the prior (N(0, 1))
        mu_0 = 0
        sigma_0 = 1

        # Mean and variance of the posterior (N(mu, exp(rho)))
        mu = self.mu
        sigma = torch.exp(self.rho)

        # KL divergence formula
        kl_div = 0.5 * (torch.log(sigma ** 2 / sigma_0 ** 2) + (sigma_0 ** 2 + (mu - mu_0) ** 2) / sigma ** 2 - 1)
        return kl_div

# Define a simple PINN with Bayesian parameters for b and k
class PINN_Bayes(nn.Module):
    def __init__(self):
        super(PINN_Bayes, self).__init__()
        self.hidden_layer = nn.Sequential(
            nn.Linear(1, 30),
            nn.Tanh(),
            nn.Linear(30,30),
            nn.Tanh(),
            nn.Linear(30, 1)
        )

        # Bayesian parameters for b (damping) and k (stiffness)
        self.b = BayesianParameter(0.1)
        self.k = BayesianParameter(1.0)

    def forward(self, t):
        return self.hidden_layer(t)

    def physics_loss(self, t,  m):
        t=t.requires_grad_()
        x_pred = pinn(t)
        v_pred = torch.autograd.grad(x_pred, t, torch.ones_like(x_pred), create_graph=True)[0]
        a_pred = torch.autograd.grad(v_pred, t, torch.ones_like(v_pred), create_graph=True)[0]

        # Use the Bayesian parameters sampled values for b and k
        b_sample = torch.exp(self.b.sample())
        k_sample = torch.exp(self.k.sample())
        f_t = torch.tensor(Moment(t.detach().cpu().numpy()), dtype=torch.float32)
        # Residual of the mass-spring-damper equation
        residual = m * a_pred + b_sample * v_pred + k_sample * (x_pred-th1[-1])-f_t

        # RMSE Loss between data and predictions

        tim1 = torch.linspace(time[0], time[-1], len(time)).view(-1, 1)
        u_p = pinn(tim1)
        th1a = torch.tensor(th1).reshape(-1, 1)

        # Initial condition of ODE
        init_cond_loss=init_cond()

        # Combined Loss
        return 0.4*torch.mean(residual ** 2)+torch.mean((th1a-u_p)**2)+0.3*init_cond_loss

def init_cond():
    t0 = torch.tensor([0.0], requires_grad=True)
    th0_pred = pinn(t0)
    th1_true = torch.tensor(th1[0])
    th0_loss = torch.mean((torch.squeeze(th0_pred) - th1_true) ** 2)

    # Ensure the same for t0
    th0_t_pred = torch.autograd.grad(th0_pred, t0, torch.ones_like(th0_pred), create_graph=True, allow_unused=True)[0]
    v0_true=torch.tensor(0.)
    v0_loss = torch.mean((torch.squeeze(th0_t_pred)-v0_true) ** 2)
    return th0_loss+v0_loss

plt.ion()
fig, ax = plt.subplots()
ax.scatter(time, th1, label='Gait Lab Data', color='black')
# Initialize a line for predictions
line, = ax.plot(time, np.zeros_like(time), label='NN Prediction', color='blue')
ax.legend()

# Function for training the Bayesian PINN
def train_bayesian_pinn(pinn, optimizer, t_train, m, epochs=800):
    kl_divergence = 0
    for epoch in range(epochs):

        # Physics loss
        physics_loss = pinn.physics_loss(t_train, m)

        # KL divergence loss for the Bayesian parameters
        kl_loss = pinn.b.kl_divergence() + pinn.k.kl_divergence()

        # Total loss (physics loss + KL divergence)
        total_loss = physics_loss + 0.05* kl_loss  # 1e-5 is a weighting factor for KL divergence
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        prd = pinn(t_train).detach().cpu().numpy()
        # Plot
        line.set_ydata(prd)
        ax.set_title(f'Epoch {epoch}, Loss: {total_loss.item():.4f}, B: {torch.exp(pinn.b.mu).item():.4e}, K:{torch.exp(pinn.k.mu).item():.4e}')
        plt.draw()
        plt.pause(0.05)


plt.ioff()
# Example usage with a mass-spring-damper system
if __name__ == "__main__":
    pinn = PINN_Bayes()
    optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)

    # Generate synthetic training data (replace with real data)
    t_train = torch.linspace(time[0], time[-1], len(time)).reshape(-1, 1)

    pinn.train()
    # Train the Bayesian PINN
    train_bayesian_pinn(pinn, optimizer, t_train, Mo)

    # After training, you can access the learned distributions for b and k
    print("Estimated b:", torch.exp(pinn.b.mu).item(), "±", pinn.b.sigma.item())
    print("Estimated k:", torch.exp(pinn.k.mu).item(), "±", pinn.k.sigma.item())

# Plot the distribution of the parameters
    num_samples = 1000  # Number of samples for the distribution
    b_samples = torch.exp(pinn.b.mu + pinn.b.sigma * torch.randn(num_samples))  # Sample from the posterior
    k_samples = torch.exp(pinn.k.mu + pinn.k.sigma * torch.randn(num_samples))  # Sample from the posterior
    b_estimated=torch.exp(pinn.b.mu).item()
    k_estimated = torch.exp(pinn.k.mu).item()
    plt.figure(figsize=(12, 6))

    # Plot for b
    plt.subplot(1, 2, 1)
    sns.histplot(b_samples.detach().numpy(), bins=30, kde=True, color='blue', stat='density')
    plt.axvline(b_estimated, color='red', linestyle='--', label='Estimated Mean')
    plt.title('Distribution of Damper')
    plt.xlabel('Damper (Nms/rad)')
    plt.ylabel('Density')
    plt.legend()

    # Plot for k
    plt.subplot(1, 2, 2)
    sns.histplot(k_samples.detach().numpy(), bins=30, kde=True, color='green', stat='density')
    plt.axvline(k_estimated, color='red', linestyle='--', label='Estimated Mean')
    plt.title('Distribution of Spring')
    plt.xlabel('Spring (Nm/rad)')
    plt.ylabel('Density')
    plt.legend()

    plt.tight_layout()
    plt.show()
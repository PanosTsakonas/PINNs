import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Define the neural network (PINN)
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(2, 640),
            nn.Tanh(),
            nn.Linear(640, 120),
            nn.ReLU(),
            nn.Linear(120, 1)
        )

    def forward(self, x):
        return self.hidden(x)

# Define the cable equation residual
def residual(model, x, t, lambda2, tau_m):
    t=t.requires_grad_(True)
    x=x.requires_grad_(True)
    xt = torch.cat((x, t), dim=1)

    V = model(xt)

    V_t = torch.autograd.grad(V, t, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    V_x = torch.autograd.grad(V, x, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    V_xx = torch.autograd.grad(V_x, x, grad_outputs=torch.ones_like(V_x), create_graph=True)[0]

    res = lambda2 * V_xx - V - tau_m * V_t
    return res

# Generate synthetic dataset
def generate_data():
    x = np.linspace(0, 1, 50)
    t = np.linspace(0, 1, 50)
    X, T = np.meshgrid(x, t)

    # Exact solution (example: decaying sinusoidal wave)
    V_exact = np.exp(-T) * np.sin(np.pi * X)

    return X, T, V_exact

# Training parameters
lambda2 = 0.1  # Length constant squared
tau_m = 0.01   # Time constant
learning_rate = 1e-3
epochs = 10000

# Prepare data
X, T, V_exact = generate_data()
x = torch.tensor(X.flatten(), dtype=torch.float32).view(-1, 1)
t = torch.tensor(T.flatten(), dtype=torch.float32).view(-1, 1)
V_data = torch.tensor(V_exact.flatten(), dtype=torch.float32).view(-1, 1)

# Initialize model and optimizer
model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    optimizer.zero_grad()

    # Data loss
    V_pred = model(torch.cat((x, t), dim=1))
    data_loss = torch.mean((V_pred - V_data)**2)

    # PDE loss
    res = residual(model, x, t, lambda2, tau_m)
    pde_loss = torch.mean(res**2)

    # Total loss
    loss = data_loss + pde_loss

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}, Data Loss: {data_loss.item():.6f}, PDE Loss: {pde_loss.item():.6f}")

# Evaluate the model
x_test = torch.tensor(X.flatten(), dtype=torch.float32).view(-1, 1)
t_test = torch.tensor(T.flatten(), dtype=torch.float32).view(-1, 1)
V_pred = model(torch.cat((x_test, t_test), dim=1)).detach().numpy()

# Reshape results
V_pred = V_pred.reshape(X.shape)

# Plot the results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title("Exact Solution")
plt.contourf(X, T, V_exact, levels=50, cmap="viridis")
plt.colorbar()
plt.xlabel("x")
plt.ylabel("t")

plt.subplot(1, 2, 2)
plt.title("PINN Prediction")
plt.contourf(X, T, V_pred, levels=50, cmap="viridis")
plt.colorbar()
plt.xlabel("x")
plt.ylabel("t")

plt.tight_layout()
plt.show()
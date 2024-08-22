import torch
import torch.nn as nn
import numpy as np

# Define the neural network architecture
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.hidden_layer = nn.Linear(1, 20)  # 1 input, 20 neurons in the hidden layer
        self.output_layer = nn.Linear(20, 1)  # 1 output

    def forward(self, x):
        x = torch.tanh(self.hidden_layer(x))
        return self.output_layer(x)


# Initialize the model
model = PINN()


# Define the loss functions
def physics_loss(x):
    u = model(x)
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    f = (torch.pi ** 2) * torch.sin(torch.pi * x)
    return torch.mean((u_xx + f) ** 2)


def boundary_loss(x_left, x_right):
    u_left = model(x_left)
    u_right = model(x_right)
    return torch.mean(u_left ** 2) + torch.mean(u_right ** 2)


# Training parameters
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
num_epochs = 5000

import matplotlib.pyplot as plt

# Test the model
x_test = torch.linspace(0, 1, 100).reshape(-1, 1)
# Exact solution
u_exact = torch.sin(torch.pi * x_test).numpy()

plt.ion()
fig, ax = plt.subplots()
ax.plot(x_test,u_exact,label='Exact Solution', color='black')
line, =ax.plot(x_test.numpy(),np.zeros_like(x_test.numpy()), label='NN Prediction', color='blue')
ax.legend()

# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()

    # Generate random points in the domain
    x = torch.rand(100, 1, requires_grad=True)
    x_left = torch.zeros(1, 1)
    x_right = torch.ones(1, 1)

    # Compute losses
    p_loss = physics_loss(x)
    b_loss = boundary_loss(x_left, x_right)

    # Total loss
    loss = p_loss + b_loss

    # Backpropagation
    loss.backward()
    optimizer.step()

    u_pred = model(x_test).detach().cpu().numpy()

    line.set_ydata((u_pred))
    ax.set_title(f'Epoch {epoch}, Loss: {loss.item():.4f}')
    plt.draw()
    plt.pause(0.1)

    if epoch % 500 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')




plt.ioff()
plt.show()

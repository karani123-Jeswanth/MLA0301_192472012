import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

class DroneDQN(nn.Module):
    def __init__(self, state_dim=4, action_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, x):
        return self.net(x)

# Training Step Demo
model = DroneDQN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Fake Replay batch: [x, y, battery, wind]
states = torch.randn(32, 4)
actions = torch.randint(0, 4, (32, 1))
rewards = torch.randn(32, 1)
next_states = torch.randn(32, 4)

q_vals = model(states).gather(1, actions)
next_q_vals = model(next_states).max(1)[0].detach().unsqueeze(1)
target = rewards + 0.99 * next_q_vals

loss = criterion(q_vals, target)
optimizer.zero_grad()
loss.backward()
optimizer.step()
print("DQN Drone Delivery Model Batch Loss:", loss.item())
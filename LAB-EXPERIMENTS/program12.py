import torch
import torch.nn as nn
import torch.optim as optim

class RoboticArmPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 2), nn.Softmax(dim=-1))
    def forward(self, x): return self.fc(x)

policy = RoboticArmPolicy()
optimizer = optim.Adam(policy.parameters(), lr=0.01)

# Synthetic trajectory (pick-and-place precision)
state = torch.tensor([0.5, 0.2, 0.1, 1.0], dtype=torch.float32)
probs = policy(state)
dist = torch.distributions.Categorical(probs)
action = dist.sample()
reward = 10.0 # High reward for accurate placement

loss = -dist.log_prob(action) * reward
optimizer.zero_grad()
loss.backward()
optimizer.step()
print(f"Policy Updated. Arm Action: {action.item()}, Loss: {loss.item():.4f}")
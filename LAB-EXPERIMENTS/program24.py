import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class TradingPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 3), nn.Softmax(dim=-1))
        # Actions: 0: Sell, 1: Hold, 2: Buy

policy = TradingPolicy()
optimizer = optim.Adam(policy.parameters(), lr=0.01)

# Backtest dummy episode
price_changes = [0.02, -0.01, 0.05]
log_probs = []
for delta in price_changes:
    p = policy(torch.tensor([delta, 0.5, -0.2]))
    dist = torch.distributions.Categorical(p)
    a = dist.sample()
    log_probs.append(dist.log_prob(a))

total_profit = 150.0 # Reward
loss = -sum(log_probs) * total_profit
optimizer.zero_grad()
loss.backward()
optimizer.step()
print("Trading Policy Optimized. Loss:", loss.item())
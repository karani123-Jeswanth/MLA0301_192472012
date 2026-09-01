import torch
import torch.nn as nn
import torch.optim as optim

class ParkingPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 3), nn.Softmax(dim=-1))
    def forward(self, s): return self.net(s)

policy = ParkingPolicy()
opt = optim.Adam(policy.parameters(), lr=0.01)

# Episode rollout simulation
log_probs = []
rewards = [ -1.0, -1.0, 15.0 ] # park completed at t=3

for r_t in rewards:
    s = torch.randn(3)
    p = policy(s)
    d = torch.distributions.Categorical(p)
    a = d.sample()
    log_probs.append(d.log_prob(a))

discounted_returns = [13.0, 14.5, 15.0]
loss = -sum(lp * G for lp, G in zip(log_probs, discounted_returns))
opt.zero_grad()
loss.backward()
opt.step()
print("REINFORCE Parking Episode Update Complete. Loss:", loss.item())
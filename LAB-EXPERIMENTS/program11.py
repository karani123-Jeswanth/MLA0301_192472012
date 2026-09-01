import torch
import torch.nn as nn

# Standard / Double DQN Head
class StandardDQN(nn.Module):
    def __init__(self, in_dim=6, out_dim=3):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, out_dim))
    def forward(self, x): return self.fc(x)

# Dueling DQN Architecture
class DuelingDQN(nn.Module):
    def __init__(self, in_dim=6, out_dim=3):
        super().__init__()
        self.feat = nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU())
        self.val = nn.Linear(32, 1)
        self.adv = nn.Linear(32, out_dim)
    def forward(self, x):
        f = self.feat(x)
        val = self.val(f)
        adv = self.adv(f)
        return val + (adv - adv.mean(dim=-1, keepdim=True))

x = torch.randn(1, 6) # Traffic features
print("Standard DQN Output:", StandardDQN()(x).detach().numpy())
print("Dueling DQN Output:", DuelingDQN()(x).detach().numpy())
import torch
import torch.nn as nn

class DDPGActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 32), # resource/unit status
            nn.ReLU(),
            nn.Linear(32, 2), # continuous actions (e.g., target coordinates)
            nn.Tanh()
        )
    def forward(self, s): return self.net(s)

actor = DDPGActor()
state = torch.randn(1, 6)
print("DDPG Continuous Control Output:", actor(state).detach().numpy())
import torch
import torch.nn as nn

class HighwayDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_net = nn.Sequential(
            nn.Linear(5, 32), # 5 sensor inputs
            nn.ReLU(),
            nn.Linear(32, 3)  # Accel, Maintain, Decel
        )
    def forward(self, x):
        return self.q_net(x)

net = HighwayDQN()
sensors = torch.tensor([[50.0, 48.0, 100.0, 20.0, 0.0]]) # distance sensors
print("Q-values for Highway Actions:", net(sensors).detach().numpy())
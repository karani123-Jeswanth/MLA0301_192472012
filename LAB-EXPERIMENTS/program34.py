import torch
import torch.nn as nn

class HVACPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 3), nn.Softmax(dim=-1))
        # 0: Heat, 1: Off, 2: Cool

    def forward(self, state): return self.net(state)

policy = HVACPolicy()
indoor_temp, target_temp = 18.0, 22.0
state = torch.tensor([indoor_temp, target_temp])
probs = policy(state)
print("HVAC Action Probabilities [Heat, Off, Cool]:", probs.detach().numpy())
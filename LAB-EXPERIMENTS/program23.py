import torch
import torch.nn as nn

class LaneChangePPO(nn.Module):
    def __init__(self):
        super().__init__()
        # Inputs: [distance_front, front_speed, left_clear, right_clear]
        self.policy_head = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 3), nn.Softmax(dim=-1))
        # Actions: 0: Keep Lane, 1: Change Left, 2: Change Right

    def forward(self, state):
        return self.policy_head(state)

net = LaneChangePPO()
telemetry = torch.tensor([12.5, 45.0, 1.0, 0.0]) # Car close ahead, left lane clear
decision = net(telemetry)
print("Lane Decision Probabilities (Keep, Left, Right):", decision.detach().numpy())
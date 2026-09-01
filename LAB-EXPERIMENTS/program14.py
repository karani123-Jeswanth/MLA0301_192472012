import torch
import torch.nn as nn

class ElevatorA2C(nn.Module):
    def __init__(self, state_dim=5, action_dim=3): # Action: Up, Down, Wait
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(state_dim, 32), nn.ReLU(), nn.Linear(32, action_dim), nn.Softmax(dim=-1))
        self.critic = nn.Sequential(nn.Linear(state_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, s):
        return self.actor(s), self.critic(s)

model = ElevatorA2C()
state = torch.randn(1, 5)
policy_dist, state_val = model(state)
print("Elevator Action Probabilities:", policy_dist.detach().numpy())
print("Expected Elevator Wait-time State Value:", state_val.item())
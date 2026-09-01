import torch
import torch.nn as nn

class DuelingNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Linear(state_dim, 64)
        self.value_stream = nn.Linear(64, 1)
        self.advantage_stream = nn.Linear(64, action_dim)

    def forward(self, state):
        feat = torch.relu(self.fc(state))
        val = self.value_stream(feat)
        adv = self.advantage_stream(feat)
        return val + (adv - adv.mean(dim=-1, keepdim=True))

duel_net = DuelingNetwork(state_dim=8, action_dim=4)
state = torch.randn(1, 8)
print("Dueling Q Output:", duel_net(state).detach().numpy())
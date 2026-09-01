import torch

def ppo_clip_loss(old_log_probs, log_probs, advantages, clip_eps=0.2):
    ratios = torch.exp(log_probs - old_log_probs)
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    return -torch.min(surr1, surr2).mean()

# Test PPO loss calculation
old_lp = torch.tensor([-0.5, -0.2, -0.8])
new_lp = torch.tensor([-0.4, -0.3, -0.7])
adv = torch.tensor([1.2, -0.5, 0.8])

loss = ppo_clip_loss(old_lp, new_lp, adv)
print("PPO Balance Loss:", loss.item())
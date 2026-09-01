import numpy as np

# Queue length MDP (0..5 vehicles)
states = list(range(6))
policy = {s: 0 for s in states} # 0: Keep Green, 1: Switch Light
gamma = 0.85

def step_queue(s, a):
    # Action 1 reduces queue faster with switch penalty
    ns = max(0, s - 2) if a == 1 else max(0, s - 1)
    r = - (s ** 1.5) - (2 if a == 1 else 0)
    return ns, r

V = {s: 0.0 for s in states}
# Policy Evaluation
for _ in range(20):
    for s in states:
        ns, r = step_queue(s, policy[s])
        V[s] = r + gamma * V[ns]
    for s in states:
        q0 = step_queue(s, 0)[1] + gamma * V[step_queue(s, 0)[0]]
        q1 = step_queue(s, 1)[1] + gamma * V[step_queue(s, 1)[0]]
        policy[s] = 1 if q1 > q0 else 0

print("Optimized Traffic Light Action Policy per Queue State:", policy)
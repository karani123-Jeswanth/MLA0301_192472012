import numpy as np

# Policy Iteration for Taxi Grid
grid_size = 4
num_states = grid_size * grid_size
gamma = 0.9

P = {s: {a: [] for a in range(4)} for s in range(num_states)}
# Populate transitions
for s in range(num_states):
    r, c = divmod(s, grid_size)
    for a, (dr, dc) in enumerate([(-1,0), (1,0), (0,-1), (0,1)]):
        nr, nc = max(0, min(grid_size-1, r + dr)), max(0, min(grid_size-1, c + dc))
        ns = nr * grid_size + nc
        r_cost = 10.0 if ns == 15 else -1.0
        P[s][a] = [(1.0, ns, r_cost, ns == 15)]

V = np.zeros(num_states)
policy = np.zeros(num_states, dtype=int)

# Policy Evaluation & Improvement
for _ in range(50):
    # Eval
    while True:
        delta = 0
        for s in range(num_states):
            a = policy[s]
            v = sum(p * (r + gamma * V[ns] * (not done)) for p, ns, r, done in P[s][a])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < 1e-4: break
    # Improvement
    for s in range(num_states):
        policy[s] = np.argmax([sum(p * (r + gamma * V[ns] * (not done)) for p, ns, r, done in P[s][a]) for a in range(4)])

print("Optimal Taxi Policy Grid (0:U, 1:D, 2:L, 3:R):\n", policy.reshape(4, 4))
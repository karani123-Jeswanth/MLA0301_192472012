import numpy as np

grid_size = 3
V = np.zeros((grid_size, grid_size))
gamma = 0.95
threshold = 1e-4

# Constraints: restricted locations
restricted_states = [(1, 1)]
goal_state = (2, 2)

rewards = np.full((grid_size, grid_size), -1.0)
rewards[goal_state] = 10.0
rewards[restricted_states[0]] = -100.0  # Constraint violation penalty

actions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Up, Down, Left, Right

# Value Iteration
for i in range(100):
    delta = 0
    next_V = np.copy(V)
    for r in range(grid_size):
        for c in range(grid_size):
            if (r, c) == goal_state or (r, c) in restricted_states:
                continue
            
            v_list = []
            for a in actions:
                nr, nc = max(0, min(grid_size-1, r + a[0])), max(0, min(grid_size-1, c + a[1]))
                v_list.append(rewards[nr, nc] + gamma * V[nr, nc])
                
            best_value = max(v_list)
            delta = max(delta, abs(best_value - V[r, c]))
            next_V[r, c] = best_value
    V = next_V
    if delta < threshold:
        break

print("Converged Value Function Space Matrix:")
print(np.round(V, 2))
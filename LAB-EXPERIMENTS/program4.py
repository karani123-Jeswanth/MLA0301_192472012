import numpy as np

# 3x3 Delivery Grid
grid = np.zeros((3, 3))
goal = (2, 2)
obstacles = [(1, 1)]
gamma = 0.95
rewards = np.full((3, 3), -1.0)
rewards[goal] = 10.0
rewards[1, 1] = -20.0

actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Bellman Optimality Value Iteration
for _ in range(100):
    new_grid = np.copy(grid)
    for r in range(3):
        for c in range(3):
            if (r, c) == goal or (r, c) in obstacles:
                continue
            q_vals = []
            for dr, dc in actions:
                nr, nc = max(0, min(2, r + dr)), max(0, min(2, c + dc))
                q_vals.append(rewards[nr, nc] + gamma * grid[nr, nc])
            new_grid[r, c] = max(q_vals)
    grid = new_grid

print("Optimal Value Matrix for Delivery Robot:\n", np.round(grid, 2))
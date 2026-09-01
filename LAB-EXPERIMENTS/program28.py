import numpy as np

grid = np.zeros((3, 3))
gamma = 0.9
reward_map = np.full((3, 3), -1)
reward_map[2, 2] = 100 # Destination

# Iteration
for _ in range(50):
    new_grid = np.zeros_like(grid)
    for r in range(3):
        for c in range(3):
            if (r, c) == (2, 2): continue
            vals = []
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = max(0, min(2, r+dr)), max(0, min(2, c+dc))
                vals.append(reward_map[nr, nc] + gamma * grid[nr, nc])
            new_grid[r, c] = max(vals)
    grid = new_grid

print("Optimal State-Value Map:\n", np.round(grid, 1))
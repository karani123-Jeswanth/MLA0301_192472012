import numpy as np

Q = np.zeros((4, 4, 4)) # 4x4 grid, 4 actions
food_pos = (3, 3)
ghost_pos = (1, 1)

def step(r, c, a):
    moves = [(-1,0), (1,0), (0,-1), (0,1)]
    nr = max(0, min(3, r + moves[a][0]))
    nc = max(0, min(3, c + moves[a][1]))
    if (nr, nc) == ghost_pos: return nr, nc, -100, True
    if (nr, nc) == food_pos: return nr, nc, 100, True
    return nr, nc, -1, False

# Mini Q-learning train
for _ in range(500):
    r, c = (0, 0)
    done = False
    while not done:
        a = np.random.randint(4) if np.random.rand() < 0.2 else np.argmax(Q[r, c])
        nr, nc, rew, done = step(r, c, a)
        Q[r, c, a] += 0.1 * (rew + 0.9 * np.max(Q[nr, nc]) - Q[r, c, a])
        r, c = nr, nc

print("Trained Q-Values at Start (0,0):", np.round(Q[0, 0], 2))
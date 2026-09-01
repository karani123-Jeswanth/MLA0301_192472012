import numpy as np

class SimpleNavEnv:
    def __init__(self):
        self.state_space = 16
        self.action_space = 4
        self.goal = 15

    def step(self, state, action):
        row, col = divmod(state, 4)
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        nr = max(0, min(3, row + moves[action][0]))
        nc = max(0, min(3, col + moves[action][1]))
        next_s = nr * 4 + nc
        reward = 100 if next_s == self.goal else -1
        done = (next_s == self.goal)
        return next_s, reward, done

env = SimpleNavEnv()
Q = np.zeros((env.state_space, env.action_space))
lr, gamma, eps = 0.1, 0.9, 0.2

for episode in range(500):
    s = 0
    done = False
    while not done:
        a = np.random.choice(4) if np.random.rand() < eps else np.argmax(Q[s])
        ns, r, done = env.step(s, a)
        Q[s, a] += lr * (r + gamma * np.max(Q[ns]) - Q[s, a])
        s = ns

print("Q-Learning Navigation Model Trained. Start State Values:\n", Q[0])
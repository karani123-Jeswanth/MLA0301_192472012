import numpy as np
from collections import defaultdict

class VacuumWorld:
    def __init__(self):
        self.states = [(r, c) for r in range(3) for c in range(3)]
        self.dirty_state = (0, 2)

    def generate_episode(self, policy):
        episode = []
        s = (2, 0) # start
        for _ in range(15):
            a = policy[s]
            moves = [(-1,0), (1,0), (0,-1), (0,1)]
            ns = (max(0, min(2, s[0] + moves[a][0])), max(0, min(2, s[1] + moves[a][1])))
            r = 10 if ns == self.dirty_state else -1
            episode.append((s, a, r))
            s = ns
            if s == self.dirty_state: break
        return episode

# MC Control
policy = {s: np.random.randint(4) for s in [(r, c) for r in range(3) for c in range(3)]}
Q = defaultdict(lambda: np.zeros(4))
returns = defaultdict(list)

for _ in range(1000):
    ep = VacuumWorld().generate_episode(policy)
    G = 0
    for s, a, r in reversed(ep):
        G = 0.9 * G + r
        returns[(s, a)].append(G)
        Q[s][a] = np.mean(returns[(s, a)])
        policy[s] = np.argmax(Q[s])

print("Monte Carlo Policy for Vacuum (Start state action):", policy[(2, 0)])
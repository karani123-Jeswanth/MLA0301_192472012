import numpy as np

# Simplified 3x3 Mini-Chess End-game MDP (King vs King/Pawn)
class MiniChessMDP:
    def __init__(self):
        # States: board positions (simplified as distance to promotion 0..3)
        self.states = [0, 1, 2, 3]  # 0: Start, 1: Mid-board, 2: Threat zone, 3: Checkmate/Win
        self.actions = ['forward', 'capture_threat', 'flank']
        self.gamma = 0.9

    def transitions(self, s, a):
        # returns list of (prob, next_state, reward)
        if s == 3:
            return [(1.0, 3, 0)]
        if a == 'forward':
            return [(0.7, min(s + 1, 3), 10 if s + 1 == 3 else 1), (0.3, max(s - 1, 0), -5)]
        elif a == 'capture_threat':
            return [(0.9, min(s + 1, 3), 5), (0.1, 0, -10)]
        elif a == 'flank':
            return [(0.8, s, 0), (0.2, min(s + 1, 3), 2)]

    def value_iteration(self, theta=1e-4):
        V = {s: 0.0 for s in self.states}
        while True:
            delta = 0
            for s in self.states:
                if s == 3:
                    continue
                v = V[s]
                q_vals = []
                for a in self.actions:
                    q = sum(p * (r + self.gamma * V[s_next]) for p, s_next, r in self.transitions(s, a))
                    q_vals.append(q)
                V[s] = max(q_vals)
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break
        return V

mdp = MiniChessMDP()
print("Optimal State Values for Chess MDP:", mdp.value_iteration())

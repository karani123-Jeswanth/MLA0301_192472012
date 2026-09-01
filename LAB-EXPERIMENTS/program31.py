import numpy as np

class TicTacToeSARSA:
    def __init__(self):
        self.Q = {}
        self.epsilon = 0.1
        self.alpha = 0.2
        self.gamma = 0.9

    def get_q(self, state, action):
        return self.Q.get((state, action), 0.0)

    def choose_action(self, state, legal_actions):
        if np.random.rand() < self.epsilon:
            return legal_actions[np.random.choice(len(legal_actions))]
        q_vals = [self.get_q(state, a) for a in legal_actions]
        return legal_actions[np.argmax(q_vals)]

    def update(self, s, a, r, s_next, a_next):
        q = self.get_q(s, a)
        q_next = self.get_q(s_next, a_next)
        self.Q[(s, a)] = q + self.alpha * (r + self.gamma * q_next - q)

agent = TicTacToeSARSA()
agent.update("000000000", 4, 1.0, "000010000", 0)
print("SARSA Action Value for Center Move:", agent.get_q("000000000", 4))
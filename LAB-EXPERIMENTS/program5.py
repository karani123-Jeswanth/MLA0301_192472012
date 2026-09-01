import numpy as np

class EpsilonGreedyBandit:
    def __init__(self, k_arms=5, epsilon=0.1):
        self.k = k_arms
        self.epsilon = epsilon
        self.q_est = np.zeros(k_arms)
        self.action_counts = np.zeros(k_arms)

    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)
        return np.argmax(self.q_est)

    def update(self, action, reward):
        self.action_counts[action] += 1
        self.q_est[action] += (reward - self.q_est[action]) / self.action_counts[action]

# True Click-Through Rates
true_ctr = [0.05, 0.12, 0.25, 0.08, 0.18]
bandit = EpsilonGreedyBandit(k_arms=len(true_ctr), epsilon=0.1)

for _ in range(1000):
    a = bandit.select_action()
    reward = 1 if np.random.rand() < true_ctr[a] else 0
    bandit.update(a, reward)

print("Estimated CTR for Ads:", np.round(bandit.q_est, 3))
print("Selected Best Ad:", np.argmax(bandit.q_est))
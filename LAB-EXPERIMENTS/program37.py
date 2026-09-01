import numpy as np

# Direct Policy Parameter Search (Random Shooting)
def evaluate_policy(theta, state):
    # Action = dot(theta, state)
    return np.dot(theta, state)

best_theta = np.random.randn(3)
best_reward = -np.inf

for _ in range(50):
    candidate_theta = np.random.randn(3)
    # Simulated fitness
    reward = -np.sum((candidate_theta - np.array([1.0, 0.5, -0.5]))**2)
    if reward > best_reward:
        best_reward = reward
        best_theta = candidate_theta

print("Best Discovered Policy Parameters:", np.round(best_theta, 3))
import numpy as np

# Comparative step on a single 5-state navigation line
states = 5
actions = 2 # 0: left, 1: right
gamma = 0.9
lr = 0.1

Q_sarsa = np.zeros((states, actions))
Q_learn = np.zeros((states, actions))
V_td0 = np.zeros(states)

# Step demonstration
s = 2; a = 1; r = 1; next_s = 3; next_a = 1

# TD(0)
V_td0[s] += lr * (r + gamma * V_td0[next_s] - V_td0[s])
# SARSA
Q_sarsa[s, a] += lr * (r + gamma * Q_sarsa[next_s, next_a] - Q_sarsa[s, a])
# Q-Learning
Q_learn[s, a] += lr * (r + gamma * np.max(Q_learn[next_s]) - Q_learn[s, a])

print("TD(0) V-values:", V_td0)
print("SARSA Q-values:", Q_sarsa[s])
print("Q-Learning Q-values:", Q_learn[s])
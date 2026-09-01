import numpy as np

# Multi-Agent Q-learning (Independent Q-Learning)
num_agents = 2
n_states = 4
n_actions = 2
Q_tables = [np.zeros((n_states, n_actions)) for _ in range(num_agents)]

# Joint step
s = [0, 0]
actions = [np.argmax(Q_tables[i][s[i]]) for i in range(num_agents)]
cooperative_reward = 10.0 if (actions[0] != actions[1]) else -5.0 # avoid collision

for i in range(num_agents):
    Q_tables[i][s[i], actions[i]] += 0.1 * (cooperative_reward - Q_tables[i][s[i], actions[i]])

print("MARL Agent 0 Q-table:", Q_tables[0][0])
print("MARL Agent 1 Q-table:", Q_tables[1][0])
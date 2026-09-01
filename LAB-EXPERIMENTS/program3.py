import numpy as np

states = ['LoadingBay', 'Aisle1', 'Aisle2', 'Storage']
actions = ['North', 'East']

# Transition probabilities T[state][action] = [(next_state, prob, reward)]
transitions = {
    'LoadingBay': {'North': [('Aisle1', 0.9, -1), ('LoadingBay', 0.1, -1)], 'East': [('Aisle2', 0.8, -1), ('LoadingBay', 0.2, -1)]},
    'Aisle1':     {'North': [('Storage', 0.95, 100), ('Aisle1', 0.05, -1)], 'East': [('Storage', 0.7, 100), ('Aisle2', 0.3, -1)]},
    'Aisle2':     {'North': [('Storage', 0.85, 100), ('Aisle2', 0.15, -1)], 'East': [('Storage', 0.9, 100), ('Aisle2', 0.1, -1)]},
    'Storage':    {'North': [('Storage', 1.0, 0)], 'East': [('Storage', 1.0, 0)]}
}

gamma = 0.9
V = {s: 0.0 for s in states}

for iteration in range(50):
    for s in states:
        if s == 'Storage': continue
        V[s] = max(sum(p * (r + gamma * V[ns]) for ns, p, r in transitions[s][a]) for a in actions)

print("Warehouse Robot Optimal Values:", V)
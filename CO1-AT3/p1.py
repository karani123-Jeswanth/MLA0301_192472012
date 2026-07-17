import numpy as np
import random

class GridWorld:
    def __init__(self, size=4):
        self.size = size
        self.state = (0, 0)
        self.goal = (3, 3)
        self.obstacles = [(1, 1), (2, 2)]
        
    def reset(self):
        self.state = (0, 0)
        return self.state
        
    def step(self, action):
        x, y = self.state
        if action == 0: x = max(0, x - 1)    # Up
        elif action == 1: y = min(self.size - 1, y + 1)  # Right
        elif action == 2: x = min(self.size - 1, x + 1)  # Down
        elif action == 3: y = max(0, y - 1)    # Left
        
        self.state = (x, y)
        
        if self.state == self.goal:
            return self.state, 100, True
        elif self.state in self.obstacles:
            return self.state, -100, True
        else:
            return self.state, -1, False

# Training Framework
env = GridWorld()
q_table = np.zeros((4, 4, 4)) # 4x4 Grid, 4 Actions
alpha, gamma, epsilon = 0.1, 0.99, 0.2

for episode in range(500):
    state = env.reset()
    done = False
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, 3)
        else:
            action = np.argmax(q_table[state[0], state[1]])
            
        next_state, reward, done = env.step(action)
        
        old_value = q_table[state[0], state[1], action]
        next_max = np.max(q_table[next_state[0], next_state[1]])
        
        q_table[state[0], state[1], action] = old_value + alpha * (reward + gamma * next_max - old_value)
        state = next_state

print("Training completed. Optimal policy map generated.")
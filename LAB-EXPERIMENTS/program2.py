import numpy as np

class SmartHomeEnv:
    def __init__(self, size=4):
        self.size = size
        self.goal = (size - 1, size - 1)
        self.obstacles = [(1, 1), (2, 2)]
        self.reset()

    def reset(self):
        self.pos = (0, 0)
        return self.pos

    def step(self, action):
        # 0: Up, 1: Down, 2: Left, 3: Right
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        nr = max(0, min(self.size - 1, self.pos[0] + moves[action][0]))
        nc = max(0, min(self.size - 1, self.pos[1] + moves[action][1]))
        next_pos = (nr, nc)

        if next_pos in self.obstacles:
            return self.pos, -10, False  # Obstacle penalty
        self.pos = next_pos
        if self.pos == self.goal:
            return self.pos, 50, True   # Goal reached
        return self.pos, -1, False      # Step cost

env = SmartHomeEnv()
print("Initialized Smart Home Robot Environment. Reset Pos:", env.reset())
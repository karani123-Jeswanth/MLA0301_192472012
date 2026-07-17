import numpy as np
import random

class WarehouseEnv:
    def __init__(self):
        self.max_time = 30 # Time constraint limit
        self.reset()
        
    def reset(self):
        self.time_left = self.max_time
        self.position = 0 # Dock location
        self.packages_delivered = 0
        return (self.position, self.time_left)
        
    def step(self, action):
        # Action 0: Deliver, Action 1: Recharge Dock
        self.time_left -= 5 # Every action costs time
        
        if self.time_left < 0:
            return (self.position, 0), -100, True # Failed constraint path
            
        if action == 0: # Delivery attempt
            self.position = 1 # Shifted to drop area
            self.packages_delivered += 1
            reward = 50
            done = False
        elif action == 1: # Dock recharge safely
            self.position = 0
            reward = 20
            done = True # Completed safely
            
        return (self.position, self.time_left), reward, done

env = WarehouseEnv()
state = env.reset()
print(f"Initial State Config: {state}")
state, reward, done = env.step(action=0)
print(f"Action: Deliver -> Next State Status: {state}, Step Reward: {reward}, Is Terminated: {done}")
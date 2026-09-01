import numpy as np

class EnergyManagementEnv:
    def __init__(self):
        self.battery = 50.0 # percentage
        self.max_grid_limit = 10.0 # kW

    def step(self, draw_power):
        # Action: draw power from grid
        penalty = 0
        if draw_power > self.max_grid_limit:
            penalty = -50 # Safe/Fair policy penalty
        cost = - (draw_power * 0.15) + penalty
        return cost

env = EnergyManagementEnv()
print("Safe Energy Cost for 8kW:", env.step(8))
print("Constrained Violation Cost for 14kW:", env.step(14))
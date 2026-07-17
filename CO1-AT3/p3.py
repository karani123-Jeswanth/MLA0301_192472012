# Conceptualizing the MDP setup matrix
import numpy as np

states = [(x, y, battery) for x in range(3) for y in range(3) for battery in [0, 50, 100]]
actions = ["Normal", "Eco"]

# Simple environment state transition lookup example
def mdp_step(state, action):
    x, y, battery = state
    if battery <= 0:
        return state, -500, True # Terminal constraint failure
        
    if action == "Normal":
        next_bat = max(0, battery - 50)
        next_state = (min(2, x+1), y, next_bat)
        reward = -10
    else: # Eco
        next_bat = max(0, battery - 5)
        next_state = (min(2, x+1), y, next_bat)
        reward = -2
        
    if next_state[0] == 2 and next_state[1] == 2: # Goal condition
        return next_state, 1000, True
        
    return next_state, reward, False

# Test run evaluation
curr_state = (0, 0, 100)
curr_state, r, done = mdp_step(curr_state, "Eco")
print(f"Transition Sample -> Next State: {curr_state}, Reward: {r}, Is Done: {done}")
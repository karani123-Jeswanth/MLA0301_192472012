import numpy as np

# Belief State Update for POMDP (Survivor present / absent)
belief = np.array([0.5, 0.5]) # [P(Survivor), P(Empty)]
# Observation probabilities given state: P(Observation=Beep | State)
O_beep = np.array([0.85, 0.10]) 

def update_belief(b, obs_prob):
    updated = b * obs_prob
    return updated / np.sum(updated)

print("Initial Belief:", belief)
belief_after_beep = update_belief(belief, O_beep)
print("Updated Belief after Sensor Beep:", np.round(belief_after_beep, 4))
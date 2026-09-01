import numpy as np

# State: (waiting_room_size, doctors_available)
# Action: 0: Standard Intake, 1: Overtime Extra Intake
def get_triage_reward(patients_waiting, action):
    cost = -5 * action # overtime cost
    health_penalty = -2 * patients_waiting # risk to waiting patients
    return cost + health_penalty

Q = np.zeros((10, 2))
for state in range(10): # waiting patients
    for action in [0, 1]:
        Q[state, action] = get_triage_reward(state, action)

print("Triage Q-Values (Wait=6): [Standard, Overtime] =", Q[6])
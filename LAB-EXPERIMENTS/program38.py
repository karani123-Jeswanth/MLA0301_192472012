import numpy as np

# 1D Grid Belief Filter (Localization)
grid_len = 5
belief = np.full(grid_len, 1.0 / grid_len)

# Robot moves right (transition model)
def predict(b):
    return np.roll(b, 1)

# Sensor reading: Distance to wall = 1
def update(b, observed_dist):
    sensor_model = np.array([0.1, 0.1, 0.2, 0.8, 0.1])
    b = b * sensor_model
    return b / np.sum(b)

belief = predict(belief)
belief = update(belief, 1)
print("POMDP Position Belief Distribution:", np.round(belief, 3))
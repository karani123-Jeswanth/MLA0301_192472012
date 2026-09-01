import numpy as np

# Comparison of Vanilla Policy Gradient vs Baseline Policy Gradient
trajectories = [[-1.2, 0.5, 2.0], [-2.0, -1.0, 0.0], [1.0, 1.5, 3.0]] # Rewards
returns = [sum(t) for t in trajectories]

baseline = np.mean(returns)
adv_with_baseline = [G - baseline for G in returns]

print("Raw Returns (Vanilla PG):", returns)
print("Adjusted Advantages with Baseline (Reduced Variance):", adv_with_baseline)
import numpy as np

price_points = [10, 20, 30, 40]
# Purchase probabilities given price
conversion = [0.8, 0.5, 0.3, 0.1]
exp_revenue = [p * c for p, c in zip(price_points, conversion)]

# UCB for Pricing
counts = np.zeros(len(price_points))
revenue = np.zeros(len(price_points))

for t in range(1, 500):
    arm = np.argmax(revenue / (counts + 1e-5) + np.sqrt(2 * np.log(t) / (counts + 1e-5))) if t > len(price_points) else t - 1
    sold = np.random.rand() < conversion[arm]
    r = price_points[arm] if sold else 0
    counts[arm] += 1
    revenue[arm] += r

print("Optimal Revenue Price Point Identified: $", price_points[np.argmax(revenue/counts)])
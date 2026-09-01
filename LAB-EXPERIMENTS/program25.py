import numpy as np

ads_ctr = [0.04, 0.15, 0.08]
n_arms = len(ads_ctr)
T = 1000

# 1. UCB
counts = np.zeros(n_arms)
rewards = np.zeros(n_arms)
for t in range(1, T + 1):
    if t <= n_arms:
        arm = t - 1
    else:
        ucb_vals = rewards / (counts + 1e-5) + np.sqrt(2 * np.log(t) / (counts + 1e-5))
        arm = np.argmax(ucb_vals)
    r = 1 if np.random.rand() < ads_ctr[arm] else 0
    counts[arm] += 1
    rewards[arm] += r

# 2. Thompson Sampling
alpha, beta = np.ones(n_arms), np.ones(n_arms)
for _ in range(T):
    samples = [np.random.beta(alpha[i], beta[i]) for i in range(n_arms)]
    arm = np.argmax(samples)
    r = 1 if np.random.rand() < ads_ctr[arm] else 0
    alpha[arm] += r
    beta[arm] += (1 - r)

print("UCB Selected Counts:", counts)
print("Thompson Sampling Estimated Beta Alpha Means:", alpha / (alpha + beta))
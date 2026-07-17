import numpy as np

def run_bandit(epsilon, trials=500, k=5):
    true_rewards = np.array([1.0, 1.5, 3.0, 2.0, 0.5]) # Arm 2 is best
    q_estimates = np.zeros(k)
    action_counts = np.zeros(k)
    total_reward = 0
    
    for t in range(trials):
        if np.random.rand() < epsilon:
            action = np.random.randint(k)
        else:
            action = np.argmax(q_estimates)
            
        reward = true_rewards[action] + np.random.normal(0, 0.1)
        total_reward += reward
        
        action_counts[action] += 1
        q_estimates[action] += (reward - q_estimates[action]) / action_counts[action]
        
    return total_reward

epsilons = [0.1, 0.3, 0.5]
for eps in epsilons:
    cum_reward = run_bandit(eps)
    print(f"Epsilon: {eps} | Total Cumulative Reward over 500 trials: {cum_reward:.2f}")
import gymnasium as gym

# Create FrozenLake environment
env = gym.make("FrozenLake-v1")

# Reset environment
state, info = env.reset()

print("Initial State:", state)

# Take a random action
action = env.action_space.sample()

next_state, reward, terminated, truncated, info = env.step(action)

print("Action Taken:", action)
print("Next State:", next_state)
print("Reward:", reward)
print("Episode Finished:", terminated or truncated)

env.close()
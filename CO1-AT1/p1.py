import gymnasium as gym

# Create FrozenLake environment
env = gym.make("FrozenLake-v1")

# Reset environment
state, info = env.reset()

print("Environment created successfully!")
print("Initial State:", state)

env.close()
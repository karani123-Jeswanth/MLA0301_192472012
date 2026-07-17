import gymnasium as gym
env = gym.make("FrozenLake-v1")
state, info = env.reset()
print("Starting State:", state)
for i in range(5):
    action = env.action_space.sample()

    next_state, reward, terminated, truncated, info = env.step(action)

    print("Step:", i + 1)
    print("Action:", action)
    print("State:", next_state)
    print("Reward:", reward)

    state = next_state

    if terminated or truncated:
        break

env.close()
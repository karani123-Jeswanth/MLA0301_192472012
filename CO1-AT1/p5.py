gamma = 0.9

rewards = [5, 10, 15]
values = [0, 0, 0]

for i in range(len(rewards)):
    values[i] = rewards[i] + gamma * values[i]

print("Updated State Values")

for i in range(len(values)):
    print("State", i, ":", values[i])
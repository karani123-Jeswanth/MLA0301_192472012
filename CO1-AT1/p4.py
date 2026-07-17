states = ["A", "B", "C"]

transitions = {
    "A": ("B", 5),
    "B": ("C", 10),
    "C": ("A", 2)
}

current_state = "A"

for i in range(3):
    next_state, reward = transitions[current_state]

    print("Current State:", current_state)
    print("Next State:", next_state)
    print("Reward:", reward)
    print()

    current_state = next_state
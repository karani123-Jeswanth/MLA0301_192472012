import numpy as np

# Simulate portfolio value under 3 alternative allocations
weights = np.array([[0.6, 0.4], [0.5, 0.5], [0.3, 0.7]])
asset_returns = np.array([0.08, 0.12]) # Mean annual returns

expected_values = weights @ asset_returns
print("Portfolio Equivalent Predicted Yields:")
for i, val in enumerate(expected_values):
    print(f"Strategy {i+1}: {val*100:.2f}% Expected Return")
import numpy as np
import matplotlib.pyplot as plt

# Define parameters
n = 20  # Upper bound of uniform distribution (1 to n)
num_samples = 1000000000  # Number of random draws

# Simulate uniform discrete random variable Z
Z_samples = np.random.randint(1, n+1, num_samples)

# Compute probability P(Z <= 5)
p_le_5_simulated = np.sum(Z_samples <= 5) / num_samples
p_le_5_theoretical = 5 / n

# Print probability results
print("Simulated P(Z ≤ 5):", p_le_5_simulated)
print("Theoretical P(Z ≤ 5):", p_le_5_theoretical)
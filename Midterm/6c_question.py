import numpy as np

# Simulation-based check
num_samples = 100_000_000
samples = np.random.uniform(low=2, high=5, size=num_samples)

print(f"Simulated E[W]  = {samples.mean():.4f}")
print(f"Simulated Var(W) = {samples.var(ddof=1):.4f}")
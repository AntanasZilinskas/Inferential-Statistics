import numpy as np

num_samples = 10_000_000
samples = np.random.uniform(2, 5, size=num_samples)

prob_greater_4 = np.mean(samples > 4)
print(f"Pr(W > 4) (simulated) = {prob_greater_4:.3f}")
import numpy as np

# Fixed seed for reproducibility
np.random.seed(1337)

p_theory = 10/36  # ~0.2778

# List of sample sizes to test
sample_sizes = [10_000, 100_000, 1_000_000, 2_000_000]

for N in sample_sizes:
    # Simulate N rolls
    die1 = np.random.randint(1, 7, size=N)
    die2 = np.random.randint(1, 7, size=N)
    sums = die1 + die2
    
    # Calculate the empirical probability for sum >= 9
    p_est = np.mean(sums >= 9)
    
    # Compute the absolute difference from the theoretical probability
    diff = abs(p_est - p_theory)
    print(f"N={N:9d}: Empirical p={p_est:.5f}, "
          f"Difference={diff:.5e}")

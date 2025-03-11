import numpy as np
import matplotlib.pyplot as plt

# We will run multiple simulations, each with N dice rolls
N = 10000          # number of dice rolls per simulation
num_runs = 10000 # total number of simulation runs

# List to store the empirical probability estimate for each run
prob_estimates = []

for _ in range(num_runs):
    # Roll two dice for N trials (no fixed seed to ensure randomness)
    die1 = np.random.randint(1, 7, size=N)
    die2 = np.random.randint(1, 7, size=N)
    sums = die1 + die2
    
    # Calculate the probability that the sum is >= 9 for these N rolls
    prob_est = np.sum(sums >= 9) / N
    prob_estimates.append(prob_est)

# Compute statistics across all runs
mean_estimate = np.mean(prob_estimates)
std_estimate = np.std(prob_estimates, ddof=1)  # Sample standard deviation

# Print results including the standard deviation
print(f"Number of independent runs: {num_runs}")
print(f"Each run has N = {N} rolls.")
print(f"Mean of empirical estimates: {mean_estimate:.5f}")
print(f"Standard deviation of empirical estimates: {std_estimate:.5f}")

# Plot a histogram with increased number of bins (50) and bars touching each other (rwidth=1.0)
plt.hist(prob_estimates, bins=50, color='skyblue', edgecolor='black', rwidth=1.0)
plt.title(f"Histogram of Empirical Probability Estimates ({num_runs} runs, N = {N})")
plt.xlabel("Empirical Probability (sum >= 9)")
plt.ylabel("Frequency")
plt.grid(axis='y', alpha=0.75)

# Add vertical lines for the mean and ±1 standard deviation
plt.axvline(mean_estimate, color='red', linestyle='dashed', linewidth=1.5,
            label=f'Mean ({mean_estimate:.4f})')
plt.axvline(mean_estimate + std_estimate, color='green', linestyle='dashed', linewidth=1.5,
            label=f'Mean + 1 SD ({mean_estimate + std_estimate:.4f})')
plt.axvline(mean_estimate - std_estimate, color='green', linestyle='dashed', linewidth=1.5,
            label=f'Mean - 1 SD ({mean_estimate - std_estimate:.4f})')

plt.legend()
plt.show()
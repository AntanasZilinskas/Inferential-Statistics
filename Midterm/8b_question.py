import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

p = 0.4
num_runs = 20        # Number of separate runs (curves)
total_draws = 10_000 # Number of draws per run

plt.figure(figsize=(10, 6))

for run_idx in range(num_runs):
    # Generate 'total_draws' Bernoulli(0.4) samples
    samples = np.random.binomial(1, p, size=total_draws)
    
    # cumulative_sums[i] = sum of samples up to index i
    cumulative_sums = np.cumsum(samples)
    
    # running_avg[i] = average of first (i+1) samples
    running_avg = cumulative_sums / np.arange(1, total_draws + 1)
    
    # Plot this run's running average as one line
    plt.plot(running_avg, alpha=0.7, label=f"Run {run_idx+1}")

# Draw a reference line at y = 0.4 (true p)
plt.axhline(y=p, color='red', linestyle='--', linewidth=2, label="True p=0.4")

plt.title(f"{num_runs} Runs of Running Averages for Bernoulli(0.4)")
plt.xlabel("Number of Bernoulli Trials")
plt.ylabel("Running Average")
plt.grid(True, alpha=0.3)
plt.legend(loc="best")
plt.show()
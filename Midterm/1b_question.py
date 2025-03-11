import numpy as np
import matplotlib.pyplot as plt

# Theoretical probability (10 favorable outcomes out of 36 possible) for dice sum >= 9
theoretical = 0.2778

# Simulation settings
max_trials = 1_000_000_000        # Maximum number of trials
min_trials = 10000            # Start searching after this many trials
convergence_length = 10000      # Must hold for 100 consecutive trials

np.random.seed(1337)  # Set seed for reproducibility

# Simulate dice rolls
die1 = np.random.randint(1, 7, size=max_trials)
die2 = np.random.randint(1, 7, size=max_trials)
sums = die1 + die2

# Boolean array: True if sum >= 9
successes = (sums >= 9)

# Calculate the cumulative number of successes and running probability
cumulative_successes = np.cumsum(successes)
trial_numbers = np.arange(1, max_trials + 1)
probabilities = cumulative_successes / trial_numbers

# Round the running probability to 4 decimal places
rounded_prob = np.round(probabilities, 4)

# Search for the earliest trial (with at least min_trials) where the rounded probability
# remains equal to the theoretical value for the next 'convergence_length' trials.
found = False
for i in range(min_trials - 1, max_trials - convergence_length + 1):
    if np.all(rounded_prob[i:i+convergence_length] == theoretical):
        trial_needed = i + 1  # Convert index to trial count
        print(f"Convergence: {convergence_length} consecutive trials starting at trial {trial_needed} all report a probability of {rounded_prob[i]:.4f}.")
        found = True
        break

if not found:
    print("No convergence found up to", max_trials, "trials.")
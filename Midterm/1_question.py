import numpy as np
import matplotlib.pyplot as plt

# Fix the random seed for reproducibility
np.random.seed(1337)

# Number of trials
N = 10000

# Simulate rolling each die N times
die1 = np.random.randint(1, 7, size=N)
die2 = np.random.randint(1, 7, size=N)

# Compute the sum of the faces for each roll
sums = die1 + die2

# Print a brief report of results
print(f"First 10 sums: {sums[:10]}")
print(f"Minimum sum observed: {sums.min()}")
print(f"Maximum sum observed: {sums.max()}")

# Plot a histogram of the sums
plt.hist(sums, bins=range(2, 14), align='left', 
         color='skyblue', edgecolor='black')
plt.title("Distribution of Sums (Two Dice, 10,000 Rolls)")
plt.xlabel("Sum of two dice")
plt.ylabel("Frequency")
plt.xticks(range(2, 13))
plt.grid(axis='y', alpha=0.75)
plt.show()

# Part b

# Estimating probability that dice sum is >= 9
successful_rolls = np.sum(sums >= 9)  # counting the number of rolls with sum >= 9
total_rolls = len(sums)               # total number of dice rolls
prob_est = successful_rolls / total_rolls  # probability calculation

print("Estimated probability (sum >= 9):", prob_est)
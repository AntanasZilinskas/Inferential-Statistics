import numpy as np
import matplotlib.pyplot as plt

# Define probability values
p_values = np.linspace(0, 1, 100)
variances = p_values * (1 - p_values)  # Compute variance p(1 - p)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(p_values, variances, label=r'Variance $p(1 - p)$', color='b')

# Highlight the peak at p = 0.5
plt.axvline(x=0.5, linestyle="--", color="red", label="Max Variance at $p=0.5$")
plt.axvline(x=0.55, linestyle="--", color="green", label="Variance at $p=0.55$")

# Labels
plt.xlabel("Probability of Heads (p)")
plt.ylabel("Variance")
plt.title("Variance of a Bernoulli Random Variable")
plt.legend()
plt.grid()

# Show plot
plt.show()
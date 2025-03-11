import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(1337)  # For reproducibility

p = 0.4
num_experiments = 2000
sample_size = 100

# We'll store all 2,000 sample means in this list
means = []
for _ in range(num_experiments):
    # Generate 100 Bernoulli(0.4) samples
    samples = np.random.binomial(1, p, size=sample_size)
    # Compute the mean of these 100 samples
    means.append(np.mean(samples))

means = np.array(means)

# Plot a histogram of the sample means
plt.hist(means, bins=30, color='skyblue', edgecolor='black', density=True)
plt.title("Histogram of 2,000 Sample Means (Each of 100 Bernoulli(0.4) Trials)")
plt.xlabel("Sample Mean")
plt.ylabel("Density")

# Overlay the theoretical normal PDF for comparison
mu = 0.4
var = p * (1 - p) / sample_size  # 0.4 * 0.6 / 100 = 0.0024
sigma = np.sqrt(var)

# Generate points for plotting the normal curve
x = np.linspace(means.min(), means.max(), 200)
pdf_theoretical = norm.pdf(x, loc=mu, scale=sigma)
plt.plot(x, pdf_theoretical, 'r--', linewidth=2, label='N(0.4, 0.0024)')

plt.legend()
plt.show()
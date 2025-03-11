from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np

# Define the parameters for the normal distribution: mean (mu) and standard deviation (sigma)
mu, sigma = 50, 4

# Calculate the probability that a normal random variable with mean 50 and standard deviation 4
# falls between 48 and 55.
# norm.cdf(55, loc=mu, scale=sigma) computes P(S ≤ 55)
# norm.cdf(48, loc=mu, scale=sigma) computes P(S ≤ 48)
# Their difference gives P(48 < S ≤ 55)
prob_48_to_55 = norm.cdf(55, loc=mu, scale=sigma) - norm.cdf(48, loc=mu, scale=sigma)

print(prob_48_to_55)
# Terminal output:
# 0.5858126876071578

# Define range for x-values for plotting (from mu-4*sigma to mu+4*sigma)
x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)
y = norm.pdf(x, loc=mu, scale=sigma)

plt.figure(figsize=(8, 5))
plt.plot(x, y, color='blue', label='Normal PDF')

# Shade the area under the curve between x = 48 and x = 55
x_fill = np.linspace(48, 55, 1000)
y_fill = norm.pdf(x_fill, loc=mu, scale=sigma)
plt.fill_between(x_fill, y_fill, color='skyblue', alpha=0.6, label='Area: 48 to 55')

# Draw vertical lines and mark endpoints at x=48 and x=55
plt.axvline(48, color='red', linestyle='--', label='x = 48')
plt.axvline(55, color='green', linestyle='--', label='x = 55')
plt.plot(48, norm.pdf(48, loc=mu, scale=sigma), 'ro')  # Red dot at 48
plt.plot(55, norm.pdf(55, loc=mu, scale=sigma), 'go')  # Green dot at 55

plt.title('Normal Distribution (μ = 50, σ = 4)')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()

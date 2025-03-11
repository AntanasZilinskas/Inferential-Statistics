from scipy.stats import norm

mu, sigma = 50, 4
prob_le_45 = norm.cdf(45, loc=mu, scale=sigma)
print(prob_le_45)
# Terminal output:
# 0.10564977366685535
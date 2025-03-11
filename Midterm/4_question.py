from scipy.stats import binom

n = 30
p = 0.9

# Method 1: Direct summation of PMF for k=28,29,30
p_28 = binom.pmf(28, n, p)
p_29 = binom.pmf(29, n, p)
p_30 = binom.pmf(30, n, p)
prob_ge_28 = p_28 + p_29 + p_30

print("P(Y >= 28) [summation]:", prob_ge_28)

# Method 2: Use the CDF to compute 1 - P(Y <= 27)
prob_ge_28_cdf = 1 - binom.cdf(27, n, p)
print("P(Y >= 28) [1 - CDF]:", prob_ge_28_cdf)
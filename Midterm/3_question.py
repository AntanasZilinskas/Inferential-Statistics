from scipy.stats import bernoulli

# Define the probability of success
p = 0.55
# Create a Bernoulli random variable object
rv = bernoulli(p)

# Compute Pr(X = 1)
prob_head = rv.pmf(1)
print("Pr(X = 1) =", prob_head)
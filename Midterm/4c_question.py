from scipy.stats import binom

n = 30
p = 0.9

mean_Y = binom.mean(n, p)
var_Y = binom.var(n, p)

print(f"E[Y] = {mean_Y:.3f}")
print(f"Var(Y) = {var_Y:.3f}")
print(f"Std Dev of Y = {var_Y**0.5:.3f}")
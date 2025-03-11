import numpy as np

# We can simulate or compute directly
# 1) Direct computation:
n = 20
E_Z = (n+1)/2
Var_Z = (n**2 - 1)/12
print(f"Analytic E[Z] = {E_Z:.2f}")
print(f"Analytic Var(Z) = {Var_Z:.2f}")

# 2) Simulation approach (for illustration)
num_trials = 100_000_000
samples = np.random.randint(1, n+1, size=num_trials)
print(f"Simulated E[Z] = {samples.mean():.2f}")
print(f"Simulated Var(Z) = {samples.var(ddof=1):.2f}")
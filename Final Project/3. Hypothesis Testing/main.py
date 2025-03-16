"""
superhost_price_test.py

Hypothesis:
 H0: mean(Price_superhost) <= mean(Price_non_superhost)
 HA: mean(Price_superhost) > mean(Price_non_superhost)

We use:
  1) Welch's t-test (one-sided).
  2) Mann–Whitney U test for additional check (one-sided).
  3) Bootstrapped 95% CI for difference in means (superhost - non-superhost).
"""

import pandas as pd
import numpy as np
import os
from scipy.stats import ttest_ind, mannwhitneyu
import matplotlib.pyplot as plt
import scipy.stats as st

# =================== SETTINGS ===================
# Get the current directory and construct path to the CSV file in the parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
CSV_FILE = os.path.join(parent_dir, "listings_with_goodness.csv")  # Path to your CSV
COL_PRICE        = "price"
COL_SUPERHOST    = "host_is_superhost"  # e.g. 't' or 'f'
SUPERHOST_VAL    = "t"
NONSUPER_VAL     = "f"
ALPHA            = 0.05
USE_LOG_PRICE    = False   # If True, analyze log(price) to reduce skew
REMOVE_OUTLIERS  = True    # If desired, remove ±3σ outliers in price
# =================================================

def remove_3sigma_outliers(arr):
    mean_val = np.mean(arr)
    std_val  = np.std(arr, ddof=1)
    low_cut  = mean_val - 3 * std_val
    high_cut = mean_val + 3 * std_val
    return arr[(arr >= low_cut) & (arr <= high_cut)]

def bootstrap_mean_diff(data1, data2, n_boot=2000, random_state=42):
    """
    Returns a (low, high) 95% CI for the difference in means (data1 - data2) 
    via simple bootstrap resampling.
    """
    rng = np.random.default_rng(random_state)
    diffs = []
    size1 = len(data1)
    size2 = len(data2)
    for _ in range(n_boot):
        sample1 = rng.choice(data1, size1, replace=True)
        sample2 = rng.choice(data2, size2, replace=True)
        diffs.append(np.mean(sample1) - np.mean(sample2))
    diffs = np.array(diffs)
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)
    return ci_lower, ci_upper

def plot_distribution(data_sh, data_nsh, mean_sh, mean_nsh, diff_means, t_stat, t_crit, ci_lower, ci_upper, save_path='distribution_plot.png'):
    """
    Plot the distribution of superhost and non-superhost prices, along with the test statistic,
    critical value, and confidence interval. Save the plot to a file.
    """
    plt.figure(figsize=(12, 6))
    plt.hist(data_sh, bins=30, alpha=0.5, label='Superhost', color='skyblue', density=True)
    plt.hist(data_nsh, bins=30, alpha=0.5, label='Non-Superhost', color='salmon', density=True)
    plt.axvline(mean_sh, color='blue', linestyle='dashed', linewidth=2, label=f'Mean Superhost: {mean_sh:.2f}')
    plt.axvline(mean_nsh, color='red', linestyle='dashed', linewidth=2, label=f'Mean Non-Superhost: {mean_nsh:.2f}')
    plt.axvline(ci_lower, color='green', linestyle='dashed', linewidth=2, label=f'95% CI Lower: {ci_lower:.2f}')
    plt.axvline(ci_upper, color='green', linestyle='dashed', linewidth=2, label=f'95% CI Upper: {ci_upper:.2f}')
    plt.axvline(diff_means, color='purple', linestyle='solid', linewidth=2, label=f'Diff Means: {diff_means:.2f}')
    plt.title('Distribution of Prices: Superhost vs Non-Superhost')
    plt.xlabel('Price')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def main():
    print(f"Reading dataset from: {CSV_FILE}")
    df = pd.read_csv(CSV_FILE)
    if COL_PRICE not in df.columns or COL_SUPERHOST not in df.columns:
        print("Error: Missing required columns. Exiting.")
        return

    # Extract superhost vs. non-superhost
    df_sh = df[df[COL_SUPERHOST] == SUPERHOST_VAL]
    df_nsh = df[df[COL_SUPERHOST] == NONSUPER_VAL]
    data_sh  = df_sh[COL_PRICE].dropna().values
    data_nsh = df_nsh[COL_PRICE].dropna().values

    # Remove outliers if desired
    if REMOVE_OUTLIERS:
        before_sh, before_nsh = len(data_sh), len(data_nsh)
        data_sh  = remove_3sigma_outliers(data_sh)
        data_nsh = remove_3sigma_outliers(data_nsh)
        after_sh, after_nsh = len(data_sh), len(data_nsh)
        print(f"Removed outliers 3σ: superhost from {before_sh} -> {after_sh}, non-superhost from {before_nsh} -> {after_nsh}")

    # Optionally log-transform the price to reduce skew
    if USE_LOG_PRICE:
        data_sh  = np.log1p(data_sh)
        data_nsh = np.log1p(data_nsh)

    # =============== Welch's T-Test (one-sided) ===============
    # default is two-sided => we interpret the p-value
    t_stat, p_val_two_sided = ttest_ind(data_sh, data_nsh, equal_var=False)
    mean_sh = np.mean(data_sh)
    mean_nsh= np.mean(data_nsh)
    diff_means = mean_sh - mean_nsh

    # For a one-sided test: "superhost > non-superhost"
    if diff_means > 0:
        p_val_one_sided = p_val_two_sided / 2.0
    else:
        p_val_one_sided = 1.0 - (p_val_two_sided / 2.0)

    print("\n=== Welch's T-Test (One-Sided) for Superhost vs Non-Superhost ===")
    print(f"Mean(Superhost) = {mean_sh:.2f}, Mean(Non-SH) = {mean_nsh:.2f}, diff={diff_means:.2f}")
    print(f"T-statistic = {t_stat:.4f}, p-value(one-sided)={p_val_one_sided:.5g}")

    if p_val_one_sided < ALPHA:
        print("=> Reject H0 => Superhosts have significantly higher mean price.")
    else:
        print("=> Fail to reject H0 => No strong evidence superhosts have higher mean price.")

    # =============== Mann–Whitney U Test (one-sided) ===============
    # "Superhost > Non-superhost" => scipys default is two-sided => we interpret
    u_stat, p_val_mw_two_sided = mannwhitneyu(data_sh, data_nsh, alternative='two-sided')
    # If the median of data_sh is > data_nsh, half the two-sided p for one-sided
    median_sh = np.median(data_sh)
    median_nsh= np.median(data_nsh)
    if median_sh > median_nsh:
        p_val_mw_onesided = p_val_mw_two_sided / 2.0
    else:
        p_val_mw_onesided = 1.0 - (p_val_mw_two_sided / 2.0)

    print("\n=== Mann–Whitney U Test (One-Sided) for Superhost vs Non-Superhost ===")
    print(f"Median(Superhost)={median_sh:.2f}, Median(Non-SH)={median_nsh:.2f}")
    print(f"U-stat={u_stat:.2f}, p-value(one-sided)={p_val_mw_onesided:.5g}")
    if p_val_mw_onesided < ALPHA:
        print("=> Reject H0 => Superhosts have significantly greater distribution stochastically.")
    else:
        print("=> Fail to reject H0 => No strong evidence superhosts have higher distribution.")

    # =============== Bootstrap 95% CI for difference ================
    boot_low, boot_high = bootstrap_mean_diff(data_sh, data_nsh, n_boot=2000)
    print("\n=== Bootstrapped 95% CI for difference in means (superhost - non-superhost) ===")
    print(f"95% CI: [{boot_low:.2f}, {boot_high:.2f}]")

    # If we used a log transform, interpret carefully if exponentiating is desired.
    # Done.
    print("\nTest concluded successfully.")

    # Calculate degrees of freedom for Welch's t-test
    var1 = np.var(data_sh, ddof=1)
    var2 = np.var(data_nsh, ddof=1)
    n1, n2 = len(data_sh), len(data_nsh)
    df_approx = (var1/n1 + var2/n2)**2 / \
                ((var1**2)/(n1**2*(n1-1)) + (var2**2)/(n2**2*(n2-1)))
    
    # Calculate the critical t-value for a two-sided test
    t_crit_2sided = st.t.ppf(1 - ALPHA/2, df_approx)

    # Ensure the figures directory exists
    figures_dir = os.path.join(current_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Plot the distribution and test results
    plot_distribution(data_sh, data_nsh, mean_sh, mean_nsh, diff_means, t_stat, t_crit_2sided, boot_low, boot_high, save_path=os.path.join(figures_dir, 'superhost_vs_non_superhost.png'))

    # Done.
    print("\nTest concluded successfully.")

if __name__ == "__main__":
    main()
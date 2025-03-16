#!/usr/bin/env python3

"""
superhost_price_test.py

Hypothesis:
  H0: mean(Price_superhost) <= mean(Price_non_superhost)
  HA: mean(Price_superhost) > mean(Price_non_superhost)

We demonstrate:
  1) Welch's t-test (one-sided).
  2) Mann–Whitney U test (non-parametric) for confirmation.
  3) A bootstrapped 95% CI for difference in means.

Additionally uses a "current_dir" approach to locate the CSV.
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu
import matplotlib.pyplot as plt

# =================== PATH SETTINGS ===================
current_dir  = os.path.dirname(os.path.abspath(__file__))
CSV_FILE     = os.path.join(os.path.dirname(current_dir), "listings_with_goodness.csv")
# =====================================================

# =================== COLUMN SETTINGS =================
COL_PRICE     = "price"
COL_SUPERHOST = "host_is_superhost"  # e.g. 't' or 'f'
SPARENT_VAL   = "t"
NONSUPER_VAL  = "f"
ALPHA         = 0.05
USE_LOG_PRICE = False        # set True if your price is heavily skewed
REMOVE_OUTLIERS  = True      # remove ±3σ outliers if you want
# =====================================================

def remove_3sigma_outliers(arr):
    mean_val = np.mean(arr)
    std_val  = np.std(arr, ddof=1)
    low_cut  = mean_val - 3 * std_val
    high_cut = mean_val + 3 * std_val
    return arr[(arr >= low_cut) & (arr <= high_cut)]

def bootstrap_mean_diff(data1, data2, n_boot=2000, random_state=42):
    """
    Returns a (low, high) 95% CI for the difference in means (data1 - data2)
    via a simple bootstrap resampling approach.
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

def main():
    print(f"Reading dataset from: {CSV_FILE}")
    df = pd.read_csv(CSV_FILE, low_memory=False)
    if COL_PRICE not in df.columns or COL_SUPERHOST not in df.columns:
        print("Error: Missing required columns. Exiting.")
        return

    df_sh  = df[df[COL_SUPERHOST] == SPARENT_VAL]
    df_nsh = df[df[COL_SUPERHOST] == NONSUPER_VAL]
    data_sh  = df_sh[COL_PRICE].dropna().values
    data_nsh = df_nsh[COL_PRICE].dropna().values

    # Optional outlier removal
    if REMOVE_OUTLIERS:
        before_sh  = len(data_sh)
        before_nsh = len(data_nsh)
        data_sh  = remove_3sigma_outliers(data_sh)
        data_nsh = remove_3sigma_outliers(data_nsh)
        after_sh  = len(data_sh)
        after_nsh = len(data_nsh)
        print(f"Removed outliers ±3σ: Superhost from {before_sh} -> {after_sh}, "
              f"Non-superhost from {before_nsh} -> {after_nsh}")

    # Optional log transform to handle skewness
    if USE_LOG_PRICE:
        data_sh  = np.log1p(data_sh)
        data_nsh = np.log1p(data_nsh)

    # Welch's t-test (one-sided: superhost mean > non-superhost mean)
    t_stat, p_val_two_sided = ttest_ind(data_sh, data_nsh, equal_var=False)
    mean_sh  = np.mean(data_sh)
    mean_nsh = np.mean(data_nsh)
    diff_means = mean_sh - mean_nsh

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

    # Mann–Whitney U test (one-sided)
    u_stat, p_val_mw_two_sided = mannwhitneyu(data_sh, data_nsh, alternative='two-sided')
    median_sh  = np.median(data_sh)
    median_nsh = np.median(data_nsh)
    if median_sh > median_nsh:
        p_val_mw_onesided = p_val_mw_two_sided / 2.0
    else:
        p_val_mw_onesided = 1.0 - (p_val_mw_two_sided / 2.0)

    print("\n=== Mann–Whitney U Test (One-Sided) for Superhost vs Non-Superhost ===")
    print(f"Median(SH)={median_sh:.2f}, Median(Non-SH)={median_nsh:.2f}")
    print(f"U-stat={u_stat:.2f}, p-value(one-sided)={p_val_mw_onesided:.5g}")

    if p_val_mw_onesided < ALPHA:
        print("=> Reject H0 => Superhosts have significantly greater distribution stochastically.")
    else:
        print("=> Fail to reject H0 => No strong evidence superhosts have higher distribution.")

    # Bootstrapped 95% CI for difference (superhost - non-superhost)
    boot_low, boot_high = bootstrap_mean_diff(data_sh, data_nsh, n_boot=2000)
    print("\n=== Bootstrapped 95% CI for difference in means (superhost - non-superhost) ===")
    print(f"95% CI: [{boot_low:.2f}, {boot_high:.2f}]")

    print("\nTest concluded successfully.")

if __name__ == "__main__":
    main()
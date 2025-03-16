"""
superhost_price_test.py

Hypothesis:
 H0: mean(Price_superhost) <= mean(Price_non_superhost)
 HA: mean(Price_superhost) > mean(Price_non_superhost)

We use:
  1) Welch's t-test (one-sided).
  2) Mann–Whitney U test for additional check (one-sided).
  3) Bootstrapped 95% CI for difference in means (superhost - non-superhost).

Enhanced plotting to improve visibility:
 - Two subplots: one up to the 99th percentile (zoomed in),
   and one for the entire range.
 - Hist + optional KDE overlay for each group.
 - Distinct color with alpha blending to see overlaps.
 - Optionally log-scale the x-axis if data is extremely skewed.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, mannwhitneyu
import scipy.stats as st
from math import sqrt

# =================== SETTINGS ===================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
CSV_FILE = os.path.join(parent_dir, "listings_with_goodness.csv")  # Path to your CSV

COL_PRICE        = "price"
COL_SUPERHOST    = "host_is_superhost"  # e.g. 't' or 'f'
SUPERHOST_VAL   = "t"
NONSUPER_VAL     = "f"
ALPHA            = 0.05
USE_LOG_PRICE    = False   # If True, analyze log(price) to reduce skew
REMOVE_OUTLIERS  = True    # If desired, remove ±3σ outliers
SHOW_KDE         = True    # Overlay a KDE on top of the histogram?
APPLY_X_LOG_SCALE= False   # If True, set x-axis to log scale (helpful if extremely skewed)
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
    Also returns the array of resampled mean differences.
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
    return ci_lower, ci_upper, diffs

def plot_two_panel_distribution(
    data_sh, data_nsh,
    mean_sh, mean_nsh,
    diff_means,
    ci_lower, ci_upper,
    percentile_cut=99,
    title="Distribution of Prices: Superhost vs Non-Superhost",
    save_path="superhost_vs_non_superhost.png",
    diffs=None,
    figures_dir=".",
    df_approx=None,
    t_stat=None,
    t_crit_2sided=None
):
    """
    Create a 2-panel figure:
     - Left: zoomed in up to the given percentile (e.g. 99th)
     - Right: full distribution range
    Each panel: Overlaid histogram for superhost vs. non-superhost,
     plus optional KDE, vertical lines for means and CI, etc.
    """
    # Compute the cutoff
    max_val_sh = np.percentile(data_sh, percentile_cut)
    max_val_nsh= np.percentile(data_nsh, percentile_cut)
    clip_bound = max(max_val_sh, max_val_nsh)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    colors = ["#1f77b4", "#ff7f0e"]  # for superhost and non-superhost
    labels = ["Superhost", "Non-Superhost"]

    # Helper function to plot a single panel
    def plot_panel(ax, xlim=None, panel_title=""):
        # histogram for each group
        sns.histplot(data_sh, ax=ax, color=colors[0], alpha=0.4, kde=False,
                     label=f"{labels[0]}", stat='density', bins=30)
        sns.histplot(data_nsh, ax=ax, color=colors[1], alpha=0.4, kde=False,
                     label=f"{labels[1]}", stat='density', bins=30)

        # Optional KDE
        if SHOW_KDE:
            sns.kdeplot(data_sh, ax=ax, color=colors[0], label=f"{labels[0]} KDE", lw=1.5)
            sns.kdeplot(data_nsh, ax=ax, color=colors[1], label=f"{labels[1]} KDE", lw=1.5)

        # Means
        ax.axvline(mean_sh, color=colors[0], linestyle='--', label=f"Mean Superhost: {mean_sh:.2f}")
        ax.axvline(mean_nsh, color=colors[1], linestyle='--', label=f"Mean Non-SH: {mean_nsh:.2f}")

        # CI lines for difference
        ax.axvline(ci_lower, color="red", linestyle=":", label="95% CI Lower")
        ax.axvline(ci_upper, color="red", linestyle=":", label="95% CI Upper")

        ax.set_title(panel_title)
        ax.set_xlabel("Price")
        ax.set_ylabel("Density")

        if xlim is not None:
            ax.set_xlim(0, xlim)

        ax.legend()

    # Left panel: zoomed in
    plot_panel(axes[0], xlim=clip_bound, panel_title=f"Up to {percentile_cut}th percentile")

    # Right panel: full range
    # find overall max
    overall_max = max(data_sh.max(), data_nsh.max())
    plot_panel(axes[1], xlim=overall_max, panel_title="Full Range")

    fig.suptitle(title, fontsize=14)

    # Optionally set log scale
    if APPLY_X_LOG_SCALE:
        for ax in axes:
            ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Plot the distribution of the resampled mean differences
    plt.figure(figsize=(8, 4))
    sns.histplot(diffs, bins=30, kde=True, color='purple', alpha=0.6)
    plt.axvline(0, color='black', linestyle='--', label='Zero Difference')
    plt.axvline(ci_lower, color='red', linestyle=':', label='95% CI Lower')
    plt.axvline(ci_upper, color='red', linestyle=':', label='95% CI Upper')
    plt.fill_betweenx([0, plt.gca().get_ylim()[1]], ci_lower, ci_upper, color='red', alpha=0.2, label='95% CI')
    plt.title('Bootstrap Distribution of Mean Differences')
    plt.xlabel('Mean Difference (Superhost - Non-Superhost)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'bootstrap_mean_diff.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Plot T-distribution with rejection region
    plt.figure(figsize=(8, 4))
    x = np.linspace(-4, 4, 400)
    y = st.t.pdf(x, df_approx)
    plt.plot(x, y, 'b-', label='T-distribution')
    plt.fill_between(x, y, where=(x > t_crit_2sided), color='red', alpha=0.3, label='Rejection Region')
    plt.axvline(t_stat, color='green', linestyle='--', label=f'T-statistic: {t_stat:.2f}')
    plt.title('T-distribution with Rejection Region')
    plt.xlabel('T-value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 't_distribution_rejection_region.png'), dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print(f"Reading dataset from: {CSV_FILE}")
    df = pd.read_csv(CSV_FILE, low_memory=False)
    if COL_PRICE not in df.columns or COL_SUPERHOST not in df.columns:
        print("Error: Missing required columns. Exiting.")
        return

    # Extract superhost vs. non-superhost
    df_sh = df[df[COL_SUPERHOST] == SUPERHOST_VAL]
    df_nsh= df[df[COL_SUPERHOST] == NONSUPER_VAL]
    data_sh  = df_sh[COL_PRICE].dropna().values
    data_nsh = df_nsh[COL_PRICE].dropna().values

    # Remove outliers if desired
    if REMOVE_OUTLIERS:
        before_sh, before_nsh = len(data_sh), len(data_nsh)
        data_sh  = remove_3sigma_outliers(data_sh)
        data_nsh = remove_3sigma_outliers(data_nsh)
        after_sh, after_nsh = len(data_sh), len(data_nsh)
        print(f"Removed outliers (±3σ): superhost from {before_sh}->{after_sh}, non-superhost from {before_nsh}->{after_nsh}")

    # Optionally log-transform
    if USE_LOG_PRICE:
        data_sh  = np.log1p(data_sh)
        data_nsh = np.log1p(data_nsh)

    # 1) Welch's T-Test (one-sided)
    t_stat, p_val_two_sided = ttest_ind(data_sh, data_nsh, equal_var=False)
    mean_sh = np.mean(data_sh)
    mean_nsh= np.mean(data_nsh)
    diff_means = mean_sh - mean_nsh

    if diff_means > 0:
        p_val_one_sided = p_val_two_sided / 2.0
    else:
        p_val_one_sided = 1.0 - (p_val_two_sided / 2.0)

    print("\n=== Welch's T-Test (One-Sided) for Superhost vs Non-Superhost ===")
    print(f"Mean(Superhost) = {mean_sh:.2f}, Mean(Non-SH) = {mean_nsh:.2f}, diff={diff_means:.2f}")
    print(f"T-statistic = {t_stat:.4f}, p-value(one-sided)={p_val_one_sided:.5g}")

    # Decision
    if p_val_one_sided < ALPHA:
        print("=> Reject H0 => Superhosts have significantly higher mean price.")
    else:
        print("=> Fail to reject H0 => No strong evidence superhosts have higher mean price.")

    # 2) Mann–Whitney U (one-sided)
    u_stat, p_val_mw_two_sided = mannwhitneyu(data_sh, data_nsh, alternative='two-sided')
    median_sh  = np.median(data_sh)
    median_nsh = np.median(data_nsh)
    if median_sh > median_nsh:
        p_val_mw_onesided = p_val_mw_two_sided / 2.0
    else:
        p_val_mw_onesided = 1.0 - (p_val_mw_two_sided / 2.0)

    print("\n=== Mann–Whitney U Test (One-Sided) ===")
    print(f"Median(SH)={median_sh:.2f}, Median(Non-SH)={median_nsh:.2f}")
    print(f"U-stat={u_stat:.2f}, p-value(one-sided)={p_val_mw_onesided:.5g}")
    if p_val_mw_onesided < ALPHA:
        print("=> Reject H0 => Superhosts have significantly greater distribution stochastically.")
    else:
        print("=> Fail to reject H0 => No strong evidence superhosts have higher distribution.")

    # 3) Bootstrap difference in means
    boot_low, boot_high, diffs = bootstrap_mean_diff(data_sh, data_nsh, n_boot=2000)
    print("\n=== Bootstrapped 95% CI for difference in means (superhost - non-superhost) ===")
    print(f"95% CI: [{boot_low:.2f}, {boot_high:.2f}]")

    # Additional info
    # degrees of freedom for Welch approx:
    var1 = np.var(data_sh, ddof=1)
    var2 = np.var(data_nsh, ddof=1)
    n1, n2 = len(data_sh), len(data_nsh)
    df_approx = (var1/n1 + var2/n2)**2 / \
                ((var1**2)/(n1**2*(n1-1)) + (var2**2)/(n2**2*(n2-1)))
    t_crit_2sided = st.t.ppf(1 - ALPHA/2, df_approx)

    print("\nTest concluded successfully.")

    # 4) Plot with two subplots (zoom + full)
    figures_dir = os.path.join(current_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    save_fig_path = os.path.join(figures_dir, 'superhost_vs_non_superhost_enhanced.png')

    plot_two_panel_distribution(
        data_sh, data_nsh,
        mean_sh, mean_nsh,
        diff_means,
        ci_lower=boot_low, ci_upper=boot_high,
        percentile_cut=99,
        title="Distribution of Prices: Superhost vs Non-Superhost (Enhanced)",
        save_path=save_fig_path,
        diffs=diffs,
        figures_dir=figures_dir,
        df_approx=df_approx,
        t_stat=t_stat,
        t_crit_2sided=t_crit_2sided
    )

if __name__ == "__main__":
    main()
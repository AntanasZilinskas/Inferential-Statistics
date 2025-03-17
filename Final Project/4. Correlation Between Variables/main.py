"""
task4_scatter_only.py

Task 4: Correlation Between Variables (No Distribution Plots)
-------------------------------------------------------------
We examine how 'price' correlates with:
   1) 'bedrooms'
   2) 'bathrooms'
   3) 'distance_km'

Steps:
1) Remove ±3σ outliers (pairwise) for each correlation.
2) Compute Pearson's r on the outlier-trimmed subset.
3) Create a scatter plot (with regression line) for each pair.

Outputs:
  - Figures in 'figures' subfolder
  - Correlation stats in the console
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# --------------------- SETTINGS ---------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR  = os.path.dirname(CURRENT_DIR)
CSV_FILE    = os.path.join(PARENT_DIR, "listings_with_goodness.csv")

COL_PRICE     = "price"
COL_BEDROOMS  = "bedrooms"
COL_BATHROOMS = "bathrooms"
COL_DISTANCE  = "distance_km"

# Choose your pairs with price:
PAIRS = [
    (COL_PRICE, COL_BEDROOMS, "Price vs Bedrooms"),
    (COL_PRICE, COL_BATHROOMS, "Price vs Bathrooms"),
    (COL_PRICE, COL_DISTANCE,  "Price vs Distance (km)"),
]

OUTPUT_DIR = os.path.join(CURRENT_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ----------------------------------------------------


def remove_3sigma_outliers_pairwise(df, x_col, y_col):
    """
    For each pair (x_col, y_col):
      1) Drop rows missing either x_col or y_col
      2) Remove rows where x_col is beyond ±3σ of its mean
      3) Remove rows where y_col is beyond ±3σ of its mean
    Return the filtered subset.
    """
    df_pair = df[[x_col, y_col]].dropna().copy()
    
    # For x_col
    x_mean = df_pair[x_col].mean()
    x_std  = df_pair[x_col].std()
    x_low, x_high = x_mean - 3*x_std, x_mean + 3*x_std

    # For y_col
    y_mean = df_pair[y_col].mean()
    y_std  = df_pair[y_col].std()
    y_low, y_high = y_mean - 3*y_std, y_mean + 3*y_std

    # Keep only rows within ±3σ for both x_col and y_col
    mask = (
        (df_pair[x_col] >= x_low) & (df_pair[x_col] <= x_high) &
        (df_pair[y_col] >= y_low) & (df_pair[y_col] <= y_high)
    )
    return df_pair[mask]


def main():
    print(f"Reading data from: {CSV_FILE}")
    df = pd.read_csv(CSV_FILE, low_memory=True)

    print("\n=== Pearson Correlations (±3σ outlier removal per pair) ===\n")
    for (col_price, col_other, title_str) in PAIRS:
        # 1) Filter out outliers pairwise
        df_filtered = remove_3sigma_outliers_pairwise(df, col_price, col_other)
        n_filtered  = len(df_filtered)
        if n_filtered < 2:
            print(f"Insufficient data for {col_price} vs {col_other} after 3σ filtering.")
            continue
        
        # 2) Compute Pearson correlation
        r_val, p_val = pearsonr(df_filtered[col_price], df_filtered[col_other])
        print(f"{col_price} vs {col_other}: r={r_val:.3f}, p={p_val:.4g}, n={n_filtered}")

        # 3) Scatter plot with regression line
        out_fname = f"scatter_{col_price}_vs_{col_other}.png"
        out_path  = os.path.join(OUTPUT_DIR, out_fname)
        plot_scatter(df_filtered, col_price, col_other, r_val, p_val, title_str, out_path)


def plot_scatter(df, col_y, col_x, r_val, p_val, plot_title, outpath):
    """
    Create a scatter plot (col_x vs col_y) with a regression line.
    Label the plot with Pearson r and p-value, and note outlier removal.
    """
    sns.set_style("whitegrid")
    plt.figure(figsize=(6,4))
    sns.regplot(x=col_x, y=col_y, data=df,
                scatter_kws={'alpha': 0.3},
                line_kws={'color': 'red'})
    plt.title(f"{plot_title}\n(±3σ filtered) r={r_val:.3f}, p={p_val:.4g}")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Saved plot: {outpath}")


if __name__ == "__main__":
    main()
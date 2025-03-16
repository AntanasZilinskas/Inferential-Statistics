"""

Task 4: Correlation Between Variables
-------------------------------------
We examine how 'price' correlates with:
  1) 'score_goodness'
  2) 'distance_km'
  3) 'number_of_reviews'

We compute Pearson's r and visualize via scatter plots.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# ============ SETTINGS ============
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
CSV_FILE = os.path.join(parent_dir, "listings_with_goodness.csv")

COL_PRICE         = "price"
COL_GOODNESS      = "score_goodness"
COL_DISTANCE      = "distance_km"
COL_NUM_REVIEWS   = "number_of_reviews"

SCATTER_OUTDIR    = os.path.join(current_dir, "figures_correlations")
os.makedirs(SCATTER_OUTDIR, exist_ok=True)
# ==================================

def main():
    # 1) Read data
    df = pd.read_csv(CSV_FILE, low_memory=False)

    # 2) Drop rows that are missing relevant columns
    cols_of_interest = [COL_PRICE, COL_GOODNESS, COL_DISTANCE, COL_NUM_REVIEWS]
    df_sub = df[cols_of_interest].dropna()

    # 3) Compute correlation (Pearson's r) for each pair with 'price'
    #    We'll store results in a dictionary for clarity
    corr_results = {}

    # a) price vs score_goodness
    r_goodness, p_goodness = pearsonr(df_sub[COL_PRICE], df_sub[COL_GOODNESS])
    corr_results["price_score_goodness"] = (r_goodness, p_goodness)

    # b) price vs distance_km
    r_distance, p_distance = pearsonr(df_sub[COL_PRICE], df_sub[COL_DISTANCE])
    corr_results["price_distance"] = (r_distance, p_distance)

    # c) price vs number_of_reviews
    r_reviews, p_reviews = pearsonr(df_sub[COL_PRICE], df_sub[COL_NUM_REVIEWS])
    corr_results["price_num_reviews"] = (r_reviews, p_reviews)

    # Print the correlation results
    print("\n=== CORRELATIONS WITH 'price' (Pearson) ===")
    print(f"1) price vs. score_goodness: r={r_goodness:.3f}, p={p_goodness:.4f}")
    print(f"2) price vs. distance_km   : r={r_distance:.3f}, p={p_distance:.4f}")
    print(f"3) price vs. number_of_reviews: r={r_reviews:.3f}, p={p_reviews:.4f}")

    # 4) Generate scatter plots
    plot_scatter_with_corr(df_sub, COL_PRICE, COL_GOODNESS,
                           corr_results["price_score_goodness"],
                           "Price vs. Score Goodness",
                           os.path.join(SCATTER_OUTDIR, "scatter_price_vs_score_goodness.png"))

    plot_scatter_with_corr(df_sub, COL_PRICE, COL_DISTANCE,
                           corr_results["price_distance"],
                           "Price vs. Distance (km)",
                           os.path.join(SCATTER_OUTDIR, "scatter_price_vs_distance_km.png"))

    plot_scatter_with_corr(df_sub, COL_PRICE, COL_NUM_REVIEWS,
                           corr_results["price_num_reviews"],
                           "Price vs. Number of Reviews",
                           os.path.join(SCATTER_OUTDIR, "scatter_price_vs_number_of_reviews.png"))

def plot_scatter_with_corr(df, x_col, y_col, corr_tuple, title, outpath):
    """
    Create a scatter plot with a regression line, labeling the Pearson correlation in the title.
    corr_tuple = (r_value, p_value)
    """
    r_val, p_val = corr_tuple
    sns.set_style("whitegrid")
    plt.figure(figsize=(6,4))
    sns.regplot(x=x_col, y=y_col, data=df,
                scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    plt.title(f"{title}\nPearson r={r_val:.2f}, p={p_val:.4f}")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
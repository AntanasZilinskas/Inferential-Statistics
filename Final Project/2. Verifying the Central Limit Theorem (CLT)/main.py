"""
main.py
Demonstrates the Central Limit Theorem (CLT) in two ways:
  1) Overlaid Independent Batches: Distros of sample means for n=10, n=50, n=200
  2) Running Averages: A single "stream" approach repeated multiple times, illustrating convergence

We also remove outliers (beyond +/- 3σ from the raw data) to avoid weird results. 
No command-line arguments are required.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import os
# --------------- DEFINED SETTINGS ---------------

current_dir = os.path.dirname(os.path.abspath(__file__))
CSV_FILE   = os.path.join(os.path.dirname(current_dir), "listings_with_goodness.csv")
COLUMN     = "distance_km"                # Numeric column to verify the CLT on
CLIP_3SIGMA = True                      # Whether to remove outliers beyond +/- 3σ
# Approach 1: sample sizes + number of means
SAMPLE_SIZES = [10, 50, 200]
N_EXPERIMENTS = 2000
# Approach 2: total draws for the "stream" + number of runs
TOTAL_DRAWS = 5000
NUM_RUNS    = 5

# Create figures directory if it doesn't exist
FIGURES_DIR = os.path.join(current_dir, "figures")
if not os.path.exists(FIGURES_DIR):
    os.makedirs(FIGURES_DIR)
    print(f"Created figures directory: {FIGURES_DIR}")
# -----------------------------------------------------

def clean_outliers_3sigma(arr):
    """
    Removes elements lying outside mean +/- 3*std. 
    Returns the cleaned array.
    """
    mean_val = np.mean(arr)
    std_val  = np.std(arr)
    low_cut  = mean_val - 3.0 * std_val
    high_cut = mean_val + 3.0 * std_val
    return arr[(arr >= low_cut) & (arr <= high_cut)]

def overlaid_independent_batches_clt(data_array, sample_sizes, n_experiments=2000):
    """
    Approach 1: For each n in sample_sizes, generate n_experiments sample means
    from data_array, then overlay them in a single histogram with different colors,
    plus theoretical normal overlays.
    """
    pop_mean = data_array.mean()
    pop_var  = data_array.var(ddof=1)

    plt.figure(figsize=(9,6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # 3 distinct colors
    bins   = 40

    min_val, max_val = float("inf"), float("-inf")
    for i, n in enumerate(sample_sizes):
        means = []
        for _ in range(n_experiments):
            sample = np.random.choice(data_array, size=n, replace=True)
            means.append(sample.mean())
        means = np.array(means)

        # Expand the global range for x-limits
        min_val = min(min_val, means.min())
        max_val = max(max_val, means.max())

        # Plot histogram (stat='density' to show PDF)
        sns.histplot(means, bins=bins, color=colors[i], alpha=0.3, 
                     stat='density', label=f"n={n}", kde=False)

        # Theoretical normal
        theory_var   = pop_var / n
        theory_sigma = np.sqrt(theory_var)
        x_vals = np.linspace(means.min(), means.max(), 200)
        pdf_theoretical = norm.pdf(x_vals, loc=pop_mean, scale=theory_sigma)
        plt.plot(x_vals, pdf_theoretical, '--', 
                 color=colors[i], linewidth=2)

    # Add vertical line for the true mean
    plt.axvline(x=pop_mean, color='red', linestyle='-', linewidth=2,
                label=f'True Mean: {pop_mean:.2f}')

    plt.title(f"Overlaid Distros of Sample Means\nn={sample_sizes} (each: {n_experiments} means)")
    plt.xlabel("Sample Mean")
    plt.ylabel("Density")
    plt.xlim([min_val, max_val])
    plt.legend()
    plt.tight_layout()
    
    # Save the figure
    fig_path = os.path.join(FIGURES_DIR, f"{COLUMN}_overlaid_batches_clt.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {fig_path}")
    
    plt.show()

def running_averages_clt(data_array, total_draws=5000, num_runs=5):
    """
    Approach 2: Show how a single stream's running average converges over time.
    Repeated num_runs times.
    """
    pop_mean = data_array.mean()

    plt.figure(figsize=(10, 5))

    for run_idx in range(num_runs):
        draws = np.random.choice(data_array, size=total_draws, replace=True)
        cumsums = np.cumsum(draws)
        running_avg = cumsums / (np.arange(1, total_draws + 1))
        plt.plot(running_avg, alpha=0.7, label=f"Run {run_idx+1}")

    plt.axhline(y=pop_mean, color='red', linestyle='--', linewidth=2, 
                label=f"True mean ~ {pop_mean:.2f}")
    plt.title(f"{num_runs} Runs of Running Averages (total_draws={total_draws})")
    plt.xlabel("Sequential Draws")
    plt.ylabel("Running Average")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    
    # Save the figure
    fig_path = os.path.join(FIGURES_DIR, f"{COLUMN}_running_averages_clt.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {fig_path}")
    
    plt.show()

def main():
    # 1) Load data
    print(f"Loading '{CSV_FILE}' ...")
    df = pd.read_csv(CSV_FILE, low_memory=False)
    if COLUMN not in df.columns:
        print(f"Error: column '{COLUMN}' not found in the DataFrame.")
        print(f"Available columns: {', '.join(df.columns)}")
        return
    
    data_array = df[COLUMN].dropna().values
    print(f"Raw data size for '{COLUMN}': {len(data_array)}")
    if len(data_array) == 0:
        print("No valid data after dropping NaNs. Exiting.")
        return

    # 2) Optionally remove +/- 3σ outliers
    if CLIP_3SIGMA:
        before_size = len(data_array)
        data_array = clean_outliers_3sigma(data_array)
        after_size = len(data_array)
        print(f"Removed {before_size - after_size} outliers beyond ±3σ. Remaining data size: {after_size}")

    pop_mean = data_array.mean()
    pop_std  = data_array.std(ddof=1)
    print(f"Cleaned data stats -> Mean: {pop_mean:.4f}, Std: {pop_std:.4f}")

    # 3) Approach 1: Overlaid hist of sample means for n=10,50,200
    print("\nRunning Overlaid Independent Batches CLT demonstration...")
    overlaid_independent_batches_clt(
        data_array,
        sample_sizes=SAMPLE_SIZES,
        n_experiments=N_EXPERIMENTS
    )

    # 4) Approach 2: Running Averages
    print("\nRunning Running Averages CLT demonstration...")
    running_averages_clt(
        data_array,
        total_draws=TOTAL_DRAWS,
        num_runs=NUM_RUNS
    )

    print("\nAll CLT demonstrations completed successfully!")

if __name__ == "__main__":
    main()
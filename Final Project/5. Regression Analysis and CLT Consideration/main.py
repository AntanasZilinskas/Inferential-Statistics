"""
main.py

Task 5: Regression + CLT, with "Above and Beyond" Enhancements
and ±3σ Outlier Removal
--------------------------------------------------------------
We predict 'price' from ['bedrooms','bathrooms','distance_km'],
and remove ±3σ outliers in these columns before fitting.

Steps:
1) Remove ±3σ outliers in [price, bedrooms, bathrooms, distance_km].
2) Fit full OLS, show summary.
3) Residual Diagnostics (resid vs fitted, Q-Q plot).
4) 5-Fold Cross-Validation (R^2, MSE).
5) Bootstrap for coefficient distributions (compare to analytic CIs).
6) Demonstrate how bigger sample fraction => smaller coef variance
   (a direct CLT illustration).

Generates multiple figures in 'figures_task5_advanced/'.
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

# Cross-validation
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

# ----------------- SETTINGS -----------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR  = os.path.dirname(CURRENT_DIR)
CSV_FILE    = os.path.join(PARENT_DIR, "listings_with_goodness.csv")

RESPONSE_COL = "price"
PREDICTORS   = ["bedrooms", "bathrooms", "distance_km"]
OUTLIER_COLS = [RESPONSE_COL] + PREDICTORS  # columns to apply ±3σ filter

FIGURES_DIR  = os.path.join(CURRENT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

N_BOOTSTRAP     = 500     # for the bootstrap
SAMPLE_FRACTIONS= [0.2,0.4,0.6,0.8] # for sub-sampling demonstration
# -------------------------------------------


def main():
    print(f"Reading data from: {CSV_FILE}")
    df = pd.read_csv(CSV_FILE, low_memory=True)

    # 1) Drop missing values in the needed columns
    df_sub = df[OUTLIER_COLS].dropna()
    print(f"Data size after dropping NA in {OUTLIER_COLS}: {len(df_sub)}")

    # 2) Remove ±3σ outliers for these columns
    df_sub = remove_outliers_3sigma_global(df_sub, OUTLIER_COLS)
    print(f"Data size after ±3σ outlier removal: {len(df_sub)}")

    # 3) Fit OLS on the entire dataset
    X_full = add_constant(df_sub[PREDICTORS])
    y_full = df_sub[RESPONSE_COL]
    model_full = OLS(y_full, X_full).fit()
    print("\n=== FULL MODEL SUMMARY ===")
    print(model_full.summary())

    # 4) Residual Diagnostics
    plot_residual_diagnostics(model_full, X_full, y_full, FIGURES_DIR)

    # 5) 5-Fold Cross Validation
    cv_res = run_kfold_cv(df_sub, RESPONSE_COL, PREDICTORS, n_splits=5)
    mean_r2  = np.mean(cv_res['r2'])
    std_r2   = np.std(cv_res['r2'])
    mean_mse = np.mean(cv_res['mse'])
    std_mse  = np.std(cv_res['mse'])
    print("\n=== 5-Fold Cross-Validation ===")
    print(f"Mean R^2 = {mean_r2:.3f} ± {std_r2:.3f}")
    print(f"Mean MSE = {mean_mse:.3f} ± {std_mse:.3f}")

    # 6) Bootstrap: collect coefficient distributions
    df_coefs_boot, dict_cis = bootstrap_coefficients(df_sub, n_boot=N_BOOTSTRAP)
    conf_analytic = model_full.conf_int(alpha=0.05)

    print("\n=== Bootstrap vs. Analytic Confidence Intervals (95%) ===")
    for param in model_full.params.index:
        b_low, b_high = dict_cis[param]
        a_low, a_high = conf_analytic.loc[param]
        print(f"Param: {param}")
        print(f"  Bootstrap   CI = [{b_low:.2f}, {b_high:.2f}]")
        print(f"  Analytic    CI = [{a_low:.2f}, {a_high:.2f}]\n")

    plot_bootstrap_coefs(df_coefs_boot, FIGURES_DIR)

    # 7) Show how coefficient std dev shrinks with bigger sample fraction
    df_var = vary_sample_size_coefs(df_sub, fractions=SAMPLE_FRACTIONS, n_rep=100)
    plot_coef_std_vs_sample_fraction(df_var, FIGURES_DIR)

    print("\nAll advanced steps (with ±3σ outlier removal) completed.")


def remove_outliers_3sigma_global(df, cols):
    """
    Removes rows where *any* column in cols is beyond ±3σ for that column.
    """
    df_clean = df.copy()
    for c in cols:
        mean_c = df_clean[c].mean()
        std_c  = df_clean[c].std()
        lower  = mean_c - 3.0*std_c
        upper  = mean_c + 3.0*std_c
        df_clean = df_clean[(df_clean[c]>=lower) & (df_clean[c]<=upper)]
    return df_clean


def plot_residual_diagnostics(model, X, y, outdir):
    """
    1) Residual vs Fitted
    2) Q-Q plot
    """
    fitted_vals = model.fittedvalues
    residuals   = model.resid

    # Plot 1: Residual vs Fitted
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=fitted_vals, y=residuals, alpha=0.3)
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Residuals vs. Fitted Values")
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.tight_layout()
    out1 = os.path.join(outdir, "resid_vs_fitted.png")
    plt.savefig(out1, dpi=300)
    plt.close()

    # Plot 2: Q-Q plot
    import statsmodels.api as sm
    fig = sm.qqplot(residuals, line='45', fit=True)
    plt.title("Q-Q Plot of Residuals")
    out2 = os.path.join(outdir, "qq_residuals.png")
    plt.savefig(out2, dpi=300)
    plt.close()

    print(f"Saved residual diagnostics:\n  {out1}\n  {out2}")


def run_kfold_cv(df, resp, preds, n_splits=5):
    """
    KFold CV with statsmodels OLS. Return R^2, MSE for each fold.
    """
    X_all = df[preds].values
    y_all = df[resp].values

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    r2_list, mse_list = [], []

    for train_idx, test_idx in kf.split(X_all):
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        X_train_c = add_constant(X_train)
        model_fold = OLS(y_train, X_train_c).fit()

        X_test_c = add_constant(X_test)
        y_pred = model_fold.predict(X_test_c)

        r2_list.append(r2_score(y_test, y_pred))
        mse_list.append(mean_squared_error(y_test, y_pred))

    return {"r2": r2_list, "mse": mse_list}


def bootstrap_coefficients(df, n_boot=500, alpha=0.05):
    """
    Repeatedly sample w/ replacement -> fit OLS -> store coefs.
    Return (df_coefs, dict_cis).
    """
    np.random.seed(42)
    coefs_records = []
    n = len(df)

    for _ in range(n_boot):
        df_samp = df.sample(n=n, replace=True)
        X_s = add_constant(df_samp[PREDICTORS])
        y_s = df_samp[RESPONSE_COL]
        model_s = OLS(y_s, X_s).fit()
        coefs_records.append(model_s.params)

    df_coefs = pd.DataFrame(coefs_records)
    dict_cis = {}
    for col in df_coefs.columns:
        lower = np.percentile(df_coefs[col], 100*alpha/2)
        upper = np.percentile(df_coefs[col], 100*(1-alpha/2))
        dict_cis[col] = (lower, upper)

    return df_coefs, dict_cis


def plot_bootstrap_coefs(df_coefs, outdir):
    """
    Plot distributions of bootstrap coefficient estimates.
    """
    sns.set_style("whitegrid")
    cols = df_coefs.columns
    num_params = len(cols)
    
    # Change from 1 row of 4 to 2 rows of 2
    nrows = 2
    ncols = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 8), sharey=False)
    axes = axes.flatten()  # Flatten to easily iterate

    for i, col in enumerate(cols):
        if i < len(axes):  # Ensure we don't exceed the number of subplots
            sns.histplot(df_coefs[col], kde=True, ax=axes[i], color='purple', alpha=0.4)
            mean_val = df_coefs[col].mean()
            std_val  = df_coefs[col].std()
            axes[i].axvline(mean_val, color='red', linestyle='--', label=f"Mean={mean_val:.2f}")
            axes[i].axvspan(mean_val - std_val, mean_val + std_val, color='red', alpha=0.1)
            axes[i].set_title(f"{col}\nMean={mean_val:.2f}, Std={std_val:.2f}")
            axes[i].legend()
    
    # Hide any unused subplots
    for i in range(num_params, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    outpath = os.path.join(outdir, "bootstrap_coefs.png")
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Saved bootstrap coefficient distributions: {outpath}")


def vary_sample_size_coefs(df, fractions=[0.2,0.4,0.6,0.8], n_rep=50):
    """
    For each fraction f, do n_rep sub-samples (no replacement),
    fit OLS, record coefs, compute std dev of each param.
    Return DataFrame with index=fraction, columns=coefs' std dev.
    """
    np.random.seed(42)
    N = len(df)
    results = {}

    for frac in fractions:
        coefs_records = []
        size = int(frac*N)
        for _ in range(n_rep):
            df_samp = df.sample(n=size, replace=False)
            X_s = add_constant(df_samp[PREDICTORS])
            y_s = df_samp[RESPONSE_COL]
            model_s = OLS(y_s, X_s).fit()
            coefs_records.append(model_s.params)

        df_coefs = pd.DataFrame(coefs_records)
        std_series = df_coefs.std()
        results[frac] = std_series

    df_var = pd.DataFrame(results).T  # fraction as index, param names as columns
    return df_var


def plot_coef_std_vs_sample_fraction(df_var, outdir):
    """
    df_var: index=fraction, columns=[const, bedrooms, bathrooms, distance_km].
    We plot each param's std dev vs. fraction.
    """
    sns.set_style("whitegrid")
    params = df_var.columns
    n_param = len(params)
    
    # Change from 1 row of 4 to 2 rows of 2
    nrows = 2
    ncols = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 8), sharey=False)
    axes = axes.flatten()  # Flatten to easily iterate

    for i, param in enumerate(params):
        if i < len(axes):  # Ensure we don't exceed the number of subplots
            axes[i].plot(df_var.index, df_var[param], marker='o', color='blue')
            axes[i].set_title(f"Std Dev of '{param}' vs. Sample Fraction")
            axes[i].set_xlabel("Sample Fraction")
            axes[i].set_ylabel("Std Dev of Coeff")
            axes[i].grid(True)
    
    # Hide any unused subplots
    for i in range(n_param, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    outpath = os.path.join(outdir, "coef_stdev_vs_samplefraction.png")
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Saved coefficient std dev vs sample fraction plot: {outpath}")


if __name__ == "__main__":
    main()
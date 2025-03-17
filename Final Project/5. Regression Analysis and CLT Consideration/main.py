"""
advanced_regression_clt.py

Task 5: Regression Analysis & CLT Demonstration
with Extra Enhancements:
-----------------------------------------------
1) Multiple Linear Regression (response='price'),
   Predictors: ['bedrooms', 'bathrooms', 'distance_km'].
2) Model Fit Summary (R^2, Coefficients, Interpretation).
3) Residual Diagnostics (resid vs fitted, Q-Q plot).
4) 5-Fold Cross-Validation (MSE, R^2).
5) Bootstrap Coefficients & Compare Empirical CIs to Analytic OLS CIs.
6) Demonstrate how bigger sample fractions reduce coefficient variance
   (overlay how increasing the "batch size" of random sub-samples
    shrinks the standard deviation of estimated coefficients).

The last part (#6) helps show the CLT:
As sample size grows, the standard error of estimates goes down.
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

# For cross-validation
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR  = os.path.dirname(CURRENT_DIR)
CSV_FILE    = os.path.join(PARENT_DIR, "listings_with_goodness.csv")

RESPONSE_COL = "price"
PREDICTORS   = ["bedrooms", "bathrooms", "distance_km"]

FIGURES_DIR  = os.path.join(CURRENT_DIR, "figures_task5_advanced")
os.makedirs(FIGURES_DIR, exist_ok=True)

N_BOOTSTRAP  = 500  # for the bootstrap (can set 1000 if desired)
SAMPLE_FRACTIONS = [0.2, 0.4, 0.6, 0.8]  # to show how coefficient var shrinks with bigger "batch size"


def main():
    # 1) Load data
    df = pd.read_csv(CSV_FILE, low_memory=True)
    needed = [RESPONSE_COL] + PREDICTORS
    df_sub = df[needed].dropna()
    print(f"Data size after dropping NA in {needed}: {len(df_sub)} rows")

    # 2) Full OLS model
    X_full = add_constant(df_sub[PREDICTORS])
    y_full = df_sub[RESPONSE_COL]
    model_full = OLS(y_full, X_full).fit()
    print("\n=== FULL MODEL SUMMARY ===")
    print(model_full.summary())

    # 3) Residual Diagnostics
    plot_residual_diagnostics(model_full, X_full, y_full, FIGURES_DIR)

    # 4) 5-Fold Cross-Validation
    cv_res = run_kfold_cv(df_sub, RESPONSE_COL, PREDICTORS, n_splits=5)
    print("\n=== 5-Fold Cross-Validation ===")
    print(f"Mean R^2 = {np.mean(cv_res['r2']):.3f} ± {np.std(cv_res['r2']):.3f}")
    print(f"Mean MSE = {np.mean(cv_res['mse']):.3f} ± {np.std(cv_res['mse']):.3f}")

    # 5) Bootstrap: coefficient distributions + compare to OLS intervals
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

    # 6) Demonstrate that bigger sample fraction => smaller coefficient variance
    # We'll do repeated sub-sampling for each fraction in SAMPLE_FRACTIONS,
    # store std dev of each coefficient, then plot them.
    df_var = vary_sample_size_coefs(df_sub, fractions=SAMPLE_FRACTIONS, n_rep=100)
    # df_var: rows = fraction, columns = each param -> standard deviation
    plot_coef_std_vs_sample_fraction(df_var, FIGURES_DIR)

    print("\nDone. All advanced analysis steps completed.")


def plot_residual_diagnostics(model, X, y, outdir):
    """
    1) Residuals vs. Fitted
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
    5-fold CV using statsmodels OLS. Return R^2, MSE for each fold.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    X_all = df[preds].values
    y_all = df[resp].values

    r2_list, mse_list = [], []
    for train_idx, test_idx in kf.split(X_all):
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        X_train_c = add_constant(X_train)
        model_cv = OLS(y_train, X_train_c).fit()

        X_test_c = add_constant(X_test)
        y_pred = model_cv.predict(X_test_c)

        r2_list.append(r2_score(y_test, y_pred))
        mse_list.append(mean_squared_error(y_test, y_pred))

    return {"r2": r2_list, "mse": mse_list}


def bootstrap_coefficients(df, n_boot=500, alpha=0.05):
    """
    For each bootstrap:
      1) sample df w/ replacement
      2) fit OLS on (price ~ bedrooms + bathrooms + distance_km)
      3) store param estimates
    Return:
      df_coefs (DataFrame: n_boot x [const, bedrooms, bathrooms, distance_km])
      dict_cis (param -> (lowCI, highCI))
    """
    np.random.seed(42)
    n = len(df)
    coefs_records = []

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
        upper = np.percentile(df_coefs[col], 100*(1 - alpha/2))
        dict_cis[col] = (lower, upper)

    return df_coefs, dict_cis


def plot_bootstrap_coefs(df_coefs, outdir):
    """
    Plot distribution of each coefficient from bootstrap.
    """
    sns.set_style("whitegrid")
    cols = df_coefs.columns
    num_params = len(cols)
    fig, axes = plt.subplots(1, num_params, figsize=(4*num_params,4), sharey=False)

    if num_params == 1:
        axes = [axes]

    for i, col in enumerate(cols):
        sns.histplot(df_coefs[col], kde=True, ax=axes[i], color='purple', alpha=0.4)
        mean_val = df_coefs[col].mean()
        std_val  = df_coefs[col].std()
        axes[i].axvline(mean_val, color='red', linestyle='--', label=f"Mean={mean_val:.2f}")
        axes[i].axvspan(mean_val - std_val, mean_val + std_val, color='red', alpha=0.1)
        axes[i].set_title(f"{col}\nMean={mean_val:.2f}, STD={std_val:.2f}")
        axes[i].legend()

    plt.tight_layout()
    outpath = os.path.join(outdir, "bootstrap_coefs.png")
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Saved bootstrap coefficient distributions: {outpath}")


def vary_sample_size_coefs(df, fractions=[0.2,0.4,0.6,0.8], n_rep=50):
    """
    Show how coefficient std dev shrinks with bigger sample fraction.
    For each fraction f in fractions:
      - sample f*N rows from df (no replacement)
      - fit OLS
      - repeat n_rep times
      - compute std dev of the coefficient estimates

    Return a DataFrame with index=fraction and columns=[const, bedrooms, bathrooms, distance_km]
    containing the std dev of each coefficient for that fraction.
    """
    results = {}
    np.random.seed(42)
    N = len(df)

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

    df_var = pd.DataFrame(results).T  # index=fraction, columns=coef names
    return df_var


def plot_coef_std_vs_sample_fraction(df_var, outdir):
    """
    df_var: index = fraction, columns = [const, bedrooms, bathrooms, distance_km]
    We plot each column's std dev vs sample fraction.
    This shows how coefficient uncertainty decreases with bigger sample fraction.
    """
    # We'll do separate subplots for each coefficient
    sns.set_style("whitegrid")
    params = df_var.columns
    num_params = len(params)
    fig, axes = plt.subplots(1, num_params, figsize=(4*num_params,4), sharey=False)

    if num_params==1:
        axes=[axes]

    for i, param in enumerate(params):
        axes[i].plot(df_var.index, df_var[param], marker='o', color='blue')
        axes[i].set_title(f"Std Dev of '{param}' vs. Sample Fraction")
        axes[i].set_xlabel("Sample Fraction")
        axes[i].set_ylabel("Std Dev of Coeff")
        axes[i].grid(True)

    plt.tight_layout()
    outpath = os.path.join(outdir, "coef_stdev_vs_samplefraction.png")
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Saved coefficient std dev vs sample fraction plot: {outpath}")


if __name__ == "__main__":
    main()
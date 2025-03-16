import os
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
CSV_FILE = os.path.join(PARENT_DIR, "listings_with_goodness.csv")

TARGET_COL = "price"

def main():
    # Load data
    df = pd.read_csv(CSV_FILE, low_memory=True)
    
    # Select numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Drop rows missing 'price'
    df_numeric = df[numeric_cols].dropna(subset=[TARGET_COL])
    
    # Compute correlation matrix among numeric columns
    corr_matrix = df_numeric.corr()
    
    # Extract correlation with 'price' and sort by absolute value (descending)
    price_corr = corr_matrix[TARGET_COL].drop(labels=[TARGET_COL])  # exclude price itself
    price_corr_sorted = price_corr.reindex(price_corr.abs().sort_values(ascending=False).index)
    
    print("\n=== CORRELATION WITH 'PRICE' (Pearson) ===")
    print("Column                       Corr     |Corr|")
    print("------------------------------------------------")
    for col in price_corr_sorted.index:
        val = price_corr_sorted[col]
        print(f"{col:<28} {val: .4f}   {abs(val): .4f}")

    print("\nColumns at the top of this list have the strongest linear correlation (±) with price.")
    print("If all values are very close to zero, it indicates no strong linear relationships in your numeric columns.")

if __name__ == "__main__":
    main()
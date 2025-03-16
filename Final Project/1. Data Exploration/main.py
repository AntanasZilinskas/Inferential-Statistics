import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from math import radians, sin, cos, asin, sqrt
import os

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Computes approximate distance (in km) between two lat/lon points
    using the Haversine formula.
    """
    rlat1, rlon1 = np.radians(lat1), np.radians(lon1)
    rlat2, rlon2 = np.radians(lat2), np.radians(lon2)
    dlon = rlon2 - rlon1
    dlat = rlat2 - rlat1
    a = (np.sin(dlat/2)**2
         + np.cos(rlat1)*np.cos(rlat2)*np.sin(dlon/2)**2)
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c  # Earth radius ~6371 km

def create_distance_price(df,
                          center_lat=51.509865,
                          center_lon=-0.118092):
    """
    1) distance_km : lat/lon -> Haversine distance from city center (London coords).
    2) price       : parse from $..., untransformed.

    Returns df with new/updated columns: distance_km, price.
    (We assume 'score_goodness' already exists in df.)
    """

    # distance_km
    if 'latitude' in df.columns and 'longitude' in df.columns:
        def compute_dist(row):
            return haversine_distance(row['latitude'], row['longitude'],
                                      center_lat, center_lon)
        df['distance_km'] = df.apply(compute_dist, axis=1)
    else:
        print("Warning: 'latitude'/'longitude' missing, cannot create distance_km.")

    # price (raw)
    if 'price' in df.columns:
        df['price'] = (
            df['price']
            .astype(str)
            .str.replace('$','', regex=False)
            .str.replace(',','', regex=False)
            .astype(float, errors='ignore')
        )
    else:
        print("Warning: 'price' not in DataFrame.")

    return df

def plot_three_features(df, columns, clip_percentile=0.99, save_dir="figures"):
    """
    - Plots distributions of numeric features (histogram + KDE and boxplot).
    - Computes mean, variance, and other summary statistics.
    - Identifies outliers that lie outside three standard deviations from the mean.
    - Adds visual indicators for mean, median, std, and 3σ outlier boundaries.
    - Saves each figure as a PNG in the specified 'save_dir'.
    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for col in columns:
        if col not in df.columns:
            print(f"\nColumn '{col}' not found in DataFrame. Skipping.")
            continue

        data = df[col].dropna()
        if data.empty:
            print(f"\nColumn '{col}' has all NaN or is empty. Skipping.")
            continue

        print(f"\n=== Exploring '{col.upper()}' ===")

        # Basic stats on the data
        mean_val = data.mean()
        var_val = data.var()
        std_val = data.std()
        median_val = data.median()
        min_val = data.min()
        max_val = data.max()

        print(f"Count: {len(data)}")
        print(f"Mean:  {mean_val:.4f}")
        print(f"Median:{median_val:.4f}")
        print(f"Var:   {var_val:.4f}")
        print(f"Std:   {std_val:.4f}")
        print(f"Min:   {min_val:.4f}")
        print(f"Max:   {max_val:.4f}")

        # Outlier detection using 3σ method (as specified in the assignment)
        cutoff_low = mean_val - 3 * std_val
        cutoff_high = mean_val + 3 * std_val
        outliers = data[(data < cutoff_low) | (data > cutoff_high)]
        print(f"Outliers (±3σ): {len(outliers)} values outside [{cutoff_low:.4f}, {cutoff_high:.4f}]")
        
        # For visualization purposes, clip extreme values to see the distribution better
        upper_bound = data.quantile(clip_percentile)
        data_vis = data.clip(upper=upper_bound)
        
        # Calculate skewness on the clipped data
        skew_val = data_vis.skew()
        print(f"Skewness (clipped): {skew_val:.4f}")
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"{col.upper()} Distribution")

        # Histogram with KDE
        sns.histplot(
            data_vis,
            kde=True,
            ax=axes[0],
            color='skyblue',
            alpha=0.7
        )
        axes[0].set_title(f"{col} (Histogram + KDE)")
        axes[0].set_xlabel(col)
        
        # Add vertical lines for statistics on histogram
        # Mean
        axes[0].axvline(x=mean_val, color='green', linestyle='-', 
                       label=f'Mean: {mean_val:.2f}')
        # Median
        axes[0].axvline(x=median_val, color='blue', linestyle='-', 
                       label=f'Median: {median_val:.2f}')
        
        # Standard deviation lines (±1σ)
        if mean_val + std_val <= upper_bound:
            axes[0].axvline(x=mean_val + std_val, color='purple', linestyle=':', 
                           label=f'Mean + 1σ: {mean_val + std_val:.2f}')
        else:
            axes[0].axvline(x=upper_bound, color='purple', linestyle=':', 
                           label=f'Mean + 1σ: {mean_val + std_val:.2f} (beyond view)')
            
        if mean_val - std_val >= data_vis.min() and mean_val - std_val > 0:
            axes[0].axvline(x=mean_val - std_val, color='purple', linestyle=':', 
                           label=f'Mean - 1σ: {mean_val - std_val:.2f}')
        elif mean_val - std_val > 0:
            axes[0].axvline(x=data_vis.min(), color='purple', linestyle=':', 
                           label=f'Mean - 1σ: {mean_val - std_val:.2f} (beyond view)')
        
        # 3σ boundaries for outlier detection
        if cutoff_high <= upper_bound:
            axes[0].axvline(x=cutoff_high, color='red', linestyle='--', 
                           label=f'3σ upper bound: {cutoff_high:.2f}')
        else:
            axes[0].axvline(x=upper_bound, color='red', linestyle='--', 
                           label=f'3σ upper bound: {cutoff_high:.2f} (beyond view)')
            
        if cutoff_low >= data_vis.min() and cutoff_low > 0:
            axes[0].axvline(x=cutoff_low, color='red', linestyle='--', 
                           label=f'3σ lower bound: {cutoff_low:.2f}')
        elif cutoff_low > 0:
            axes[0].axvline(x=data_vis.min(), color='red', linestyle='--', 
                           label=f'3σ lower bound: {cutoff_low:.2f} (beyond view)')
        
        # Add legend to histogram
        axes[0].legend(loc='upper right', fontsize='small')

        # Boxplot
        sns.boxplot(x=data_vis, ax=axes[1], color='orange')
        axes[1].set_title("Boxplot")
        axes[1].set_xlabel(col)
        
        # Add vertical lines for statistics on boxplot
        # Mean
        axes[1].axvline(x=mean_val, color='green', linestyle='-')
        
        # Standard deviation lines (±1σ)
        if mean_val + std_val <= upper_bound:
            axes[1].axvline(x=mean_val + std_val, color='purple', linestyle=':')
        
        if mean_val - std_val >= data_vis.min() and mean_val - std_val > 0:
            axes[1].axvline(x=mean_val - std_val, color='purple', linestyle=':')
        
        # 3σ boundaries for outlier detection
        if cutoff_high <= upper_bound:
            axes[1].axvline(x=cutoff_high, color='red', linestyle='--')
        else:
            axes[1].axvline(x=upper_bound, color='red', linestyle='--')
            
        if cutoff_low >= data_vis.min() and cutoff_low > 0:
            axes[1].axvline(x=cutoff_low, color='red', linestyle='--')
        
        # Add text annotations for boxplot
        y_pos = 0.9
        axes[1].text(0.02, y_pos, f"Mean: {mean_val:.2f}", transform=axes[1].transAxes, 
                    color='green', fontsize=9, ha='left')
        axes[1].text(0.02, y_pos-0.05, f"Median: {median_val:.2f}", transform=axes[1].transAxes, 
                    color='blue', fontsize=9, ha='left')
        axes[1].text(0.02, y_pos-0.10, f"Var: {var_val:.2f}", transform=axes[1].transAxes, 
                    color='purple', fontsize=9, ha='left')
        axes[1].text(0.02, y_pos-0.15, f"Std: {std_val:.2f}", transform=axes[1].transAxes, 
                    color='purple', fontsize=9, ha='left')
        axes[1].text(0.02, y_pos-0.20, f"Outliers (±3σ): {len(outliers)}", transform=axes[1].transAxes, 
                    color='red', fontsize=9, ha='left')

        plt.tight_layout()

        # Save figure as PNG
        fig_path = os.path.join(save_dir, f"{col}_dist.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {fig_path}")
        
        # Show the plot
        plt.show()


if __name__ == "__main__":
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Path to the CSV file (one level up from the current directory)
    csv_path = os.path.join(os.path.dirname(current_dir), "listings_with_goodness.csv")
    
    print(f"Loading data from: {csv_path}")
    
    # 1) Load the updated listings CSV (which already has 'score_goodness')
    df = pd.read_csv(csv_path)  # use the correct path

    # 2) Create/parse 'distance_km' and 'price' columns
    df = create_distance_price(df)
    
    # 3) Save the updated DataFrame back to the CSV file with the new columns
    print(f"Saving updated DataFrame with 'distance_km' and 'price' columns to: {csv_path}")
    df.to_csv(csv_path, index=False)
    print(f"CSV file updated successfully.")

    # 4) Plot the three columns:
    #    1) 'distance_km'
    #    2) 'price'
    #    3) 'score_goodness' (assumed to exist in CSV)
    columns_to_plot = ["distance_km", "price", "score_goodness"]

    # 5) Explore them and save figures to the 'figures' directory
    save_dir = os.path.join(current_dir, "figures")
    plot_three_features(df, columns=columns_to_plot, clip_percentile=0.99, save_dir=save_dir)
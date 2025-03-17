"""
main.py

Price Prediction with an Embedding-Based Deep Learning Model (Even Further Enhanced)
------------------------------------------------------------------------------------
This script:
  1) Loads "listings_with_goodness.csv".
  2) Cleans data (parse price, remove outliers, compute log_price).
  3) Label-encodes categorical columns, so we can embed them.
  4) Defines a PyTorch feed-forward network that has embeddings for each cat col,
     plus numeric inputs, feeding into a deeper MLP for log(price) – now with 
     BatchNorm, Dropout, skip connections, and an even larger embedding dimension (32).
  5) Uses a cyclical learning rate to help converge to better minima.
  6) Trains the network on 80% of the data, validates on 20%.
  7) Saves the model and demonstrates inference with an example dict of features.
  8) Evaluates on test set to produce an R^2 and a predicted-vs-actual plot.

Changes from previous version:
  • Increased emb_dim from 16 to 32
  • Expanded the MLP depth to 4 hidden layers, each with skip connections
  • Replaced StepLR with torch.optim.lr_scheduler.CyclicLR
  • Retained AdamW and SmoothL1Loss for robust training

────────────────────────────────────────────────────────────────
"""

import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from collections import defaultdict

# --------------------- PATHS ---------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR  = os.path.dirname(CURRENT_DIR)
CSV_FILE    = os.path.join(PARENT_DIR, "listings_with_goodness.csv")
MODEL_SAVE_PATH = os.path.join(CURRENT_DIR, "price_predictor.pth")
# -------------------------------------------------

# ------------------- COLUMNS ---------------------
NUMERIC_COLS = [
    "distance_km",
    "bedrooms",
    "bathrooms",
    "accommodates",
    "number_of_reviews",
    "host_listings_count",
    "beds",
    "review_scores_rating",
    "latitude",
    "longitude",
]
CAT_COLS = [
    "room_type",
    "host_is_superhost",
    "property_type"
]
LOG_PRICE_COL = "log_price"
RESPONSE_COL  = LOG_PRICE_COL  # We'll predict log_price
# -------------------------------------------------

#########################
# Outlier Removal
#########################
def remove_3sigma_outliers_global(df, cols):
    """
    Removes rows where any col in `cols` is beyond ±3σ for that column.
    """
    df_out = df.copy()
    for c in cols:
        mean_c = df_out[c].mean()
        std_c  = df_out[c].std()
        lower  = mean_c - 3.0 * std_c
        upper  = mean_c + 3.0 * std_c
        df_out = df_out[(df_out[c] >= lower) & (df_out[c] <= upper)]
    return df_out

#########################
# Data Cleaning
#########################
def clean_dataframe(df):
    """
    1) Parse 'price' -> float
    2) Force numeric columns to float
    3) Remove rows with price <= 0
    4) Create 'log_price'
    5) Remove ±3σ outliers in numeric columns & log_price
    6) Remove any rows missing numeric/cat columns
    """
    # parse price
    df["price"] = (
        df["price"].astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # Convert the numeric columns
    for c in NUMERIC_COLS:
        # Handle 'bathrooms' strings like "1 bath" or "2 baths"
        if c == "bathrooms":
            df["bathrooms"] = (
                df["bathrooms"].astype(str)
                .str.replace(" bath", "", regex=False)
                .str.replace(" baths", "", regex=False)
            )
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # remove price <= 0
    df = df[df["price"] > 0].copy()

    # create log_price
    df[LOG_PRICE_COL] = df["price"].apply(lambda p: math.log1p(p))

    # remove any rows missing numeric columns
    needed_numeric = NUMERIC_COLS + [LOG_PRICE_COL]
    df = df.dropna(subset=needed_numeric)

    # remove outliers
    df = remove_3sigma_outliers_global(df, needed_numeric)

    # remove rows missing cat columns
    needed_all = NUMERIC_COLS + CAT_COLS + [LOG_PRICE_COL]
    df_clean = df.dropna(subset=needed_all).copy()

    print(f"Data size after cleaning & outlier removal: {len(df_clean)}")
    return df_clean

#########################
# Label Encoder
#########################
class MultiColumnLabelEncoder:
    """
    A simple class that stores label->int mappings for each cat column.
    """
    def __init__(self):
        self.label2idx = {}
        self.idx2label = {}
        self.fitted_   = False

    def fit(self, df, cat_cols):
        for col in cat_cols:
            # gather unique string labels
            unique_vals = sorted(df[col].dropna().astype(str).unique())
            label_map   = {val: i for i, val in enumerate(unique_vals)}
            rev_map     = {i: val for val, i in label_map.items()}
            self.label2idx[col] = label_map
            self.idx2label[col] = rev_map
        self.fitted_ = True

    def transform(self, df, cat_cols):
        df_out = df.copy()
        for col in cat_cols:
            map_dict = self.label2idx[col]
            # map unknown -> 0 or fallback
            df_out[col] = df_out[col].apply(lambda x: map_dict.get(str(x), 0))
        return df_out

    def fit_transform(self, df, cat_cols):
        self.fit(df, cat_cols)
        return self.transform(df, cat_cols)

#########################
# Dataset
#########################
class AirbnbEmbedDataset(Dataset):
    """
    This dataset returns:
      - numeric array [n_numeric]
      - cat array [n_cat]
      - target (log_price)
    We assume cat columns are already label-encoded to int IDs.
    """
    def __init__(self, df, numeric_cols, cat_cols):
        self.num_data = df[numeric_cols].values.astype(np.float32)
        self.cat_data = df[cat_cols].values.astype(np.int64)
        self.y        = df[RESPONSE_COL].values.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_num = self.num_data[idx]
        x_cat = self.cat_data[idx]
        y_val = self.y[idx]
        return (torch.tensor(x_num, dtype=torch.float),
                torch.tensor(x_cat, dtype=torch.long),
                torch.tensor(y_val, dtype=torch.float))


#########################
# Embedding Model
#########################
class PricePredictorEmb(nn.Module):
    """
    One embedding for each categorical column, plus numeric inputs.
    These get concatenated and fed into a deeper MLP for log(price),
    with skip connections, dropout, and batchnorm for more robust training.
    """
    def __init__(self, cat_input_dim_dict, num_input_dim, emb_dim=32):
        super().__init__()
        self.cat_cols = list(cat_input_dim_dict.keys())
        self.emb_layers = nn.ModuleDict()

        for col, vocab_size in cat_input_dim_dict.items():
            self.emb_layers[col] = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=emb_dim
            )

        self.cat_emb_total_dim = len(self.cat_cols) * emb_dim
        
        # all four hidden layers = 512
        self.hidden1 = nn.Linear(num_input_dim + self.cat_emb_total_dim, 512)
        self.bn1     = nn.BatchNorm1d(512)

        self.hidden2 = nn.Linear(512, 512)
        self.bn2     = nn.BatchNorm1d(512)

        self.hidden3 = nn.Linear(512, 512)
        self.bn3     = nn.BatchNorm1d(512)

        self.hidden4 = nn.Linear(512, 512)
        self.bn4     = nn.BatchNorm1d(512)

        self.output  = nn.Linear(512, 1)

        self.dropout = nn.Dropout(p=0.2)
        self.relu    = nn.ReLU()

    def forward(self, x_num, x_cat):
        # Build embeddings
        emb_list = []
        for i, col in enumerate(self.cat_cols):
            col_ids = x_cat[:, i]
            emb_vec = self.emb_layers[col](col_ids)
            emb_list.append(emb_vec)
        cat_emb = torch.cat(emb_list, dim=1)

        merged = torch.cat([x_num, cat_emb], dim=1)

        # layer 1
        h1 = self.hidden1(merged)
        h1 = self.bn1(h1)
        h1 = self.relu(h1)
        h1 = self.dropout(h1)

        # layer 2
        h2 = self.hidden2(h1)
        h2 = self.bn2(h2)
        h2 = self.relu(h2)
        h2 = self.dropout(h2)
        # partial skip: same shape => no error
        h2_out = h2 + 0.3 * h1

        # layer 3
        h3 = self.hidden3(h2_out)
        h3 = self.bn3(h3)
        h3 = self.relu(h3)
        h3 = self.dropout(h3)
        h3_out = h3 + 0.3 * h2_out

        # layer 4
        h4 = self.hidden4(h3_out)
        h4 = self.bn4(h4)
        h4 = self.relu(h4)
        h4 = self.dropout(h4)
        h4_out = h4 + 0.3 * h3_out

        out = self.output(h4_out).squeeze(-1)
        return out

#########################
# Training + Evaluation
#########################
def train_model(df, epochs=40):
    """
    1) Label-encode cat columns
    2) Split train/test
    3) Build dataset + model
    4) Train embedding model with AdamW + cyclical LR (CyclicLR)
    5) Save model state, plus cat vocab sizes
    """
    # 1) label-encode
    cat_encoder = MultiColumnLabelEncoder()
    df_lab = cat_encoder.fit_transform(df, CAT_COLS)

    # 2) train/test
    df_train, df_test = train_test_split(df_lab, test_size=0.2, random_state=42)

    train_ds = AirbnbEmbedDataset(df_train, NUMERIC_COLS, CAT_COLS)
    test_ds  = AirbnbEmbedDataset(df_test, NUMERIC_COLS, CAT_COLS)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False)

    # figure out cat vocab size
    cat_input_dim_dict = {}
    for col in CAT_COLS:
        max_id = df_lab[col].max()
        cat_input_dim_dict[col] = int(max_id + 1)

    # build model
    model = PricePredictorEmb(
        cat_input_dim_dict,
        num_input_dim=len(NUMERIC_COLS),
        emb_dim=32  # bigger embedding dimension
    )
    model.train()

    # AdamW + weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Cyclical LR
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=1e-4,
        max_lr=1e-3,
        step_size_up=100,
        mode='triangular2',
        cycle_momentum=False
    )

    # SmoothL1Loss for robust training
    loss_fn = nn.SmoothL1Loss()

    best_val_loss = float("inf")
    patience = 10
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x_num, x_cat, y in train_loader:
            optimizer.zero_grad()
            preds = model(x_num, x_cat)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()

            # step scheduler each batch
            scheduler.step()

            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_num, x_cat, y in test_loader:
                val_preds = model(x_num, x_cat)
                val_loss += loss_fn(val_preds, y).item()

        train_loss_avg = total_loss / len(train_loader)
        val_loss_avg   = val_loss   / len(test_loader)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss_avg:.3f} | Val Loss: {val_loss_avg:.3f}")

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    # save model
    torch.save({
        "model_state": model.state_dict(),
        "cat_input_dim_dict": cat_input_dim_dict,
        "cat_encoder_label2idx": cat_encoder.label2idx
    }, MODEL_SAVE_PATH)
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    return model, cat_encoder

def load_model_for_inference():
    """
    Recreate the embedding model from the saved checkpoint.
    We'll need cat_input_dim_dict to build PricePredictorEmb.
    """
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location="cpu")
    model_state = checkpoint["model_state"]
    cat_input_dim_dict = checkpoint["cat_input_dim_dict"]

    model = PricePredictorEmb(
        cat_input_dim_dict,
        num_input_dim=len(NUMERIC_COLS),
        emb_dim=32  # must match training
    )
    model.load_state_dict(model_state)
    model.eval()

    # label map, so user can do cat -> int
    cat_label2idx = checkpoint["cat_encoder_label2idx"]
    return model, cat_label2idx

def single_inference(model, cat_label2idx, example):
    """
    example: { numeric + cat columns together }
    We'll build numeric array [1, len(NUMERIC_COLS)] and cat array [1, len(CAT_COLS)].
    Predict log_price, then convert to real price by expm1.
    """
    # label-encode cat columns
    cat_ids = []
    for c in CAT_COLS:
        val_str = str(example[c])
        cat_ids.append(cat_label2idx[c].get(val_str, 0))
    x_cat = np.array([cat_ids], dtype=np.int64)

    # numeric
    num_vals = [example[c] for c in NUMERIC_COLS]
    x_num = np.array([num_vals], dtype=np.float32)

    # forward pass
    with torch.no_grad():
        log_pred = model(torch.tensor(x_num), torch.tensor(x_cat)).item()

    return math.expm1(log_pred)  # real price

def evaluate_and_plot_test_performance(model, cat_label2idx, df, figure_path="test_performance.png"):
    """
    We do the same train/test split as in train_model to replicate the test set,
    then run predictions in a batched manner to compute final R^2 vs actual price.
    """
    df_lab = df.copy()
    for col in CAT_COLS:
        df_lab[col] = df_lab[col].apply(lambda x: cat_label2idx[col].get(str(x), 0))

    # replicate train/test
    _, df_test = train_test_split(df_lab, test_size=0.2, random_state=42)
    test_ds  = AirbnbEmbedDataset(df_test, NUMERIC_COLS, CAT_COLS)
    test_dl  = DataLoader(test_ds, batch_size=64, shuffle=False)

    y_pred_list = []
    y_true_list = []

    model.eval()
    with torch.no_grad():
        for x_num, x_cat, y_true_log in test_dl:
            log_preds = model(x_num, x_cat)
            y_pred_batch = torch.expm1(log_preds)   # actual price
            y_true_batch = torch.expm1(y_true_log)  # actual price
            y_pred_list.extend(y_pred_batch.tolist())
            y_true_list.extend(y_true_batch.tolist())

    y_test = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    r2 = r2_score(y_test, y_pred)
    print(f"Test R^2 (on actual price): {r2:.3f}")

    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.4)
    mn = min(y_test.min(), y_pred.min())
    mx = max(y_test.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], 'r--', label='Perfect Prediction')

    # Add ±10% fill:
    x_line = np.linspace(mn, mx, 100)
    lower_band = x_line * 0.90
    upper_band = x_line * 1.10
    plt.fill_between(x_line, lower_band, upper_band,
                     color='gray', alpha=0.15, label='±10% Band')

    plt.title(f"Deep Residual Embedding Model (R^2={r2:.3f})")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    print(f"Saved test performance figure: {figure_path}")
    plt.close()

def main():
    # load CSV
    print(f"Loading data: {CSV_FILE}")
    df = pd.read_csv(CSV_FILE, low_memory=True)

    # clean
    df_clean = clean_dataframe(df)

    # train
    print("Training deeper embedding model with skip connections and cyclical LR...")
    model, cat_encoder = train_model(df_clean, epochs=40)

    # load model for inference
    model_infer, cat_label2idx = load_model_for_inference()

    # example
    example_input = {
       "distance_km": 5.0,
       "bedrooms": 2,
       "bathrooms": 1,
       "accommodates": 4,
       "number_of_reviews": 10,
       "host_listings_count": 3,
       "beds": 2,
       "review_scores_rating": 95.0,
       "latitude": 51.5074,
       "longitude": -0.1278,
       "room_type": "Entire home/apt",
       "host_is_superhost": "t",
       "property_type": "Apartment"
    }
    pred_price = single_inference(model_infer, cat_label2idx, example_input)
    print(f"Example inference => predicted price: ${pred_price:.2f}")

    # Evaluate & plot
    evaluate_and_plot_test_performance(
        model_infer, cat_label2idx, df_clean,
        figure_path=os.path.join(CURRENT_DIR, "test_performance.png")
    )

if __name__ == "__main__":
    main() 
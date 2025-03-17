import pandas as pd
import numpy as np
import sys
import os

from sentence_transformers import SentenceTransformer, util

def main():
    """
    Usage:
        python embed_listings_in_place.py listings.csv

    1) Loads listings.csv (which has columns like 'id', 'description', etc.).
    2) Embeds each listing's 'description' using a sentence transformer model.
       (Will download the model if not available locally)
    3) Computes similarity to the word 'great'.
    4) Stores the numeric similarity in a new column 'score_goodness'.
    5) Outputs a new CSV 'listings_with_goodness.csv'.
    """

    # ----------------------------------------------------------------
    # 1. Check command-line arguments
    # ----------------------------------------------------------------
    if len(sys.argv) < 2:
        print("Usage: python embed_listings_in_place.py listings.csv")
        sys.exit(1)

    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Handle relative or absolute paths for the input CSV
    listings_csv = sys.argv[1]
    if not os.path.isabs(listings_csv):
        # If a relative path is provided, look for it in the parent directory first
        parent_dir_path = os.path.join(os.path.dirname(current_dir), listings_csv)
        if os.path.exists(parent_dir_path):
            listings_csv = parent_dir_path
        # Otherwise, resolve it relative to the current directory
        elif not os.path.exists(listings_csv):
            listings_csv = os.path.join(current_dir, listings_csv)
    
    if not os.path.exists(listings_csv):
        print(f"Error: Could not find file '{listings_csv}'. Exiting.")
        sys.exit(1)
        
    print(f"Loading listings from: {listings_csv}")

    # ----------------------------------------------------------------
    # 2. Load the listings DataFrame
    # ----------------------------------------------------------------
    df = pd.read_csv(listings_csv)

    # We need a text column, e.g. 'description'.
    # If you prefer 'host_about' or 'neighborhood_overview', change the column name below.
    text_col = 'description'
    if text_col not in df.columns:
        print(f"Error: Column '{text_col}' not in DataFrame. Exiting.")
        sys.exit(1)

    print(f"Number of listings: {len(df)}")

    # ----------------------------------------------------------------
    # 3. Initialize the embedding model
    # ----------------------------------------------------------------
    # Define models to try, in order of preference
    models_to_try = [
        "thenlper/gte-small",  # Smaller, faster model
        "sentence-transformers/all-MiniLM-L6-v2",  # Popular, well-balanced model
        "all-MiniLM-L6-v2"  # Fallback name format
    ]
    
    model = None
    for model_name in models_to_try:
        try:
            print(f"Attempting to load model: {model_name}")
            print("(If this is your first time using this model, it will be downloaded automatically)")
            print("Note: No Hugging Face account is required for these public models")
            
            # Using default cache folder and don't require token authentication
            # This ensures it works without a Hugging Face account
            model = SentenceTransformer(model_name, 
                                        cache_folder=os.path.expanduser("~/.cache/huggingface/hub"),
                                        token=None) 
            
            print(f"Successfully loaded model: {model_name}")
            break
        except Exception as e:
            print(f"Failed to load {model_name}: {str(e)}")
            
            # Check for specific auth-related errors
            if "401 Client Error" in str(e) or "authentication" in str(e).lower():
                print("This appears to be an authentication error.")
                print("Trying next model...")
            elif "Connection" in str(e) or "Timeout" in str(e):
                print("This appears to be a network error.")
                print("Checking internet connection and trying next model...")
    
    if model is None:
        print("\nError: Could not load any embedding model.")
        print("This could be due to network issues or missing dependencies.")
        print("Please ensure you have an internet connection and try again.")
        print("You can also manually install the required packages with:")
        print("pip install -U sentence-transformers torch")
        print("\nNote: The models used in this script are public and do NOT require")
        print("a Hugging Face account or authentication token.")
        sys.exit(1)

    # ----------------------------------------------------------------
    # 4. Embed the word "great" once
    # ----------------------------------------------------------------
    print("Embedding the word 'great'...")
    embedding_great = model.encode("great", convert_to_tensor=True, show_progress_bar=False)

    # ----------------------------------------------------------------
    # 5. Embed each listing's text, compute similarity
    # ----------------------------------------------------------------
    # a) Convert NaN to empty strings to avoid errors
    df[text_col] = df[text_col].fillna("").astype(str)

    # b) Encode all listing texts in one go
    print(f"Embedding the '{text_col}' column for all listings...")
    text_embeddings = model.encode(df[text_col].tolist(), convert_to_tensor=True, show_progress_bar=True)
    # shape = [num_listings, embedding_dim]

    # c) Compute cosine similarity with "great"
    #    shape: [num_listings, 1]
    similarities = util.cos_sim(text_embeddings, embedding_great).squeeze(dim=1).cpu().numpy()
    # shape: [num_listings]

    # d) Store in a new column
    df['score_goodness'] = similarities
    print("Calculated 'score_goodness' for each listing.")

    # ----------------------------------------------------------------
    # 6. Save the result to a new CSV
    # ----------------------------------------------------------------
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Path for the output CSV (one level up from the current directory)
    output_csv = os.path.join(os.path.dirname(current_dir), "listings_with_goodness.csv")
    
    df.to_csv(output_csv, index=False)
    print(f"Saved updated DataFrame to: {output_csv}")

    # Show a sample
    print("\nSample rows with 'score_goodness':")
    print(df[['id', text_col, 'score_goodness']].head(10).to_string(index=False))

    print("Done.")

if __name__ == "__main__":
    main()
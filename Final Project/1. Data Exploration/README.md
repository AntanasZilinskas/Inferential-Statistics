# Listing Embedding Script

This script embeds Airbnb listing descriptions using a sentence transformer model and computes their similarity to the word "great".

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Run the script with your listings CSV file:

```bash
python embed_listings_in_place.py listings.csv
```

## Notes

- **No Hugging Face account is required** - the script uses public models that can be downloaded anonymously
- The script will automatically download the embedding model if it's not available locally
- The first run may take some time as it downloads the model (approximately 100-200MB)
- Subsequent runs will use the cached model and be much faster
- The output will be saved as `listings_with_goodness.csv` in the parent directory

## Requirements

- Python 3.6+
- Internet connection (for first-time model download)
- CSV file with a 'description' column

## Troubleshooting

If you encounter any issues:

1. Ensure you have a working internet connection
2. Check that you have sufficient disk space (~200MB) for the model
3. Make sure you have the latest versions of the dependencies:
   ```bash
   pip install -U sentence-transformers torch
   ``` 
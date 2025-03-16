"""
Script to download datasets from Google Drive links.
The files will be saved in the same directory as this script.
"""

import os
import sys
import gdown

def download_from_gdrive(file_id, output_name):
    """
    Download a file from Google Drive using its file ID.
    
    Args:
        file_id (str): The Google Drive file ID
        output_name (str): The name to save the file as
    
    Returns:
        bool: True if download was successful, False otherwise
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create the full output path
    output_path = os.path.join(script_dir, output_name)
    
    # URL format for Google Drive files
    url = f"https://drive.google.com/uc?id={file_id}"
    
    print(f"Downloading {output_name}...")
    try:
        gdown.download(url, output_path, quiet=False)
        print(f"Successfully downloaded {output_name} to {output_path}")
        return True
    except Exception as e:
        print(f"Error downloading {output_name}: {e}")
        return False

def main():
    # Replace these with your actual Google Drive file IDs and desired output names
    datasets = [
        # Format: (file_id, output_name)
        ("1_wnYGVx5lD9KV6dwPmjlRtQjcrxPKZLG", "listings_with_goodness.csv"),
        ("1MWwgkE1wwn9XuTsKIp2AkT-rdxZHuuJK", "listings.csv"),
    ]

    # Check if gdown is installed
    try:
        import gdown
    except ImportError:
        print("The 'gdown' package is required but not installed.")
        print("Please install it using: pip install gdown")
        sys.exit(1)
    
    # Download each dataset
    success_count = 0
    for file_id, output_name in datasets:
        if download_from_gdrive(file_id, output_name):
            success_count += 1
    
    print(f"Downloaded {success_count} out of {len(datasets)} datasets.")

if __name__ == "__main__":
    main()

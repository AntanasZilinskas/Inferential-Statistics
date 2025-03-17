# Inferential-Statistics

## Getting Started

Follow these 3 steps to set up and run the project:

### 1. Install Dependencies

Install the required packages (requires Python 3.11 or higher):

```bash
pip install -r requirements.txt
```

### 2. Download Datasets

Navigate to the Final Project folder and either:
- Place the provided datasets directly into the directory, or
- Run the download script to fetch them from Google Drive:

```bash
python download_dataset.py
```

### 3. Run the Project

Each of the project components can be run individually. Within each numbered folder (e.g., "1. Data Exploration", "2. Verifying the Central Limit Theorem (CLT)", etc.), there is a main.py script you can run to reproduce that module's results:

Example:
```bash
cd "Final Project/2. Verifying the Central Limit Theorem (CLT)"
python main.py
```

This command processes the data in that folder and produces the specified results (figures, CSVs, etc.).

## Running the Deep Learning Web App

To launch the interactive price prediction web application (located in "Final Project/6. Price Prediction with Deep Learning"), navigate into that folder and run:

```bash
cd "Final Project/6. Price Prediction with Deep Learning"
streamlit run webapp.py
```

This will open a local Streamlit web app in your browser. There, you can provide property details and see the predicted nightly rate.

### Using the Pre-Trained Model or Training From Scratch

Inside "Final Project/6. Price Prediction with Deep Learning", there is a file named price_predictor.pth, which is the saved PyTorch model. You have two options:

1. Train the model by running:
   ```bash
   python main.py
   ```
   This will produce a fresh price_predictor.pth once training is complete.

2. Place a valid price_predictor.pth file manually into the same directory (if you have a pre-trained model ready), allowing the webapp and scripts to use it for inference without requiring a new training run.

## Project Structure

The project is organized as follows:
- `Final Project/`: Contains the main project files and datasets
  - `download_dataset.py`: Script to download required datasets from Google Drive
  - Folders labeled 1–6, each containing a main.py that shows a different aspect of the project
- Additional code and data files for each module

## License

This project is for educational purposes only and is not intended for commercial use. No specific license is applied.
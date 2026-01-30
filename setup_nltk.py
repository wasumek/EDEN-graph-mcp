import nltk
import os

# Define the download directory
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')

# Create the directory if it doesn't exist
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Append to NLTK path
nltk.data.path.append(nltk_data_dir)

# Download required packages
# 'punkt_tab' is required for newer NLTK versions (>=3.8.2) for sentence tokenization
packages = ['stopwords', 'wordnet', 'punkt', 'punkt_tab']

print(f"Downloading NLTK data to {nltk_data_dir}...")

for package in packages:
    try:
        nltk.download(package, download_dir=nltk_data_dir)
        print(f"Successfully downloaded {package}")
    except Exception as e:
        print(f"Error downloading {package}: {e}")

print("NLTK setup complete.")

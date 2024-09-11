import pandas as pd
from pathlib import Path


# Load your CSV file
df = pd.read_csv('datasets/dataset2.csv')


# Define the base directory to make paths relative to
# base_dir = Path('C:/Users/Ashish Mahendran/Documents/lung cancer preprocessing/')


# Update the image paths to be relative
# df['image_path'] = df['image_path'].apply(lambda x: str(Path(x).relative_to(base_dir)))


# Replace backslashes with forward slashes
df['image_path'] = df['image_path'].str.replace('\\', '/', regex=False)


# Save the updated CSV with relative paths
df.to_csv('updated_datasets/dataset3.csv', index=False)
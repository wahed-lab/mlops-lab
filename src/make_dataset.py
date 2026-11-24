import os
from sklearn.datasets import load_breast_cancer

# Create the raw data directory if it does not exist
os.makedirs("data/raw", exist_ok=True)

# Load the Breast Cancer dataset
df = load_breast_cancer(as_frame=True).frame

# Save the dataset as a CSV file
df.to_csv("data/raw/dataset.csv", index=False)

# Display the dataset information
print("Saved data/raw/dataset.csv with shape", df.shape)

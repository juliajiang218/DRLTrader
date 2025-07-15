import pandas as pd
import os

# Load the train_data.csv file from datasets directory
file_path = os.path.join('datasets', 'train_data.csv')
df = pd.read_csv(file_path)

# Display basic information about the DataFrame
print("DataFrame loaded successfully!")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nFirst few rows:")
print(df.head())

print("\nDataFrame info:")
print(df.info()) 
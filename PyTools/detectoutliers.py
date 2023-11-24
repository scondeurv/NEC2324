import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from scipy import stats

# Function to detect outliers using Z-Score for a single feature
def detect_outliers_zscore(data_column, threshold):
    z_scores = np.abs(stats.zscore(data_column))
    return data_column[z_scores > threshold]

# Function to detect outliers using IQR for a single feature
def detect_outliers_iqr(data_column):
    Q1 = data_column.quantile(0.25)
    Q3 = data_column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data_column[(data_column < lower_bound) | (data_column > upper_bound)]
    return outliers

# Function to detect and visualize outliers for each feature
def detect_and_visualize_outliers(df, output_folder, method, threshold):
    for column in df.columns:
        if method == "zscore":
            outliers = detect_outliers_zscore(df[column], threshold)
        else:
            method = "iqr"
            outliers = detect_outliers_iqr(df[column])
        
        plt.figure(figsize=(8, 6))
        plt.scatter(df.index, df[column], label='Data')
        plt.scatter(outliers.index, outliers, color='red', label='Outliers')
        plt.xlabel('Data Point')
        plt.ylabel('Value')
        plt.legend()
        plt.title(f'{column}')
        
        # Save the plot as an image in the output folder
        output_path = os.path.join(output_folder, f'{column}_outliers_{method}.png')
        plt.savefig(output_path)
        plt.close()

if len(sys.argv) < 4:
    print("Usage: python detectoutliers.py <file_path> <delimiter> <output_folder> [<method>] [<threshold>]")
    sys.exit(1)

file_path = sys.argv[1]
delimiter = sys.argv[2]
output_folder = sys.argv[3]
method = sys.argv[4]
threshold = float(sys.argv[5])

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load the dataset from the CSV file
try:
    df = pd.read_csv(file_path, sep=delimiter)
except FileNotFoundError:
    print(f"File '{file_path}' not found.")
    sys.exit(1)

# Perform outlier detection and visualization for each feature and save the plots
detect_and_visualize_outliers(df, output_folder, method, threshold)

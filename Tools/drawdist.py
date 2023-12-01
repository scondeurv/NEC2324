import argparse
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
import numpy as np

def main(input_file, output_folder, delimiter, scaling_method):

    data = pd.read_csv(input_file, delimiter=delimiter)

    scaler = None
    if scaling_method == "min-max":
        scaler = MinMaxScaler()
    elif scaling_method == "z-score":
        scaler = StandardScaler()

    os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist

    statistics = {}

    for column in data.columns:
        if scaler is not None:
            scaled_column = scaler.fit_transform(data[column].values.reshape(-1, 1))
            data[column] = scaled_column.squeeze()
            
        # Handle NaNs after scaling
        data[column].fillna(data[column].mean(), inplace=True)

        # Calculate statistics
        mean = data[column].mean()
        std_dev = data[column].std()
        min_value = data[column].min()
        max_value = data[column].max()

        # Store statistics in a dictionary
        statistics[column] = {
            'Min': min_value,
            'Max': max_value,
            'Mean': mean,
            'Std Dev': std_dev
        }

        # Calculate the number of bins using the square root choice
        num_bins = int(np.sqrt(len(data[column])))

        # Plot distribution using histogram
        plt.figure(figsize=(8, 6))
        n, bins, patches = plt.hist(data[column], bins=num_bins, density=True, alpha=0.5, color='b')

        # Plot the normal distribution curve with more points
        x = np.linspace(min_value, max_value, 1000)
        y = norm.pdf(x, mean, std_dev)
        plt.plot(x, y, 'r--', linewidth=2)

        # Add axis labels
        plt.xlabel(f'{column}')
        plt.ylabel('Probability Density')

        # Save the distribution plot
        distribution_image = os.path.join(output_folder, f'{column}.png')
        plt.savefig(distribution_image)
        plt.close()

    # Save statistics to a CSV file
    statistics_df = pd.DataFrame(statistics).T
    statistics_file = os.path.join(output_folder, 'feature_statistics.csv')
    statistics_df.to_csv(statistics_file)

    print("Distribution histograms with normal distribution curves and feature statistics saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate distribution histograms with normal distribution curves and feature statistics for a delimited file")
    parser.add_argument("input_file", help="Input delimited file containing the dataset")
    parser.add_argument("output_folder", help="Output folder for saving the histograms and feature statistics")
    parser.add_argument("--delimiter", required=False, default='\t', help="Delimiter used in the input file")
    parser.add_argument("--scaling-method", required=False, help="Scaling/standardization method ('min-max' or 'z-score')")

    args = parser.parse_args()
    main(args.input_file, args.output_folder, args.delimiter, args.scaling_method)

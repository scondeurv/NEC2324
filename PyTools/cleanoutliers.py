import argparse
import pandas as pd
import os

def remove_outliers(data, method, z_threshold):
    if method == 'z-score':
        z_scores = (data - data.mean()) / data.std()
        data = data[(z_scores.abs() <= z_threshold).all(axis=1)]
    elif method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR)).any(axis=1))]
    return data

def main(input_file, delimiter, method, z_threshold, output_file):
    # Read the dataset
    data = pd.read_csv(input_file, delimiter=delimiter)

    # Remove outliers
    cleaned_data = remove_outliers(data, method, z_threshold)

    # Get the folder containing the input file
    input_folder = os.path.dirname(input_file)

    # Save the cleaned dataset to the output file in the same folder as the input file
    output_path = os.path.join(input_folder, output_file)
    cleaned_data.to_csv(output_path, sep=delimiter, index=False)

    print(f"Cleaned dataset saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove outliers from a dataset and save the cleaned dataset in the same folder as the input file")
    parser.add_argument("input_file", help="Input dataset filename")
    parser.add_argument("--delimiter", default='\t', help="Delimiter used in the input dataset (default: tab)")
    parser.add_argument("method", choices=["z-score", "iqr"], help="Outliers detection method (z-score or iqr)")
    parser.add_argument("--z-threshold", type=float, default=3, help="Z-score threshold for outlier detection (default: 3)")
    parser.add_argument("output_file", help="Output filename for the cleaned dataset")

    args = parser.parse_args()
    main(args.input_file, args.delimiter, args.method, args.z_threshold, args.output_file)

'''
Stephen Venable
2024-03-05

This module contains a function to generate user probabilities based on access data.
'''
import pandas as pd
import numpy as np
import os
from preprocessing import preprocess_csv
from tqdm import tqdm

def generate_user_probabilities():
    """
    Generate user probabilities based on access data.

    This function prompts the user to select a file from a directory,
    loads the data from the selected file, processes the data to calculate
    probabilities, normalizes the probabilities, and saves the results to a CSV file.

    Returns:
        None
    """

    data = preprocess_csv()

    # Create a DataFrame to store the probabilities
    row_headers = []
    app_paths = data["app_path"].unique()
    app_paths = np.insert(app_paths, 0, "empty")

    for user in data["user_name"].unique():
        for app_path in app_paths:
            row_headers.append(user + " " + app_path)

    column_headers = data["app_path"].unique()

    probabilities = pd.DataFrame(index=row_headers, columns=column_headers)

    probabilities = pd.DataFrame(
        np.zeros((len(row_headers), len(column_headers))),
        index=row_headers,
        columns=column_headers,
    )

    # Calculate the probabilities
    already_tested = []

    for k in tqdm(range(len(data))):
        current_user = data["user_name"][k]
        current_app_path = data["app_path"][k]
        if current_user + " " + current_app_path in already_tested:
            continue
        already_tested.append(current_user + " " + current_app_path)
        for i in range(len(data) - 1):
            data_user = data["user_name"][i]
            next_user = data["user_name"][i + 1]
            data_app_path = data["app_path"][i]
            next_app_path = data["app_path"][i + 1]
            if (
                (data_user == current_user)
                and (data_app_path == current_app_path)
                and (next_app_path != current_app_path)
            ):
                if (next_user == current_user):
                    probabilities.loc[current_user + " " + current_app_path, next_app_path] += 1
                else:
                    probabilities.loc[current_user + " empty" , next_app_path] += 1

    # Normalize the probabilities
    probabilities = probabilities.div(probabilities.sum(axis=1), axis=0)

    # Replace NaN values with 0
    probabilities = probabilities.fillna(0)

    # Set the current directory
    current_dir = os.path.dirname(__file__)

    # Set FilePath
    file_path = os.path.join(current_dir, "../tables/user_probabilities.csv")

    # Save the normalized probabilities to a CSV file
    probabilities.to_csv(file_path, index=True, header=True)

# Build a main
if __name__ == "__main__":
    generate_user_probabilities()

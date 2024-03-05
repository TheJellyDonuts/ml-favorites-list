'''
Stephen Venable
2024-03-05

'''
import pandas as pd
import numpy as np
import os
from preprocessing import preprocess_csv

def generate_generic_stats():

    data = preprocess_csv()

    # Create a DataFrame to store the probabilities
    row_headers = data["app_path"].unique()
    row_headers = np.insert(row_headers, 0, "empty")

    column_headers = data["app_path"].unique()

    probabilities = pd.DataFrame(index=row_headers, columns=column_headers)

    probabilities = pd.DataFrame(
        np.zeros((len(row_headers), len(column_headers))),
        index=row_headers,
        columns=column_headers,
    )

    # Calculate the probabilities
    already_tested = []

    for k in range(len(data)):
        current_app_path = data["app_path"][k]
        if current_app_path in already_tested:
            continue
        already_tested.append(current_app_path)
        for i in range(len(data) - 1):
            data_user = data["user_name"][i]
            next_user = data["user_name"][i + 1]
            data_app_path = data["app_path"][i]
            next_app_path = data["app_path"][i + 1]
            if (
                (data_app_path == current_app_path)
                and (next_app_path != current_app_path)
            ):
                if data_user != next_user:
                    probabilities.loc["empty", next_app_path] += 1
                else:
                    probabilities.loc[current_app_path, next_app_path] += 1

    # Normalize the probabilities
    probabilities = probabilities.div(probabilities.sum(axis=1), axis=0)

    # Replace NaN values with 0
    probabilities = probabilities.fillna(0)

    # Set the current directory
    current_dir = os.path.dirname(__file__)

    # Set FilePath
    file_path = os.path.join(current_dir, "../results/generic_probabilities.csv")

    # Save the normalized probabilities to a CSV file
    probabilities.to_csv(file_path, index=True, header=True)

if __name__ == "__main__":
    generate_generic_stats()

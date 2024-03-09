'''
Stephen Venable
2024-03-05

This module contains a function to preprocess a CSV file.
'''
import os
import inquirer
import pandas as pd

def preprocess_csv():
    """
    Preprocess the data from a CSV file.

    This function prompts the user to select a file from a directory,
    loads the data from the selected file, processes the data to remove
    duplicates and sort the data, and returns the processed data.

    Returns:
        pd.DataFrame: The processed data

    """
    # Set the current directory
    current_dir = os.path.dirname(__file__)

    # Set the data directory
    data_dir = os.path.join(current_dir, "../../data/input")

    # List all the files in the 'data' directory
    file_options = os.listdir(data_dir)

    # Select the file to use
    questions = [
        inquirer.List(
            "selected_file",
            message="Select the file to use",
            choices=file_options,
        ),
    ]

    answers = inquirer.prompt(questions)
    selected_file = answers["selected_file"]

    # Print the selected file
    print("Processing the file: " + selected_file)

    # Load the data from the CSV file
    csv_file_path = os.path.join(data_dir, selected_file)
    data = pd.read_csv(csv_file_path, header=0)

    # Convert 'access_date' to datetime format
    data["access_date"] = pd.to_datetime(data["access_date"])

    # Remove the prefix '/apps/kardia/modules/' from 'app_path'
    data["app_path"] = data["app_path"].str.replace("/apps/kardia/modules/", "")

    # Remove rows that are complete duplicates in every value
    data = data.drop_duplicates(ignore_index=True)

    # Sort the data by 'user_name' and 'access_date'
    data = data.sort_values(by=["user_name", "access_date"])

    return data
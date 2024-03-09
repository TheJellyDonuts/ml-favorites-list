"""
Stephen Venable
2024-03-05
"""

import pandas as pd
import inquirer
import os
from termcolor import colored
from prettytable import PrettyTable

# Select type of probability to use
# Set the current directory
current_dir = os.path.dirname(__file__)

# Set the data directory
tables_dir = os.path.join(current_dir, "tables")

# List all the files in the 'tables' directory
file_options = os.listdir(tables_dir)
questions = [
    inquirer.List(
        "selected_probability",
        message="Enter the type of probability to use",
        choices=file_options,
    ),
]

answers = inquirer.prompt(questions)
selected_file = answers["selected_probability"]


# Load the probabilities from the CSV file
file_path = os.path.join(tables_dir, selected_file)
normal_probabilities = pd.read_csv(file_path, header=0, index_col=0)

next_paths = None
# Given a user name and current path, using the probabilities, predict the next path
if selected_file == "user_probabilities.csv":
    input_user = input(
        colored("Enter the user name ", "green")
        + colored("(e.g. 'tadeovbb')", "yellow")
        + colored(": ", "green")
    )
    input_path = input(
        colored("Enter the current path ", "green")
        + colored("(e.g. '/apps/kardia/modules/base/partner_window.app')", "yellow")
        + colored(": ", "green")
    )
    input_path = input_path.replace("/apps/kardia/modules/", "")
    input_path = input_user + " " + input_path

else:
    input_path = input(
        colored("Enter the current app path ", "green")
        + colored("(e.g. '/apps/kardia/modules/base/partner_window.app')", "yellow")
        + colored(": ", "green")
    )

input_path = input_path.replace("/apps/kardia/modules/", "")

if input_path not in normal_probabilities.index:
    print(
        colored(
            "The given information was not found in " + selected_file + ".",
            "red",
        )
    )
    exit()

next_paths = normal_probabilities.loc[input_path].nlargest(3)

# Create a new PrettyTable
table = PrettyTable()

# Add columns
table.field_names = ["Path", "Probability"]

# Add rows
for index, value in next_paths.items():
    table.add_row([index, "{:.2%}".format(value)])

# Print the table
print("The next 3 most probable paths are:")
print(table)

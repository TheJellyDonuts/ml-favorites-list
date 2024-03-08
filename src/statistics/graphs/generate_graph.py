'''
Stephen Venable
2024-03-07

This script generates a pie chart with the number of accesses per app for a given path.
The user can select the file to use, and then enter the user name and current path, 
or just the current path. The script will then generate the pie chart 
and save it in the 'images' directory.
'''

# Import the necessary libraries
import os
import pandas
import matplotlib.pyplot as plt
import inquirer
from termcolor import colored

# Set the current directory
current_dir = os.path.dirname(__file__)

# Set the data directory
results_dir = os.path.join(current_dir, "../tables")

# List all the files in the 'results' directory
file_options = os.listdir(results_dir)

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

file_to_use = os.path.join(results_dir, selected_file)

# Read the data from the file
data = pandas.read_csv(file_to_use, index_col=0, header=0)

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

if input_path not in data.index:
    print(
        colored(
            "The given information was not found in " + selected_file + ".",
            "red",
        )
    )
    exit()

# Ask the user if they want to see the probabilities
questions = [
    inquirer.Confirm("see_probabilities", message="Do you want to see the probabilities?", default=True)
]

answers = inquirer.prompt(questions)
see_probabilities = answers["see_probabilities"]

# Ask the user if they want to save the graph
questions = [
    inquirer.Confirm("save_graph", message="Do you want to save the graph?", default=False)
]

answers = inquirer.prompt(questions)
save_graph = answers["save_graph"]

# Get the row of data from the input
row = data.loc[input_path]

# Get the apps
apps = row.index[1:]

# Get the number of accesses
accesses = row[1:]

# Remove all elements with 0 accesses
apps = apps[accesses > 0]
accesses = accesses[accesses > 0]

# sort the data
accesses, apps = zip(*sorted(zip(accesses, apps)))

# Only show the top 5 elements, and group the rest into 'Other'
if len(apps) > 5:
    apps = apps[-5:] + ("Other",)
    accesses = accesses[-5:] + (sum(accesses[:-5]),)

# Create a pie chart
plt.pie(accesses, autopct="%1.1f%%")
plt.axis("equal")
plt.title(input_path + " - Number of accesses per app")

plt.legend(apps, loc="best")


# Save the graph
output_dir = os.path.join(current_dir, "images")

# Sanitize the input path
input_path = input_path.replace("/", "_")
input_path = input_path.replace(" ", "_")
input_path = input_path.replace(".app", "")

output_path = os.path.join(output_dir, input_path + ".png")

print("Saving the graph to " + output_path)
print("Save Graph", save_graph)
print("See Probabilities", see_probabilities)

if save_graph:
    plt.savefig(output_path)

# Show the graph
if see_probabilities:
    plt.show()
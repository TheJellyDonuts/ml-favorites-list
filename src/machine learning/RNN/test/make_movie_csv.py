import pandas as pd
import os

# Get Current Directory
cur_dir = os.path.dirname(__file__)

# Set the data file path
data_dir = os.path.join(cur_dir, '../../../data')

# Set File Path
file_path = os.path.join(data_dir, "kardia_data_one_year.csv")

# Load the data
data = pd.read_csv(file_path, header=0)

# Get all the unique app paths
unique_apps = data['app_path'].unique()

# Store the unique app paths in a CSV file
unique_apps_df = pd.DataFrame(unique_apps, columns=['app_path'])

# Output Path
output_file_path = os.path.join(cur_dir, 'unique_apps.csv')

unique_apps_df.to_csv(output_file_path, index=False)

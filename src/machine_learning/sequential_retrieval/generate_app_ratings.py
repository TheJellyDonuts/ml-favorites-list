import pandas as pd
import os

# Get Current Directory
cur_dir = os.path.dirname(__file__)

# Set the data file path
data_dir = os.path.join(cur_dir, '../../data/input')

# Set File Path
file_path = os.path.join(data_dir, "kardia_data_five_years.csv")

# Load the data
data = pd.read_csv(file_path, header=0)

# * Create a CSV file with Unique App Paths
# Get all the unique app paths
unique_apps = data['app_path'].unique()

# Store the unique app paths in a CSV file
unique_apps_df = pd.DataFrame(unique_apps, columns=['app_path'])

# Output Path
output_file_path = os.path.join(cur_dir, '../../data/generated/unique_apps.csv')

# Save the unique app paths to a CSV file
unique_apps_df.to_csv(output_file_path, index=False)


# * Create a CSV file with App Ratings
# Generate app ratings based on how frequently apps are used
app_ratings = data['app_path'].value_counts().reset_index()
app_ratings.columns = ['app_path', 'rating']

# Output Path
output_file_path = os.path.join(cur_dir, '../../data/generated/app_ratings.csv')

# Save the app ratings to a CSV file
app_ratings.to_csv(output_file_path, index=False)
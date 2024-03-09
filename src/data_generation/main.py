import random, csv, pandas, argparse
from datetime import datetime
from tqdm import tqdm
import os

# Initialize according to user-provided arguments
parser = argparse.ArgumentParser(
  prog='Data Generator',
  description='Generates random data to be analyzed by our various algorithms.',
)
parser.add_argument(
  'num_entries',
  type=int,
  default=100,
  help='The number of data entries to generate.',
  nargs='?'
)
args = parser.parse_args()

# Get the current directory
cur_dir = os.path.dirname(__file__)

# Generate a list of dateTimes within a range
random.seed()
dates = []
start_date = datetime(2023, 1, 1, 00, 00, 00)
end_date = datetime(2023, 12, 31, 23, 59, 59)

for i in tqdm(range(args.num_entries), desc='Generating random dates'):
  random_date = start_date + (end_date - start_date) * random.random()
  formatted_date = datetime.strftime(random_date, '%Y-%m-%d %H:%M:%S')
  dates.append(formatted_date)

# Import users and apps from csvs
data_dir = os.path.join(cur_dir, 'data')
user_csv_path = os.path.join(data_dir, "default_users.csv")
data = pandas.read_csv(user_csv_path, header=0)
users = data['user_name'].tolist()
app_csv_path = os.path.join(data_dir, "default_apps.csv")
data = pandas.read_csv(app_csv_path, header=0)
apps = data['app_path_concise'].tolist()

# Pair every date with a random user and random app
entries = []
for i in tqdm(range(len(dates)), desc='Generating table entries'):
  new_entry = {
    'access_date': (str)(dates[i]),
    'user_name': random.choice(users),
    'app_path': random.choice(apps)
  }
  entries.append(new_entry)

# Write entries to csv
output_dir = os.path.join(cur_dir, '../data')
output_file_path = os.path.join(output_dir, "dummy_data.csv")
print("Writing entries to " + output_file_path + "...")
with open(output_file_path, 'w', newline='') as file:
  writer = csv.DictWriter(file, fieldnames=['access_date', 'user_name', 'app_path'])
  writer.writeheader()
  writer.writerows(entries)

# Print sorted csv
print('Previewing csv as sorted pandas dataframe:')
dataframe = pandas.read_csv(output_file_path, header=0)
dataframe = dataframe.sort_values(by=['user_name', 'access_date'])
print(dataframe)
print("Entries saved to " + output_file_path + ".")

# Future prospects: have an option for users to use and/or generate their own usernames and apps using the Faker library

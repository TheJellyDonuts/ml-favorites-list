import random, csv, pandas, argparse
from datetime import datetime
from tqdm import tqdm

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
data = pandas.read_csv('default_users.csv', header=0)
users = data['user_name'].tolist()
data = pandas.read_csv('default_apps.csv', header=0)
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

print('Writing entries to "test_data/dummy_data.csv"...')
with open('dummy_data.csv', 'w', newline='') as file:
  writer = csv.DictWriter(file, fieldnames=['access_date', 'user_name', 'app_path'])
  writer.writeheader()
  writer.writerows(entries)

# Print sorted csv
print('Previewing csv as sorted pandas dataframe:')
dataframe = pandas.read_csv('dummy_data.csv', header=0)
dataframe = dataframe.sort_values(by=['user_name', 'access_date'])
print(dataframe)
print('Entries saved to "test_data/dummy_data.csv"')

# Future prospects: have an option for users to use and/or generate their own usernames and apps using the Faker library

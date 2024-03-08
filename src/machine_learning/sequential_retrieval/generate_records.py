import pandas as pd
import os
from tqdm import tqdm

# Get Cur Dir
cur_dir = os.path.dirname(__file__)
input_path = os.path.join(cur_dir, '../../data/input/kardia_data_five_years.csv')

# Read input.csv
input_data = pd.read_csv(input_path)

# Generate App.csv
unique_app_path = input_data['app_path'].drop_duplicates().tolist()

app_full_names = [path.replace('/apps/kardia/modules/', "") for path in unique_app_path]

# Split app_full_name into app_name and parent_dir
app_name = [app.split('/')[-1] for app in app_full_names]
parent_dir = [app.split('/')[-2] for app in app_full_names]

app_df = pd.DataFrame({'name': app_name, 'category': parent_dir})

app_path = os.path.join(cur_dir, '../../data/generated/apps.csv')
app_df.to_csv(app_path, header=False)

# Generate User.csv (columsn of UserID and Username)
unique_users = input_data['user_name'].drop_duplicates().tolist()

user_df = pd.DataFrame({'name': unique_users})

user_path = os.path.join(cur_dir, '../../data/generated/users.csv')
user_df.to_csv(user_path, header=False)


# Generate Ratings.csv
app_data = pd.read_csv(app_path, header=None)
user_data = pd.read_csv(user_path, header=None)
app_rating = pd.read_csv(os.path.join(cur_dir, '../../data/generated/app_ratings.csv'), header=0)

print(app_rating.head())

# Rating {columns of userID, appID, rating, timestamp}

# Generate empty PD table with headers of userID, appID, rating, timestamp
ratings_df = pd.DataFrame(columns=['user_id', 'app_id', 'rating', 'timestamp'])

for index in tqdm(range(len(input_data))):
    for app_index in range(len(app_data)):
        if app_data[1][app_index] in input_data['app_path'][index]:
            app_id = app_data[0][app_index]
            break
    for user_index in range(len(user_data)):
        if user_data[1][user_index] in input_data['user_name'][index]:
            user_id = user_data[0][user_index]
            break
    for rating_index in range(len(app_rating)):
        if app_rating["app_path"][rating_index] in input_data['app_path'][index]:
            rating = app_rating["rating"][rating_index]
            # Modify the rating value so it is between 0 and 10
            rating = (rating/app_rating.max().iloc[1]) * 10
            break
    timestamp = pd.to_datetime(input_data['access_date'][index]).timestamp()
    ratings_df = pd.concat([ratings_df, pd.DataFrame({'user_id': [user_id], 'app_id': [app_id], 'rating': [rating], 'timestamp': [timestamp]})])

ratings_path = os.path.join(cur_dir, '../../data/generated/ratings.csv')
ratings_df.to_csv(ratings_path, header=False, index=False)

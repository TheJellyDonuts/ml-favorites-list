# Read for more information: https://surprise.readthedocs.io/en/stable/index.html
'''
This program finds PREDICTION_NUM recommended apps per user given their open 
history from ratings.csv. It is unclear how accurate this is, as the statistics
don't have a comprable analysis.
The values in param_grid are adjustable; the program automatically does a search
within the ranges specified for each value to find a relative optimum. Beware of 
using too large of a range, or it will be to large a grain to find more optimal
values. The best result we have so far is in surprise_res.json.
'''

import os
from surprise import Dataset, Reader, SVD
from surprise.model_selection import GridSearchCV
from collections import defaultdict
import pandas as pd

PREDICTION_NUM = 3
param_grid = {"n_epochs": [500, 15000], "lr_all": [0.005, 0.5], "reg_all": [0.005, 0.5]}

def get_top_n(predictions, n=10):
    """Return the top-N recommendation for each user from a set of predictions.

    Note: This function depends on the existence of ./src/data/generated/ratings.csv.
    Run the ./src/machine_learning/sequential_retrieval/generate_records.py script first.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


# get data from dataset file
file_path = os.path.expanduser("./src/data/generated/ratings.csv")
reader = Reader(line_format="user item rating timestamp", sep=",", skip_lines=1)
data = Dataset.load_from_file(file_path, reader=reader)

# optimize parameters
print('Optimizing parameters...')
gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=5, n_jobs=-2)
gs.fit(data)


# use the algorithm that yields the best rmse:
algo = gs.best_estimator["rmse"]
print('Building training set...')
trainset = data.build_full_trainset()
print('Fitting...')
algo.fit(trainset)


# predict ratings for all pairs (u, i) that are NOT in the training set.
testset = trainset.build_anti_testset()
predictions = algo.test(testset)
top_n = get_top_n(predictions, n=PREDICTION_NUM)

# translate iid to app_name and uid to user_name
cur_dir = os.path.dirname(__file__)
app_path = './src/data/generated/apps.csv' #os.path.join(cur_dir, './src/data/generated/apps.csv')
user_path = './src/data/generated/users.csv' #os.path.join(cur_dir, './src/data/generated/users.csv')
app_df = pd.read_csv(app_path, header=None, names=['app_id', 'app_name', 'app_category'])
user_df = pd.read_csv(user_path, header=None, names=['user_id', 'user_name'])

# make dictionaries with ids as key and names as values
app_df = app_df.set_index('app_id').to_dict()['app_name']
user_df = user_df.set_index('user_id').to_dict()['user_name']


# best RMSE score
print('RSME:', gs.best_score["rmse"])

# combination of parameters that gave the best RMSE score
print('Params:', gs.best_params["rmse"])

# print the recommended items by user
for uid, user_ratings in top_n.items():
    u = "{:<15} ".format(user_df[int(uid)])
    print(u, [app_df[int(iid)] for (iid, _) in user_ratings], ',', sep='')

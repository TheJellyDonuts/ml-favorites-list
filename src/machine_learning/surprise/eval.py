import os

from surprise import Dataset, Reader
from surprise import Dataset, SVD
from surprise.model_selection import GridSearchCV
from collections import defaultdict
import pandas as pd

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


# path to dataset file
file_path = os.path.expanduser("./src/data/generated/ratings.csv")
reader = Reader(line_format="user item rating timestamp", sep=",", skip_lines=1)
data = Dataset.load_from_file(file_path, reader=reader)

print('Optimizing parameters...')
param_grid = {"n_epochs": [500, 15000], "lr_all": [0.005, 0.5], "reg_all": [0.005, 0.5]}
gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=5, n_jobs=-2)
# {'n_epochs': 50, 'lr_all': 0.1, 'reg_all': 5e-05}
gs.fit(data)



# We can now use the algorithm that yields the best rmse:
algo = gs.best_estimator["rmse"]
print('Building training set...')
trainset = data.build_full_trainset()
print('Fitting...')
algo.fit(trainset)


# Than predict ratings for all pairs (u, i) that are NOT in the training set.
testset = trainset.build_anti_testset()
predictions = algo.test(testset)

top_n = get_top_n(predictions, n=3)


# Translate iid to app_name
cur_dir = os.path.dirname(__file__)
app_path = './src/data/generated/apps.csv' #os.path.join(cur_dir, './src/data/generated/apps.csv')
user_path = './src/data/generated/users.csv' #os.path.join(cur_dir, './src/data/generated/users.csv')
app_df = pd.read_csv(app_path, header=None, names=['app_id', 'app_name', 'app_category'])
user_df = pd.read_csv(user_path, header=None, names=['user_id', 'user_name'])

# Make app_df a dictionary with app_id as key and app_name as value
app_df = app_df.set_index('app_id').to_dict()['app_name']
user_df = user_df.set_index('user_id').to_dict()['user_name']


# best RMSE score
print('RSME:', gs.best_score["rmse"])

# combination of parameters that gave the best RMSE score
print('Params:', gs.best_params["rmse"])

# Print the recommended items for each user
for uid, user_ratings in top_n.items():
    u = "{:<15} ".format(user_df[int(uid)])
    print(u, [app_df[int(iid)] for (iid, _) in user_ratings], ',', sep='')

'''
2.397637048187614
{'n_epochs': 500, 'lr_all': 0.0005, 'reg_all': 0.25}
ariaecd        ['cr_batches.app', 'gift_entry.app', 'cd_batches.app']
cwotg          ['cd_batches.app', 'disbursements.app', 'batches.app']
ebwfrre        ['cr_batches.app', 'gift_entry.app', 'disbursements.app']
eidohxy        ['partner_search.app', 'security_permissions.app', 'motd_edit.app']
fnnglumiip     ['cr_batches.app', 'gift_entry.app', 'disbursements.app']
hpoewnocnntmp  ['cd_batches.app', 'disbursements.app', 'gift_entry_new.app']
iblingai       ['gift_entry.app', 'cd_batches.app', 'disbursements.app']
litawhe        ['cr_batches.app', 'gift_entry.app', 'disbursements.app']
nhw            ['partner_window.app', 'cr_batches.app', 'disbursements.app']
nuyeno         ['cr_batches.app', 'gift_entry.app', 'cd_batches.app']
paau           ['gift_entry_new.app', 'partner_search.app', 'security_permissions.app']
sfhemodp       ['cd_batches.app', 'disbursements.app', 'gift_import.app']
sgacsbo        ['cd_batches.app', 'disbursements.app', 'batches.app']
smklfop        ['partner_search.app', 'Receipting.app']
tadeovbb       ['cd_batches.app', 'disbursements.app', 'gift_entry_new.app']
tqdwtr         ['partner_window.app', 'cr_batches.app', 'gift_entry.app']

2.3936304769760732
{'n_epochs': 400, 'lr_all': 0.001, 'reg_all': 0.001}
ariaecd:        ['cr_batches.app', 'gift_entry.app', 'cd_batches.app'],
cwotg:          ['cd_batches.app', 'disbursements.app', 'batches.app'],
ebwfrre:        ['cr_batches.app', 'gift_entry.app', 'disbursements.app'],
eidohxy:        ['TableMaintenance.app', 'data_qa.app', 'partner_search.app'],
fnnglumiip:     ['cr_batches.app', 'gift_entry.app', 'disbursements.app'],
hpoewnocnntmp:  ['cd_batches.app', 'disbursements.app', 'gift_entry_new.app'],
iblingai:       ['gift_import.app', 'gift_entry.app', 'cd_batches.app'],
litawhe:        ['cr_batches.app', 'gift_entry.app', 'disbursements.app'],
nhw:            ['partner_window.app', 'cr_batches.app', 'cd_batches.app'],
nuyeno:         ['cr_batches.app', 'gift_entry.app', 'disbursements.app'],
paau:           ['gift_entry_new.app', 'partner_search.app', 'data_qa.app'],
sfhemodp:       ['cd_batches.app', 'disbursements.app', 'gift_import.app'],
sgacsbo:        ['cd_batches.app', 'disbursements.app', 'batches.app'],
smklfop:        ['partner_search.app', 'Receipting.app'],
tadeovbb:       ['cd_batches.app', 'disbursements.app', 'gift_entry_new.app']
tqdwtr:         ['partner_window.app', 'cr_batches.app', 'gift_entry.app']

2.3949103954659057
{'n_epochs': 800, 'lr_all': 5e-05, 'reg_all': 5e-07}
ariaecd         ['cr_batches.app', 'gift_entry.app', 'cd_batches.app'],
cwotg           ['cd_batches.app', 'disbursements.app', 'batches.app'],
ebwfrre         ['cr_batches.app', 'gift_entry.app', 'disbursements.app'],
eidohxy         ['TableMaintenance.app', 'data_qa.app', 'motd_edit.app'],
fnnglumiip      ['cr_batches.app', 'gift_entry.app', 'disbursements.app'],
hpoewnocnntmp   ['cd_batches.app', 'disbursements.app', 'batches.app'],
iblingai        ['gift_entry.app', 'cd_batches.app', 'disbursements.app'],
litawhe         ['cr_batches.app', 'gift_entry.app', 'disbursements.app'],
nhw             ['partner_window.app', 'cr_batches.app', 'disbursements.app'],
nuyeno          ['cr_batches.app', 'gift_entry.app', 'cd_batches.app'],
paau            ['gift_entry_new.app', 'partner_search.app', 'data_qa.app'],
sfhemodp        ['gift_import.app', 'cd_batches.app', 'disbursements.app'],
sgacsbo         ['disbursements.app', 'cd_batches.app', 'batches.app'],
smklfop         ['partner_search.app', 'Receipting.app'],
tadeovbb        ['cd_batches.app', 'disbursements.app', 'gift_entry_new.app'],
tqdwtr          ['partner_window.app', 'cr_batches.app', 'gift_entry.app'],
'''

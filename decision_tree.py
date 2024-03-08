from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import csv

# import app access ta
df = pd.read_csv('kardia_data_five_years.csv', header=0)[:100]
df.drop_duplicates(inplace = True)

# import user ids
user_file = open('src/data_generation/data/user_ids.csv', 'r')
user_file_csv = csv.reader(user_file)
next(user_file_csv)

# import app ids
app_file = open('src/data_generation/data/app_ids.csv', 'r')
app_file_csv = csv.reader(app_file)
next(app_file_csv)

# make dictionary to translate user_names to ids
ids_to_users = dict(user_file_csv)
users_to_ids = dict()
for key, value in ids_to_users.items():
    users_to_ids.update({value : int(key)})

# make ascending-ordered list of app_paths
ids_to_apps = dict(app_file_csv)
apps = []
for app in ids_to_apps.values():
    apps.append(app)

# format dates
dates = df.access_date.to_list()
for i in range(len(dates)):
    timestamp = datetime.datetime.strptime(dates[i], '%Y-%m-%d %H:%M:%S').time()
    hour, minute, second = timestamp.hour, timestamp.minute, timestamp.second
    dates[i] = (hour * 3600) + (minute * 60) + second

# format users
users = df.user_name.to_list()
for i in range(len(users)):
    users[i] = users_to_ids.get(users[i])

# prepare dates and users to pass into decision tree
X = [list(a) for a in zip(dates, users)]
y = df.app_path.to_list()

# generate decision tree
clf = tree.DecisionTreeClassifier()
clf.fit(X,y)
tree.plot_tree(
    clf, 
    node_ids=True, 
    rounded=True, 
    feature_names=['Timestamp', 'User'],
    class_names=apps,
    filled=True,
    fontsize=2
)

# display decision tree
plt.title('Kardia App Access Recommendation Training')
fig = plt.gcf()
fig.set_size_inches(32, 32)
fig.savefig('decision_tree.pdf', format='pdf')

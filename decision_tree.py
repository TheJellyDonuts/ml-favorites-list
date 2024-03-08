from sklearn.datasets import load_iris
from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import csv

# iris example
#%%
iris = load_iris()
X, y = iris.data, iris.target

# import and sanitize csv data
df = pd.read_csv('kardia_data_five_years.csv', header=0)[:100]
df.drop_duplicates(inplace = True)
file = open('src/data_generation/data/user_ids.csv', 'r')
f_csv = csv.reader(file)
next(f_csv)

uids = dict(f_csv)
user_ids = dict()
for key, value in uids.items():
    user_ids.update({value : int(key)})

# format data
dates = df.access_date.to_list()
for i in range(len(dates)):
    timestamp = datetime.datetime.strptime(dates[i], '%Y-%m-%d %H:%M:%S').time()
    hour, minute, second = timestamp.hour, timestamp.minute, timestamp.second
    dates[i] = (hour * 3600) + (minute * 60) + second
users = df.user_name.to_list()
# TODO: make user ids actually the right ones and not arbitrary
for i in range(len(users)):
    users[i] = user_ids.get(users[i])
X = [list(a) for a in zip(dates, users)]
y = df.app_path.to_list()

clf = tree.DecisionTreeClassifier()
clf.fit(X,y)
tree.plot_tree(clf, node_ids=True, rounded=True)
# plt.savefig('myimg.pdf', format="pdf")
plt.figure()
plt.show()

# %%

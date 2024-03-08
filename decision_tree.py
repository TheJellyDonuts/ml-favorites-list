from sklearn.datasets import load_iris
from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd
import datetime

# iris example
iris = load_iris()
X, y = iris.data, iris.target

# import and sanitize csv data
df = pd.read_csv('test_data/kardia-obfuserdata.csv', header=0)
df.drop_duplicates(inplace = True)

# format data
dates = df.access_date.to_list()
for i in range(len(dates)):
    dates[i] = datetime.datetime.strptime(dates[i], '%Y-%m-%d %H:%M:%S').timestamp()
users = df.user_name.to_list()
# TODO: make user ids actually the right ones and not arbitrary
for i in range(len(users)):
    users[i] = i
X = [list(a) for a in zip(dates, users)]
y = df.app_path.to_list()

clf = tree.DecisionTreeClassifier()
clf.fit(X,y)
tree.plot_tree(clf)
plt.show()

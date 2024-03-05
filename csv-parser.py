import csv
from datetime import datetime

class userdata():
    def __init__(self, name):
        self.times = []
        self.name = name
        
    def add_appopen(self, ao):
        dt = datetime.strptime(ao.dt, '%Y-%m-%d %H:%M:%S')
        self.times.append(dt)
    
    def __str__(self):
        ...
    

class appopen():
    def __init__(self, dt, user, app):
        self.dt = dt
        self.user = user
        self.app = app
        
users = {}
entries = []

with open('kardia-obfuserdata.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(spamreader, None)  # Skip header row
    for row in spamreader:
        entries.append(",".join(row))
        # print(row)
        row = [line.replace('"','') for line in row]
        ao = appopen(row[0], row[1], row[2])
        if ao.user not in users:
            users.update({ao.user: userdata(ao.user)})
        users[ao.user].add_appopen(ao)

entries = list(set(entries))
with open("userdata.csv", 'w') as f:
    f.write('\n'.join(entries))


print(users)
            

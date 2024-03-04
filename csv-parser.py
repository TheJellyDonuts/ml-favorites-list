import csv
import datetime

def to_datetime(day, time):
    ...

class userdata():
    def __init__(self):
        self.times = []
        
    def add_appopen(self, data):
        self.times.append(to_datetime(data[0], data[1]))
    

with open('kardia-obfuserdata.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        print(', '.join(row))
        

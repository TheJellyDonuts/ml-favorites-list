import pandas as pd
import json


df = pd.read_csv('kardia-obfuserdata.csv', header=0)
df.drop_duplicates(inplace = True)
df[['day', 'time']] = df['access_date'].str.split(' ', expand=True)
df = df.drop(columns=['access_date'])
grouped = df.groupby(df['day'])

day_tools = {}

for i, day in enumerate(pd.unique(df['day'])):
    current_group = grouped.get_group(day)
    last_app = None
    
    by_day = []
    for index, item in current_group.iterrows():
        if last_app != item['app_path']:
            by_day.append(item['app_path'][21:])
        last_app = item['app_path']
            
    print(day, by_day, '\n')
    day_tools.update({day : by_day})
        

with open('day_tools.json', 'w') as f:
    json.dump(day_tools, f)

'''
[cr_batches, gift_entry, cr_bathces, de_batches]
'''
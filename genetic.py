from itertools import chain
import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import json
import re
import csv
from counted_follow_sets import counted_follow_sets
STRING_LEN = 250


toolnums = {}
tools_by_day = {}
toolnums = {}

# load data from csv into our data structures
with open('kardia-obfuserdata.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    next(spamreader, None)  # Skip header row
    num = 0
    for row in spamreader:
        date, user, path = [line.replace('"','') for line in row]
        day = date[:date.index(' ')]
        tool = path

        if tool not in toolnums:
            toolnums.update({tool : num})
            toolnums.update({num : tool})
            num += 1
        
        if day not in tools_by_day:
            tools_by_day.update({day : []})

        tools_by_day[day].append(path)
        
        
# concatenate the item numbers by day
tools_by_day_list = []
full_num_list = []
for value in tools_by_day.values():
    if len(value) >= 1:
        temp = []
        for val in value:
            n = toolnums.get(val)
            temp.append(str(n))
            full_num_list.append(n)
        tools_by_day_list.append(' '.join(temp))

# fitness function to pass to the GA
def f(tool_num_list):
    # format the input that the GA gives
    # input is a list (not an array!) of numpy numbers
    joined_ary = np.array2string(np.asarray(tool_num_list, dtype=int))[1:-1]
    joined_ary = re.sub(r'\s+', r' ', joined_ary).strip()

    # algorithm minimizes score, but score is a num of counts (which we want to max),
    # so return the inverse of the score    
    return 1/(calc_score(joined_ary)+1)

# iterate past each day's tool num string and find all matches in the GA input 
def calc_score(joined_ary):
    score = 0
    for lst in tools_by_day_list:
        matches = re.findall(lst, joined_ary)
        score += len(matches)
    return score

# Set GA algorithm parameters
algorithm_param = {'max_num_iteration': 5000,
                   'population_size':100, # no touch
                   'mutation_probability':0.15, # 0.1
                   'elit_ratio': 0.01,
                   'crossover_probability': 0.4, # 0.5
                   'parents_portion': 0.3,
                   'crossover_type':'uniform',
                   'max_iteration_without_improv': None
                   }

varbound=np.array([[0,len(toolnums)/2-1]]*STRING_LEN)

model=ga(function=f,
         dimension=STRING_LEN,
         variable_type='int',
         algorithm_parameters=algorithm_param,
         variable_boundaries=varbound
         )


model.run()
solution=model.output_dict['variable']

# formatting to get results
app_lst = [toolnums.get(int(app)) for app in solution]
s = re.sub(r' +', r' ', re.sub(r'\.|\n', r'', np.array2string(solution)[1:-1])).strip()
score = calc_score(s)
print('Score:', score)

# write results to results.json
results = {}
results.update({'apps' : app_lst})
results.update({'nums' : s})
results.update({'score' : score})
with open('results2.json', 'w') as f:
    json.dump(results, f)

# take the solutions and create a dictionary that contains an ordered follow set of every app
solutions = [int(num) for num in s.split(' ')]
cfs = counted_follow_sets(solutions)

# output sorted dictionary to a json
with open("apps2.json", 'w') as af:
    json.dump(cfs.get_sets(), af)
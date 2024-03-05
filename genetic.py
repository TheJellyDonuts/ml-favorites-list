import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import json
import re
import csv
STRING_LEN = 250


toolnums = {}
tools_by_day = {}

toolnums = {}
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
for value in tools_by_day.values():
    if len(value) >= 1:
        temp = []
        for val in value:
            temp.append(str(toolnums.get(val)))
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
algorithm_param = {'max_num_iteration': 50,
                   'population_size':100, # no touch
                   'mutation_probability':0.15, # 0.1
                   'elit_ratio': 0.01,
                   'crossover_probability': 0.25, # 0.5
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
with open('results.json', 'w') as f:
    json.dump(results, f)

# take the solutions and create a dictionary that contains an ordered follow set of every app
solutions = [int(num) for num in s.split(' ')]
app_dict = dict()
for app_num, next_app_num in zip(solutions[:-1], solutions[1:]):
    if (curr_app_list := app_dict.get(app_num)) is None:
        app_dict.update({app_num : dict({next_app_num : 1})})
    else:
        if (next_app_count := curr_app_list.get(next_app_num)) is None:
            curr_app_list.update({next_app_num : 1})
        else:
            curr_app_list.update({next_app_num : next_app_count+1})

# sort the dictionary values by the frequency of apps within any app's given follow set
out_dict = {}
for key, value in app_dict.items():
    sorted_vals = dict(sorted(value.items(), key=lambda item: item[1], reverse=True))
    out_dict.update({key : sorted_vals})

# output sorted dictionary to a json
with open("apps.json", 'w') as af:
    json.dump(out_dict, af)
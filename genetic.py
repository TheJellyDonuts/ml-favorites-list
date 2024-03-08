import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import json
import re
import csv
from counted_follow_sets import counted_follow_sets
import os
from time import time

STRING_LEN = 50
FILENAME = 'kardia_data_five_years.csv'
GENERATIONS = 1000

toolnums = dict()
users = dict()

# load data from csv into our data structures
with open(FILENAME) as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # Skip header row
    num = 0
    for row in reader:
        date, user, tool = row
        day = date[:date.index(' ')]    # Ignore the time

        # Add tools to bidirectional dictionary
        # Allows for conversion between number and name, and vice versa
        if tool not in toolnums:
            toolnums.update({tool : num})
            toolnums.update({num : tool})
            num += 1
        
        # Add users to a dictionary that contains another dictionary that pairs dates with a list of apps opened on that date
        # Example entry: {john_doe : {'2023-08-02' : ['google', 'spotify', 'stellarium']}}
        if (curr_user := users.get(user)) is None:
            users.update({user : dict()})
            curr_user = users.get(user)
        if (curr_user_day := curr_user.get(day)) is None:
            curr_user.update({day : []})
            curr_user_day = curr_user.get(day)
        new_tool_list = curr_user_day.append(tool)  # This object is still connected to the dictionary, so this works


# concatenate the item numbers by user and by day
tools_by_user_dict = dict()
for user, date in users.items():
    tools_by_day_list = []
    for tools in date.values():
        if len(tools) > 2:
            curr = f''
            for tool in tools:
                curr = f'{curr}{str(toolnums.get(tool))} '
            tools_by_day_list.append(curr[:-1])
    tools_by_user_dict.update({user: tools_by_day_list})

# Since tools_by_user_dict entry contains a list of tools_by_day for each user, we want a custom function that only gets ...
# tools for each user so that the GA can run on a user-by-user basis. This function creates a function that uses a particular...
# element of the tools_by_user_dict list (again, each element represents a user). The for-loop below creates as many of these...
# functions as there are users. Later when the GA is run, it is run for each user.
def fitness_creator(user):
    def f(tool_num_list):
        return abstract_f(tool_num_list, tools_by_user_dict.get(user))
    return f
user_funcs = dict()
for user in users.keys():
    user_funcs.update({user : fitness_creator(user)})


# Abstract fitness function that will be used by the GA inside of the custom functions
def abstract_f(tool_num_list, user_tools_by_day):
    # format the input that the GA gives
    # input is a list (not an array!) of numpy numbers
    joined_ary = np.array2string(np.asarray(tool_num_list, dtype=int))[1:-1]
    joined_ary = re.sub(r'\s+', r' ', joined_ary).strip()

    # algorithm minimizes score, but score is a num of counts (which we want to max),
    # so return the inverse of the score    
    return 1/(calc_score(joined_ary, user_tools_by_day)+1)

# iterate past each day's tool num string and find all matches in the GA input 
def calc_score(joined_ary, user_tools_by_day):
    score = 0
    for lst in user_tools_by_day:
        matches = re.findall(lst, joined_ary)
        score += len(matches)
    return score


# Set GA algorithm parameters
algorithm_param = {'max_num_iteration': GENERATIONS,
                   'population_size':100, # no touch
                   'mutation_probability':0.15, # 0.1
                   'elit_ratio': 0.01,
                   'crossover_probability': 0.4, # 0.5
                   'parents_portion': 0.3,
                   'crossover_type':'uniform',
                   'max_iteration_without_improv': None
                   }

# define the boundaries of the toolnums that the GA values take
varbound=np.array([[0,len(toolnums)/2-1]]*STRING_LEN)

# Create unique directory to store all the results
curr_time = round(time())
os.mkdir(f'{curr_time}')

user_scores = []

# Run the model for as many users as there are (this will be tracked accurately by the user_funcs dictionary)
for user, func in user_funcs.items():
    model=ga(function=func,
            dimension=STRING_LEN,
            variable_type='int',
            algorithm_parameters=algorithm_param,
            variable_boundaries=varbound
            )

    model.run()
    print(f'Running GA model for user {user}')

    solution=model.output_dict['variable']

    # formatting to get results
    app_lst = [toolnums.get(int(app)) for app in solution]
    s = re.sub(r' +', r' ', re.sub(r'\.|\n', r'', np.array2string(solution)[1:-1])).strip()
    score = calc_score(s, tools_by_user_dict.get(user))
    print(f'Score for {user}:', score)
    user_scores.append([user, score])

    # write results
    results = {}
    results.update({'apps' : app_lst})
    results.update({'nums' : s})
    results.update({'score' : score})
    with open(f'{curr_time}/results_{user}.json', 'w') as f:
        json.dump(results, f)

    # take the solutions and create a dictionary that contains an ordered follow set of every app
    solutions = [int(num) for num in s.split(' ')]
    cfs = counted_follow_sets(solutions, toolnums)

    # output sorted dictionary to a json
    with open(f'{curr_time}/apps_{user}.json', 'w') as af:
        json.dump(cfs.get_sets(), af)

for user, score in user_scores:
    print(f'Score for {user}: {score}')

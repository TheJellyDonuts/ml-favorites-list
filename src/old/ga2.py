# genetic.py and counted_follow_sets.py do what was trying to be done here but better
# this file is preserved for posterity. Could be helpful if GAs need to be relooked at

import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import json
import re

string_len = 250

num_to_tool = {}
with open('enum_tools.json', 'r') as f:
    num_to_tool = json.load(f)
    
tool_to_num = {v: k for k, v in num_to_tool.items()}
    
day_tools = {}
with open('day_tools.json', 'r') as f:
    day_tools = json.load(f)

tool_strings = []
for value in day_tools.values():
    if len(value) >= 1:
        temp = []
        for val in value:
            temp.append(tool_to_num.get(val))
        tool_strings.append(temp)
    

def f(tool_num_list):
    joined_ary = np.array2string(np.asarray(tool_num_list, dtype=int))[1:-1]
    joined_ary = joined_ary.replace(r' +', ' ')
    score = 0
    for lst in tool_strings:
        lll = ' '.join(lst)
        ll = re.findall(lll, joined_ary)
        score += len(ll)

    # if score != 0:
    #     print(score)
    return 1/(score+1)

def f2(joined_ary):
    score = 0
    for lst in tool_strings:
        lll = ' '.join(lst)
        ll = re.findall(lll, joined_ary)
        score += len(ll)

    return score



varbound=np.array([[0,len(num_to_tool)-1]]*string_len)

algorithm_param = {'max_num_iteration': 5000,
                   'population_size':100, # no touch
                   'mutation_probability':0.1,
                   'elit_ratio': 0.01,
                   'crossover_probability': 0.5,
                   'parents_portion': 0.3,
                   'crossover_type':'uniform',
                   'max_iteration_without_improv': None
                   }

model=ga(function=f,
         dimension=string_len,
         variable_type='int',
         algorithm_parameters=algorithm_param,
         variable_boundaries=varbound
         )


model.run()
convergence=model.report
solution=model.output_dict
# print(convergence)
# print(solution)
with open('result.txt', 'w') as f:
    app_lst = [num_to_tool[str(int(app))] for app in solution['variable']]
    s = np.array2string(solution['variable'], max_line_width=5000000)[1:-1].replace('.', '').replace('  ', ' ')
    score = f2(s)
    print('Score:', score, app_lst)
    f.write(' '.join(app_lst))
    f.write("\n")
    f.write(s)
    f.write(f"\nScore: {str(score)}")
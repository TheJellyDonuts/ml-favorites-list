import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import json
import re

string_len = 25

num_to_tool = {}
with open('enum_tools.json', 'r') as f:
    num_to_tool = json.load(f)
    
tool_to_num = {v: k for k, v in num_to_tool.items()}
    
day_tools = {}
with open('day_tools.json', 'r') as f:
    day_tools = json.load(f)

tool_strings = []
for value in day_tools.values():
    if len(value) >= 10:
        temp = []
        for val in value:
            temp.append(tool_to_num.get(val))
        tool_strings.append(temp)
    
print(tool_strings[0:3])

def f(tool_num_list):
    score = 0
    joined_ary = " ".join(tool_num_list)
    for lst in tool_strings:
        score += len(re.findall(joined_ary, ' '.join(lst)))
    
    return 1/(score+1)

z= tool_strings[0]
print(f(z))


x= 1/0


varbound=np.array([[0,len(num_to_tool)]]*string_len)

algorithm_param = {'max_num_iteration': 1000,
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
print(convergence)
print(solution)
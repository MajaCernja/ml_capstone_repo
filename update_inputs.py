import numpy as np
from utils import add_to_npy_file


queries = []

print(np.load('initial_data/function_3/initial_inputs.npy'))

i = 1
for arr in queries:

    add_to_npy_file('initial_data/function_'+str(i)+'/initial_inputs.npy', arr)
    i += 1

print(np.load('initial_data/function_3/initial_inputs.npy'))
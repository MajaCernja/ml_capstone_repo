import numpy as np
from utils import add_to_npy_file


ys = []

print(np.load('initial_data/function_3/initial_outputs.npy'))

i = 1
for y in ys:

    arr = np.array([y])
    add_to_npy_file('initial_data/function_'+str(i)+'/initial_outputs.npy', arr)
    i += 1

print(np.load('initial_data/function_3/initial_outputs.npy'))
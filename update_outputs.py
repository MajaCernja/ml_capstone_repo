import numpy as np
from utils import add_to_npy_file


ys = [-6.515538361770915e-06, 0.012265170460017318, -0.0492450559024272, -9.892785243638617, 237.8040385524087, -1.5951343763757544, 0.12138886755506177, 7.8589485009024]

print(np.load('initial_data/function_3/initial_outputs.npy'))

i = 1
for y in ys:

    arr = np.array([y])
    add_to_npy_file('initial_data/function_'+str(i)+'/initial_outputs.npy', arr)
    i += 1

print(np.load('initial_data/function_3/initial_outputs.npy'))
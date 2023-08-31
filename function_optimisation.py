import numpy as np
from bayes_optimisation_class import BayesianOptimizer

for i in range(1,9):
    X = np.load('initial_data/function_'+str(i)+'/initial_inputs.npy')
    Y = np.load('initial_data/function_'+str(i)+'/initial_outputs.npy')

    optimizer = BayesianOptimizer(X, Y)
    next_query = optimizer.get_next_probable_point()
    print('Function ' + str(i) + ' next query: ', next_query)

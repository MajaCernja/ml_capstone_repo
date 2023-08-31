from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm
from scipy.optimize import minimize
import numpy as np


class BayesianOptimizer:

    def __init__(self, x_init, y_init, scale=0.1):
        self.x_init = x_init
        self.y_init = y_init
        self.scale = scale  # in case of experimenting with other acquisition functions
        self.gauss_pr = GaussianProcessRegressor()
        self.gauss_pr.fit(self.x_init, self.y_init)

    def get_expected_improvement(self, x_new):
        # Using estimate from Gaussian surrogate instead of actual function for
        # a new trial data point

        mean_y_new, sigma_y_new = self.gauss_pr.predict(np.array([x_new]), return_std=True)
        sigma_y_new = sigma_y_new.reshape(-1, 1)
        if sigma_y_new == 0.0:
            return 0.0

        # Using estimates from Gaussian surrogate instead of actual function for
        # entire prior distribution

        mean_y = self.gauss_pr.predict(self.x_init)
        max_mean_y = np.max(mean_y)
        z = (mean_y_new - max_mean_y) / sigma_y_new
        exp_imp = (mean_y_new - max_mean_y) * norm.cdf(z) + sigma_y_new * norm.pdf(z)

        return exp_imp

    def get_next_probable_point(self):

        result = minimize(
            fun=lambda x: -self.get_expected_improvement(x),
            x0=np.random.random(len(self.x_init[0])),
            bounds=[(0, 1)] * len(self.x_init[0]),
            method='L-BFGS-B'
        )
        optimal_query = result.x

        # formatting output to adhere to query submission format requirements
        stringified_query = ''
        for x in optimal_query:
            if x == optimal_query[-1]:
                stringified_query += f'{x:.6f}'
            else:
                stringified_query += f'{x:.6f}' + '-'

        return stringified_query

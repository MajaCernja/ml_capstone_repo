import sys
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


def add_to_npy_file(file_name, data):
    existing_data = np.load(file_name)

    if len(data.shape) < len(existing_data.shape):
        # Reshape data to match the number of dimensions in existing_data
        data = np.expand_dims(data, axis=0)

    new_data = np.concatenate((existing_data, data), axis=0)
    np.save(file_name, new_data)


def remove_last_entries_from_npy(file_path, num_entries):
    try:
        # Load the original data from the .npy file
        data = np.load(file_path)

        # Remove the last `num_entries` entries
        new_data = data[:-num_entries]

        # Save the modified data back to the .npy file
        np.save(file_path, new_data)
        print(f"Last {num_entries} entries removed from {file_path}")

    except Exception as e:
        print("Error:", e)


def expected_improvement(x, model, current_best):
    mean, std = model.predict(x, return_std=True)
    z = (mean - current_best) / std
    return (mean - current_best) * norm.cdf(z) - std * norm.pdf(z)


def ei_acquisition_function(x, model, current_best):
    return -expected_improvement(x, model, current_best)


def get_next_probable_point(batch_size, x_init, model,current_best):
    min_ei = float(sys.maxsize)
    x_optimal = None

    for x_start in (np.random.random((batch_size, x_init.shape[1]))):
        response = minimize(fun=ei_acquisition_function, x0=x_start, args=(model, current_best))
        if response.fun[0] < min_ei:
            min_ei = response.fun[0]
            x_optimal = response.x

    return round(x_optimal, 6), min_ei



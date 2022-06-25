import os

import numpy as np

from mlpy.lib.utils.path import makedirs

DIR_DATA = './TimeGANSine'


def sine_data_generation(no, seq_len, dim):
    """ copy from: https://github.com/jsyoon0823/TimeGAN/blob/master/data_loading.py
    Sine data generation.

  Args:
    - no: the number of samples
    - seq_len: sequence length of the time-series
    - dim: feature dimensions

  Returns:
    - data: generated data
  """
    # Initialize the output
    data = list()

    # Generate sine data
    for i in range(no):
        # Initialize each time-series
        temp = list()
        # For each feature
        for k in range(dim):
            # Randomly drawn frequency and phase
            freq = np.random.uniform(0, 0.1)
            phase = np.random.uniform(0, 0.1)

            # Generate sine signal based on the drawn frequency and phase
            temp_data = [np.sin(freq * j + phase) for j in range(seq_len)]
            temp.append(temp_data)

        # Align row/column
        temp = np.transpose(np.asarray(temp))
        # Normalize to [0,1]
        temp = (temp + 1) * 0.5
        # Stack the generated data
        data.append(temp)

    return data


def sine_dim5_len24_random(r=0):  # the primitive data
    seq_len = 24
    no, dim = 10000, 5
    data = sine_data_generation(no, seq_len, dim)
    data = np.stack(data)
    np.save(os.path.join(DIR_DATA, f'sine_dim{dim}_len{seq_len}_r{r}'), data)


def sine_dim1_len100_random(r=0):
    seq_len = 100
    no, dim = 10000, 1
    data = sine_data_generation(no, seq_len, dim)
    data = np.stack(data)
    np.save(os.path.join(DIR_DATA, f'sine_dim{dim}_len{seq_len}_r{r}'), data)


if __name__ == '__main__':
    makedirs(DIR_DATA)

    # Prepare datasets for for multiple random runs.
    for r in range(5):
        sine_dim5_len24_random(r)

    for r in range(5):
        sine_dim1_len100_random(r)


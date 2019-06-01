import os
import numpy as np
from numba import jit


class Logger:
    def __init__(self, path):
        self.path = path
        if path != '':
            folder = '/'.join(path.split('/')[:-1])
            if not os.path.exists(folder):
                os.makedirs(folder)

    def print(self, message):
        print(message)
        if self.path != '':
            with open(self.path, 'a') as f:
                f.write(message + '\n')
                f.flush()


def get_contiguous_indices(indices_opt_init):
    # needed when indices_opt_init are not contiguous (e.g. mnist_2_6: coord=613 has [12, 13, 15, 16, 18])
    running_diffs = indices_opt_init[1:] - indices_opt_init[:-1]  # [1, 2, 1, 2]
    where_change_contiguous_regions = np.where(running_diffs != 1)[0]  # find the last contiguous index - 1

    if len(where_change_contiguous_regions) > 0:
        last_el_first_contiguous_region = where_change_contiguous_regions[0]
    elif np.sum(running_diffs != np.ones(len(running_diffs))) == 0:  # if all are optimal (mnist_2_6: coord=72)
        last_el_first_contiguous_region = len(running_diffs)
    elif len(indices_opt_init) == 1:  # the easiest and most common situation - just 1 optimal index
        last_el_first_contiguous_region = 0
    else:
        raise Exception('this case has not been handled')
    indices_opt = indices_opt_init[:last_el_first_contiguous_region + 1]  # [12, 13]
    return indices_opt


@jit(nopython=True)
def minimum(arr1, arr2):
    # take element-wise minimum of 2 arrays compatible with numba (instead of np.minimum(arr1, arr2))
    return arr1 * (arr1 < arr2) + arr2 * (arr1 >= arr2)


@jit(nopython=True)
def clip(val, val_min, val_max):
    # identical to np.clip
    return min(max(val, val_min), val_max)


def print_arr(arr):
    """ Pretty printing of a 2D numpy array. """
    for i, row in enumerate(arr):
        string = ''
        for el in row:
            string += '{:.3f} '.format(el)
        print(i+1, string)

import os
import numpy as np
import glob
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


def extract_hyperparam(model_name, substr):
    return model_name.split(substr)[1].split(' ')[0]


def finalize_curr_row(latex_str, weak_learner, flag_n_trees_latex):
    # finalizing the current row: apply boldfacing and add \\
    # (relies on the fact that we have only 3 metrics, i.e. TE,RTE,URTE or TE,LRTE,URTE or 4 metrics if flag_n_trees_latex is on)
    # result: 'breast-cancer & 0.3 & 0.7 & 85.4 & 85.4 & 5.1 & 11.7 & 11.7 & 5.1 & 11.7 & 11.7'
    curr_row = latex_str.split(r'\\')[-1]
    curr_str_bf = ' & '.join(curr_row.split(' & ')[:2]) + ' &   '
    metrics_str = ' & '.join(curr_row.split(' & ')[2:])
    n_metrics = 4 if flag_n_trees_latex else 3
    metrics_curr_row = dict([(i, []) for i in range(n_metrics)])
    # result: {0: [0.7, 5.1, 5.1], 1: [85.4, 11.7, 11.7], 2: [85.4, 11.7, 11.7]}
    for i_val, val_str in enumerate(metrics_str.split(' & ')):
        # for n_trees we need int, for the rest float
        val = int(val_str) if flag_n_trees_latex and i_val % n_metrics == n_metrics - 1 else float(val_str)
        metrics_curr_row[i_val % n_metrics].append(val)
    # form the boldfaced str that corresponds to the current row
    for tup in zip(*metrics_curr_row.values()):
        for i_m, m in enumerate(tup):
            # boldfacing condition: if minimum and it's not the number of trees (if the flag is turned on)
            if (m == min(metrics_curr_row[i_m]) and not (flag_n_trees_latex and i_m == 3) and
                    not (weak_learner == 'stump' and i_m == 2)):  # if URTE for stumps, don't boldface
                curr_str_bf += '\\textbf{' + str(m) + '} & '
            else:
                curr_str_bf += '{} & '.format(m)
        curr_str_bf += '  '  # just a margin for better latex code quality
    curr_str_bf = curr_str_bf.strip()[:-1]  # get rid of the last ' &   '
    curr_row_final = curr_str_bf + r'\\' + '\n'  # new table line
    return curr_row_final


def get_model_names(datasets, models, exp_folder, weak_learner, tree_depth):
    model_names = []
    for dataset in datasets:
        for model in models:
            depth_str = 'max_depth=' + str(tree_depth) if weak_learner == 'tree' else ''
            search_str = '{}/*dataset={} weak_learner={} model={}*{}*.metrics'.format(
                exp_folder, dataset, weak_learner, model, depth_str)
            model_names_curr = glob.glob(search_str)
            model_names_curr.sort(key=lambda x: os.path.getmtime(x))
            if model_names_curr != []:
                # model_name_final = model_names_curr[-1]
                for model_name_final in model_names_curr:
                    model_name_final = model_name_final.split('.metrics')[0].split(exp_folder+'/')[1]
                    model_names.append(model_name_final)
    return model_names


def get_n_proc(n_ex):
    if n_ex > 40000:
        n_proc = 50
    elif n_ex > 20000:
        n_proc = 40
    elif n_ex > 2500:
        n_proc = 25
    elif n_ex > 1000:
        n_proc = 10
    elif n_ex > 200:
        n_proc = 5
    else:
        n_proc = 1
    return n_proc

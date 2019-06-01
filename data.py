import numpy as np
import csv
import scipy.io
import ipdb as pdb
from tensorflow.keras.datasets import mnist, fashion_mnist


def split_train_test(X_all, y_all, frac_train):
    """
    The first X% of X_all, y_all become the training set, the rest (1-X)% become the test set.
    Note that this assumes that the samples are already shuffled or if not (e.g. if we were to split MNIST) that
    this behavior is intended.
    """
    num_total = X_all.shape[0]
    num_train = int(frac_train * num_total)

    X_train, y_train = X_all[:num_train], y_all[:num_train]
    X_test, y_test = X_all[num_train:], y_all[num_train:]

    return X_train, y_train, X_test, y_test


def normalize_per_feature_0_1(X_train, X_test):
    """
    We are not allowed to touch the test data, thus we do the normalization just based on the training data.
    """
    X_train_max = X_train.max(axis=0, keepdims=True)
    X_train_min = X_train.min(axis=0, keepdims=True)
    X_train = (X_train - X_train_min) / (X_train_max - X_train_min)
    X_test = (X_test - X_train_min) / (X_train_max - X_train_min)
    return X_train, X_test


def split_train_validation(X_train_orig, y_train_orig, shuffle=True):
    num_total = X_train_orig.shape[0]
    frac_train = 0.8
    n_train = int(frac_train*num_total)
    idx = np.random.permutation(num_total) if shuffle else np.arange(num_total)
    X_train, y_train = X_train_orig[idx][:n_train], y_train_orig[idx][:n_train]
    X_valid, y_valid = X_train_orig[idx][n_train:], y_train_orig[idx][n_train:]
    return X_train, y_train, X_valid, y_valid


def binary_from_multiclass(X_train, y_train, X_test, y_test, classes):
    classes = np.array(classes)  # for indexing only arrays work, not lists

    idx_train1, idx_train2 = y_train == classes[0], y_train == classes[1]
    idx_test1, idx_test2 = y_test == classes[0], y_test == classes[1]
    X_train, X_test = X_train[idx_train1 + idx_train2], X_test[idx_test1 + idx_test2]

    y_train = idx_train1 * 1 + idx_train2 * -1
    y_test = idx_test1 * 1 + idx_test2 * -1
    y_train, y_test = y_train[idx_train1 + idx_train2], y_test[idx_test1 + idx_test2]

    return X_train, y_train, X_test, y_test


def toy_2d_stumps():
    X = np.array([[0.38, 0.75], [0.50, 0.93], [0.05, 0.70], [0.30, 0.90], [0.15, 0.80],
                  # [0.15, 1.0], [0.125, 0.75], [0.1, 0.85], [0.045, 0.22], [0.725, 0.955],  # small margin
                  # [0.15, 1.0], [0.125, 0.75], [0.1, 0.85], [0.075, 0.2], [0.775, 0.925],  # small margin
                  [0.15, 1.0], [0.125, 0.5], [0.1, 0.85], [0.02, 0.25], [0.775, 0.975],
                  [0.05, 0.05], [0.2, 0.1], [0.4, 0.075], [0.6, 0.22], [0.8, 0.1],
                  [0.95, 0.05], [0.9, 0.2], [0.925, 0.4], [0.79, 0.6], [0.81, 0.8]])
    y = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    eps_dataset = 0.075
    return X, y, eps_dataset


def toy_2d_trees():
    X = np.array([[0.38, 0.75], [0.50, 0.93], [0.05, 0.70], [0.30, 0.90], [0.15, 0.80],
                  [0.75, 0.38], [0.95, 0.48], [0.70, 0.05], [0.65, 0.30], [0.80, 0.30],
                  [0.05, 0.1], [0.35, 0.1], [0.45, 0.075], [0.3, 0.2], [0.25, 0.1],
                  [0.95, 0.65], [0.7, 0.9], [0.925, 0.7], [0.79, 0.55], [0.81, 0.8]])
    y = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    eps_dataset = 0.075
    return X, y, eps_dataset


def toy_2d_xor():
    X = np.array([[0.05, 0.05], [0.95, 0.95], [0.05, 0.95], [0.95, 0.05]])
    y = np.array([-1, -1, 1, 1])
    eps_dataset = 0.15
    return X, y, eps_dataset


def toy_2d_wong():
    # random points at least 2r apart
    m = 12
    # seed=10 illustrates that by default the margin can be easily close to 0
    # both plain and robust model have 0 train error, but the robust model additionally enforces a large margin
    np.random.seed(10)
    x = [np.random.uniform(size=2)]
    r = 0.16
    while len(x) < m:
        p = np.random.uniform(size=2)
        if min(np.abs(p - a).sum() for a in x) > 2 * r:
            x.append(p)
    eps_dataset = r / 2

    X = np.array(x)
    y = np.sign(np.random.uniform(-0.5, 0.5, size=m))
    return X, y, eps_dataset


def breast_cancer():
    """
    Taken from the UCI repository:
    http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28diagnostic%29 file: breast-cancer-wisconsin.data

    After filtering the points with missing data, we have exactly the same as Chen et al, 2019
    train: 546x10, test: 137x10
    """
    eps_dataset = 0.3  # same as in Chen et al, 2019, worked well for them
    path = 'data/breast_cancer/breast-cancer-wisconsin.data'

    lst = []
    for line in csv.reader(open(path, 'r').readlines()):
        if '?' not in line:
            lst.append(line)
    data_arr = np.array(lst, dtype=int)

    X_all, y_all = data_arr[:, :10], data_arr[:, 10]
    y_all[y_all == 2], y_all[y_all == 4] = -1, 1  # from 2, 4 to -1, 1

    X_train, y_train, X_test, y_test = split_train_test(X_all, y_all, frac_train=0.8)
    X_train, X_test = normalize_per_feature_0_1(X_train, X_test)

    return X_train, y_train, X_test, y_test, eps_dataset


def diabetes():
    """
    Taken from Kaggle:
    https://www.kaggle.com/uciml/pima-indians-diabetes-database file: diabetes.csv

    train: 614x8, test: 154x8
    """
    eps_dataset = 0.05  # Chen et al, 2019 used 0.2, but it was too high
    path = 'data/diabetes/diabetes.csv'
    data_arr = np.loadtxt(path, delimiter=',', skiprows=1)  # loaded as float64

    X_all, y_all = data_arr[:, :8], data_arr[:, 8]
    y_all[y_all == 0], y_all[y_all == 1] = -1, 1  # from 0, 1 to -1, 1

    X_train, y_train, X_test, y_test = split_train_test(X_all, y_all, frac_train=0.8)
    X_train, X_test = normalize_per_feature_0_1(X_train, X_test)

    return X_train, y_train, X_test, y_test, eps_dataset


def ijcnn1():
    """
    Taken from LIBSVM data repository:
    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html

    train: 49990x22, test: 91701x22
    note: imbalanced classes (-1: 90.3% vs 1: 9.7%)
    """
    eps_dataset = 0.05  # Chen et al, 2019 used 0.1, but it was too high
    folder = 'data/ijcnn1/'
    path_train, path_val, path_test = folder + 'ijcnn1.tr', folder + 'ijcnn1.val', folder + 'ijcnn1.t'

    num_train, num_test, dim = 49990, 91701, 22
    X_train = np.zeros((num_train, dim))
    y_train = np.zeros(num_train)
    num_train_orig = 0
    for i, line in enumerate(open(path_train, 'r').readlines()):
        y_train[i] = int(float(line.split(' ')[0]))  # -1 or 1
        for s in line.split(' ')[1:]:
            coord_str, val_str = s.replace('\n', '').split(':')
            coord, val = int(coord_str) - 1, float(val_str)  # -1 is needed to have pythonic numeration from 0
            X_train[i, coord] = val
        num_train_orig += 1

    num_val_orig = 0
    for i, line in enumerate(open(path_val, 'r').readlines()):
        y_train[num_train_orig + i] = int(float(line.split(' ')[0]))  # -1 or 1
        for s in line.split(' ')[1:]:
            coord_str, val_str = s.replace('\n', '').split(':')
            coord, val = int(coord_str) - 1, float(val_str)
            X_train[num_train_orig + i, coord] = val
        num_val_orig += 1

    assert num_train == num_train_orig + num_val_orig  # Check that we have not introduced extra zero rows

    X_test = np.zeros((num_test, dim))
    y_test = np.zeros(num_test)
    num_test_orig = 0
    for i, line in enumerate(open(path_test, 'r').readlines()):
        y_test[i] = int(float(line.split(' ')[0]))  # -1 or 1
        for s in line.split(' ')[1:]:
            coord_str, val_str = s.replace('\n', '').split(':')
            coord, val = int(coord_str) - 1, float(val_str)
            X_test[i, coord] = val
        num_test_orig += 1

    assert num_test == num_test_orig

    X_train, X_test = normalize_per_feature_0_1(X_train, X_test)

    return X_train, y_train, X_test, y_test, eps_dataset


def cod_rna():
    """
    Taken from LIBSVM data repository:
    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html

    train: 59535x8, test: 271617x8
    """
    eps_dataset = 0.025  # Chen et al, 2019 used 0.2, but it was too high
    folder = 'data/cod_rna/'
    path_train, path_test = folder + 'cod-rna.tr', folder + 'cod-rna.t'

    num_train, num_test, dim = 59535, 271617, 8
    X_train = np.zeros((num_train, dim))
    y_train = np.zeros(num_train)
    num_train_orig = 0
    for i, line in enumerate(open(path_train, 'r').readlines()):
        y_train[i] = int(float(line.split(' ')[0]))  # -1 or 1
        for s in line.split(' ')[1:]:
            coord_str, val_str = s.replace('\n', '').split(':')
            coord, val = int(coord_str) - 1, float(val_str)  # -1 is needed to have pythonic numeration from 0
            X_train[i, coord] = val
        num_train_orig += 1

    assert num_train == num_train_orig  # Check that we have not introduced extra zero rows

    X_test = np.zeros((num_test, dim))
    y_test = np.zeros(num_test)
    num_test_orig = 0
    for i, line in enumerate(open(path_test, 'r').readlines()):
        y_test[i] = int(float(line.split(' ')[0]))  # -1 or 1
        for s in line.split(' ')[1:]:
            coord_str, val_str = s.replace('\n', '').split(':')
            coord, val = int(coord_str) - 1, float(val_str)
            X_test[i, coord] = val
        num_test_orig += 1

    assert num_test == num_test_orig

    X_train, X_test = normalize_per_feature_0_1(X_train, X_test)

    return X_train, y_train, X_test, y_test, eps_dataset


def mnist_2_6():
    """
    train: (11876, 784), test: (1990, 784)
    """
    eps_dataset = 0.3
    classes = [2, 6]  # 2 is 1, 6 is -1 in the binary classification scheme

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train, X_test = X_train.astype(np.float64) / 255.0, X_test.astype(np.float64) / 255.0
    X_train = np.reshape(X_train, [X_train.shape[0], -1])
    X_test = np.reshape(X_test, [X_test.shape[0], -1])

    X_train, y_train, X_test, y_test = binary_from_multiclass(X_train, y_train, X_test, y_test, classes)
    return X_train, y_train, X_test, y_test, eps_dataset


def mnist_1_5():
    """
    train: (11876, 784), test: (1990, 784)
    """
    eps_dataset = 0.3
    classes = [1, 5]  # 2 is 1, 6 is -1 in the binary classification scheme

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train, X_test = X_train.astype(np.float64) / 255.0, X_test.astype(np.float64) / 255.0
    X_train = np.reshape(X_train, [X_train.shape[0], -1])
    X_test = np.reshape(X_test, [X_test.shape[0], -1])

    X_train, y_train, X_test, y_test = binary_from_multiclass(X_train, y_train, X_test, y_test, classes)
    return X_train, y_train, X_test, y_test, eps_dataset


def fmnist_sandal_sneaker():
    """
    Classes:
    0	T-shirt/top
    1	Trouser
    2	Pullover
    3	Dress
    4	Coat
    5	Sandal
    6	Shirt
    7	Sneaker
    8	Bag
    9	Ankle boot

    train: (12000, 784), test: (2000, 784)
    """
    eps_dataset = 0.1
    classes = [5, 7]  # 5 is 1, 7 is -1 in the binary classification scheme

    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train, X_test = X_train.astype(np.float64) / 255.0, X_test.astype(np.float64) / 255.0
    X_train = np.reshape(X_train, [X_train.shape[0], -1])
    X_test = np.reshape(X_test, [X_test.shape[0], -1])

    X_train, y_train, X_test, y_test = binary_from_multiclass(X_train, y_train, X_test, y_test, classes)
    return X_train, y_train, X_test, y_test, eps_dataset


def gts_30_70():
    """
    the class ids can be checked in the original data folders, for example:
    1: speed 30, 4: speed 70, 7: speed 100, 8: speed 120, 18: warning, 25: roadworks

    train: 4200x3072, test: 1380x3072
    """
    eps_dataset = 8 / 255  # following Madry et al, 2017 for cifar10
    classes = [1, 4]

    # Originally, all pixels values are uint8 values in [0, 255]
    train = scipy.io.loadmat('data/gts/gts_int_train.mat')
    test = scipy.io.loadmat('data/gts/gts_int_test.mat')
    X_train, y_train, X_test, y_test = train['images'], train['labels'], test['images'], test['labels']
    X_train, X_test = X_train.reshape(X_train.shape[0], -1), X_test.reshape(X_test.shape[0], -1)
    X_train, X_test = X_train / 255.0, X_test / 255.0
    y_train, y_test = y_train[0], y_test[0]  # get rid of the extra dimension

    X_train, y_train, X_test, y_test = binary_from_multiclass(X_train, y_train, X_test, y_test, classes)
    return X_train, y_train, X_test, y_test, eps_dataset


def gts_100_roadworks():
    """
    the class ids can be checked in the original data folders, for example:
    1: speed 30, 4: speed 70, 7: speed 100, 8: speed 120, 18: warning, 25: roadworks

    train: (2940, 3072), test: (930, 3072)
    """
    eps_dataset = 8 / 255  # following Madry et al, 2017 for cifar10
    classes = [7, 25]

    # Originally, all pixels values are uint8 values in [0, 255]
    train = scipy.io.loadmat('data/gts/gts_int_train.mat')
    test = scipy.io.loadmat('data/gts/gts_int_test.mat')
    X_train, y_train, X_test, y_test = train['images'], train['labels'], test['images'], test['labels']
    X_train, X_test = X_train.reshape(X_train.shape[0], -1), X_test.reshape(X_test.shape[0], -1)
    X_train, X_test = X_train / 255.0, X_test / 255.0
    y_train, y_test = y_train[0], y_test[0]  # get rid of the extra dimension

    X_train, y_train, X_test, y_test = binary_from_multiclass(X_train, y_train, X_test, y_test, classes)
    return X_train, y_train, X_test, y_test, eps_dataset


def har():
    eps_dataset = 0.05
    path = 'data/har/'
    X_train, X_test = np.loadtxt(path + 'X_train.txt'), np.loadtxt(path + 'X_test.txt')  # (7352, 561), (2947, 561)
    y_train, y_test = np.loadtxt(path + 'y_train.txt'), np.loadtxt(path + 'y_test.txt')  # 6 classes
    y_train, y_test = y_train - 1, y_test - 1  # make the class numeration start from 0
    return X_train, y_train, X_test, y_test, eps_dataset


all_datasets_dict = {
    'toy_2d_stumps': toy_2d_stumps,
    'toy_2d_trees': toy_2d_trees,
    'toy_2d_xor': toy_2d_xor,
    'toy_2d_wong': toy_2d_wong,

    'breast_cancer': breast_cancer,
    'diabetes': diabetes,
    'ijcnn1': ijcnn1,
    'cod_rna': cod_rna,

    'mnist_1_5': mnist_1_5,
    'mnist_2_6': mnist_2_6,
    'fmnist_sandal_sneaker': fmnist_sandal_sneaker,
    'gts_100_roadworks': gts_100_roadworks,
    'gts_30_70': gts_30_70,

    'har': har,
}

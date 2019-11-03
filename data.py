import numpy as np
import csv
import scipy.io
from tensorflow.keras.datasets import mnist as mnist_keras, fashion_mnist as fashion_mnist_keras, \
    cifar10 as cifar10_keras

data_dir = '/home/maksym/boost/data/'


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


def split_train_validation(X_train_orig, y_train_orig, frac_valid, shuffle=True):
    num_total = X_train_orig.shape[0]
    n_valid = int(frac_valid*num_total)
    idx = np.random.permutation(num_total) if shuffle else np.arange(num_total)
    if shuffle:
        X_valid, y_valid = X_train_orig[idx][:n_valid], y_train_orig[idx][:n_valid]
        X_train, y_train = X_train_orig[idx][n_valid:], y_train_orig[idx][n_valid:]
    else:
        # If no shuffle, then one has to ensure that the classes are balanced
        idx_valid, idx_train = [], []
        for cls in np.unique(y_train_orig):
            indices_cls = np.where(y_train_orig == cls)[0]
            proportion_cls = len(indices_cls) / num_total
            n_class_balanced_valid = int(proportion_cls * n_valid)
            idx_valid.extend(list(indices_cls[:n_class_balanced_valid]))
            idx_train.extend(list(indices_cls[n_class_balanced_valid:]))
        idx_valid, idx_train = np.array(idx_valid), np.array(idx_train)
        X_valid, y_valid = X_train_orig[idx_valid], y_train_orig[idx_valid]
        X_train, y_train = X_train_orig[idx_train], y_train_orig[idx_train]
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


def transform_labels_one_vs_all(y_train_orig, y_valid_orig, y_test_orig):
    n_cls = int(y_train_orig.max()) + 1
    if n_cls == 2:
        return y_train_orig[None, :], y_valid_orig[None, :], y_test_orig[None, :]

    labels = np.unique(y_train_orig)
    n_cls = len(labels)
    n_train, n_valid, n_test = y_train_orig.shape[0], y_valid_orig.shape[0], y_test_orig.shape[0]
    y_train, y_valid, y_test = np.zeros([n_cls, n_train]), np.zeros([n_cls, n_valid]), np.zeros([n_cls, n_test])
    for i_cls in range(n_cls):
        # convert from False/True to -1/1 compatible with One-vs-All formulation
        y_train[i_cls] = 2 * (y_train_orig == i_cls) - 1
        y_valid[i_cls] = 2 * (y_valid_orig == i_cls) - 1
        y_test[i_cls] = 2 * (y_test_orig == i_cls) - 1
    return y_train, y_valid, y_test


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
    path = data_dir + 'breast_cancer/breast-cancer-wisconsin.data'

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
    path = data_dir + 'diabetes/diabetes.csv'
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
    eps_dataset = 0.01  # Chen et al, 2019 used 0.1, but it was too high
    folder = data_dir + 'ijcnn1/'
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
    folder = data_dir + 'cod_rna/'
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

    # n_test_final = 10000  # take 10k test examples instead of all 270k
    n_test_final = num_test
    idx = np.random.permutation(num_test)[:n_test_final]
    X_test, y_test = X_test[idx], y_test[idx]
    return X_train, y_train, X_test, y_test, eps_dataset


def mnist_1_5():
    """
    train: (12163, 784), test: (2027, 784)
    """
    eps_dataset = 0.3
    classes = [1, 5]  # 2 is 1, 6 is -1 in the binary classification scheme

    (X_train, y_train), (X_test, y_test) = mnist_keras.load_data()
    X_train, X_test = X_train.astype(np.float64) / 255.0, X_test.astype(np.float64) / 255.0
    X_train = np.reshape(X_train, [X_train.shape[0], -1])
    X_test = np.reshape(X_test, [X_test.shape[0], -1])

    X_train, y_train, X_test, y_test = binary_from_multiclass(X_train, y_train, X_test, y_test, classes)
    return X_train, y_train, X_test, y_test, eps_dataset


def mnist_2_6():
    """
    train: (11876, 784), test: (1990, 784)
    """
    eps_dataset = 0.3
    classes = [2, 6]  # 2 is 1, 6 is -1 in the binary classification scheme

    (X_train, y_train), (X_test, y_test) = mnist_keras.load_data()
    X_train, X_test = X_train.astype(np.float64) / 255.0, X_test.astype(np.float64) / 255.0
    X_train = np.reshape(X_train, [X_train.shape[0], -1])
    X_test = np.reshape(X_test, [X_test.shape[0], -1])

    X_train, y_train, X_test, y_test = binary_from_multiclass(X_train, y_train, X_test, y_test, classes)
    return X_train, y_train, X_test, y_test, eps_dataset


def mnist():
    """
    train: (60000, 784), test: (10000, 784)
    """
    eps_dataset = 0.3

    (X_train, y_train), (X_test, y_test) = mnist_keras.load_data()
    X_train, X_test = X_train.astype(np.float64) / 255.0, X_test.astype(np.float64) / 255.0
    X_train = np.reshape(X_train, [X_train.shape[0], -1])
    X_test = np.reshape(X_test, [X_test.shape[0], -1])

    return X_train, y_train, X_test, y_test, eps_dataset


def cifar10():
    """
    train: (60000, 3072), test: (10000, 3072)
    """
    eps_dataset = 8/255

    (X_train, y_train), (X_test, y_test) = cifar10_keras.load_data()
    X_train, X_test = X_train.astype(np.float64) / 255.0, X_test.astype(np.float64) / 255.0
    X_train = np.reshape(X_train, [X_train.shape[0], -1])
    X_test = np.reshape(X_test, [X_test.shape[0], -1])

    y_train, y_test = y_train.flatten(), y_test.flatten()

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

    (X_train, y_train), (X_test, y_test) = fashion_mnist_keras.load_data()
    X_train, X_test = X_train.astype(np.float64) / 255.0, X_test.astype(np.float64) / 255.0
    X_train = np.reshape(X_train, [X_train.shape[0], -1])
    X_test = np.reshape(X_test, [X_test.shape[0], -1])

    X_train, y_train, X_test, y_test = binary_from_multiclass(X_train, y_train, X_test, y_test, classes)
    return X_train, y_train, X_test, y_test, eps_dataset


def fmnist():
    """
    train: (60000, 784), test: (10000, 784)
    """
    eps_dataset = 0.1

    (X_train, y_train), (X_test, y_test) = fashion_mnist_keras.load_data()
    X_train, X_test = X_train.astype(np.float64) / 255.0, X_test.astype(np.float64) / 255.0
    X_train = np.reshape(X_train, [X_train.shape[0], -1])
    X_test = np.reshape(X_test, [X_test.shape[0], -1])

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
    train = scipy.io.loadmat(data_dir + 'gts/gts_int_train.mat')
    test = scipy.io.loadmat(data_dir + 'gts/gts_int_test.mat')
    X_train, y_train, X_test, y_test = train['images'], train['labels'], test['images'], test['labels']
    X_train, X_test = X_train.reshape(X_train.shape[0], -1), X_test.reshape(X_test.shape[0], -1)
    X_train, X_test = X_train / 255.0, X_test / 255.0
    y_train, y_test = y_train[0], y_test[0]  # get rid of the extra dimension

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
    train = scipy.io.loadmat(data_dir + 'gts/gts_int_train.mat')
    test = scipy.io.loadmat(data_dir + 'gts/gts_int_test.mat')
    X_train, y_train, X_test, y_test = train['images'], train['labels'], test['images'], test['labels']
    X_train, X_test = X_train.reshape(X_train.shape[0], -1), X_test.reshape(X_test.shape[0], -1)
    X_train, X_test = X_train / 255.0, X_test / 255.0
    y_train, y_test = y_train[0], y_test[0]  # get rid of the extra dimension

    X_train, y_train, X_test, y_test = binary_from_multiclass(X_train, y_train, X_test, y_test, classes)
    return X_train, y_train, X_test, y_test, eps_dataset


def har():
    """
    Human activity recognition dataset from https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
    Note: Wong and Kolter, ICML 2018 used eps=0.05, but the data points were from -1 to 1.
    We use equivalently eps=0.025, but data points from 0 to 1.

    The labels are in {0, 1, 2, 3, 4, 5}.

    train: (7352, 561), test: (2947, 561), n classes: 6.
    """
    eps_dataset = 0.025
    path_train, path_test = data_dir + 'har/train/', data_dir + 'har/test/'
    X_train, X_test = np.loadtxt(path_train + 'X_train.txt'), np.loadtxt(path_test + 'X_test.txt')
    y_train, y_test = np.loadtxt(path_train + 'y_train.txt'), np.loadtxt(path_test + 'y_test.txt')
    y_train, y_test = y_train - 1, y_test - 1  # make the class numeration start from 0
    X_train, X_test = (X_train + 1) / 2, (X_test + 1) / 2  # from [-1, 1] to [0, 1]
    return X_train, y_train, X_test, y_test, eps_dataset


def convert_to_float32(X):
    return X.astype(np.float32)


def random_crop(image, n_crop):
    h, w, _ = image.shape
    top = np.random.randint(0, n_crop)
    left = np.random.randint(0, n_crop)
    bottom = h - (n_crop - top)
    right = w - (n_crop - left)
    image = image[top:bottom, left:right, :]
    return image


def horizontal_flip(images, prob=0.5):
    if np.random.rand() < prob:
        images = images[:, :, ::-1, :]
    return images


def data_augment(X, dataset):
    num, dim = X.shape
    img_shape = datasets_img_shapes[dataset]
    X_img = np.reshape(np.copy(X), [num, *img_shape])
    if len(img_shape) == 2:  # introduce a fake last dimension for grayscale datasets
        X_img = X_img[:, :, :, None]

    n_crop = 2
    X_img_pad = np.pad(X_img, [(0, 0), (n_crop//2, n_crop//2), (n_crop//2, n_crop//2), (0, 0)], 'constant', constant_values=0)  # zero padding
    for i in range(num):
        X_img[i] = random_crop(X_img_pad[i], n_crop=n_crop)  # up to `n_crop` pixels are cropped

    if dataset in ['cifar10']:
        X_img = horizontal_flip(X_img)

    return np.reshape(X_img, [num, dim])


def crop_batch(X_img, n_h, n_w, n_crop):
    _, h, w, _ = X_img.shape
    bottom, right = h - (n_crop - n_h), w - (n_crop - n_w)
    return X_img[:, n_h:bottom, n_w:right, :]


def extend_dataset(X, dataset):
    num, dim = X.shape
    img_shape = datasets_img_shapes[dataset]
    X_img = np.reshape(np.copy(X), [num, *img_shape])
    if len(img_shape) == 2:  # introduce a fake last dimension for grayscale datasets
        X_img = X_img[:, :, :, None]

    n_crop = 2
    X_img_pad = np.pad(X_img, [(0, 0), (n_crop // 2, n_crop // 2), (n_crop // 2, n_crop // 2), (0, 0)], 'constant',
                       constant_values=0)

    # Note: (1, 1) is the original image
    X_img_l = crop_batch(X_img_pad, 1, 0, n_crop)
    X_img_r = crop_batch(X_img_pad, 1, 2, n_crop)
    X_img_t = crop_batch(X_img_pad, 0, 1, n_crop)
    X_img_b = crop_batch(X_img_pad, 2, 1, n_crop)

    X_img_extended = np.vstack([X_img, X_img_l, X_img_r, X_img_t, X_img_b])

    # if dataset in ['cifar10']:  # would lead to 10x expansion of the training data - might be too comp. expensive
    #     X_img_horiz_flip = X_img_extended[:, :, ::-1, :]
    #     X_img_extended = np.vstack([X_img_extended, X_img_horiz_flip])

    X_final = np.reshape(X_img_extended, [-1, dim])
    return X_final


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
    'mnist': mnist,
    'fmnist': fmnist,
    'cifar10': cifar10,
}
dataset_names_dict = {
    'toy_2d_stumps': 'toy_2d_stumps',
    'toy_2d_trees': 'toy_2d_trees',
    'toy_2d_xor': 'toy_2d_xor',
    'toy_2d_wong': 'toy_2d_wong',

    'breast_cancer': 'breast-cancer',
    'diabetes': 'diabetes',
    'ijcnn1': 'IJCNN1',
    'cod_rna': 'cod-rna',

    'mnist_1_5': 'MNIST 1-5',
    'mnist_2_6': 'MNIST 2-6',
    'fmnist_sandal_sneaker': 'FMNIST shoes',
    'gts_100_roadworks': 'GTS 100-rw',
    'gts_30_70': 'GTS 30-70',

    'har': 'har',
    'mnist': 'mnist',
    'fmnist': 'fmnist',
    'cifar10': 'cifar10',
}
datasets_img_shapes = {
    'mnist_1_5': (28, 28),
    'mnist_2_6': (28, 28),
    'mnist': (28, 28),
    'fmnist': (28, 28),
    'fmnist_sandal_sneaker': (28, 28),
    'gts_100_roadworks': (32, 32, 3),
    'gts_30_70': (32, 32, 3),
    'cifar10': (32, 32, 3),
}
datasets_feature_names = {
    'breast_cancer': ['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity', 'concave points', 'symmetry', 'fractal dimension'],
    'diabetes': ['# pregnancies', 'glucose', 'blood pressure', 'skin thickness', 'insulin', 'body mass index', 'diabetes pedigree', 'age'],
    'cod_rna': ['Dynalign score', 'shorter seq. length', 'A freq. of seq. 1', 'U freq. of seq. 1', 'C freq. of seq. 1', 'A freq. of seq. 2', 'U freq. of seq. 2', 'C freq. of seq. 2'],
}


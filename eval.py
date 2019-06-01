import argparse
import numpy as np
import data
from stump_ensemble import StumpEnsemble
from utils import Logger

np.random.seed(1)
np.set_printoptions(precision=10)

parser = argparse.ArgumentParser(description='Define hyperparameters.')
parser.add_argument('--dataset', type=str, default='mnist_2_6', help='Dataset: toy2d, mnist_2_6, har.')
parser.add_argument('--n_eval', type=int, default='-1', help='On how many points to eval.')
parser.add_argument('--model_path', type=str, default='2019-05-21 15:41:28 dataset=mnist_2_6 model=robust_exact n_train=-1 n_trials_coord=10 eps=0.300 lr=1.0',
                    help='Model name.')
args = parser.parse_args()

X_train, y_train, X_test, y_test, eps = data.all_datasets_dict[args.dataset]()

ensemble = StumpEnsemble('stump', 10, 1.0)  # the hps here do not matter (they matter only for training)
ensemble.load('exps/{}.model'.format(args.model_path))

test_err = np.mean(y_test * ensemble.predict(X_test) < 0.0)
print('test err: {:.2%}'.format(test_err))

if args.n_eval != -1:
    X_test, y_test = X_test[:args.n_eval], y_test[:args.n_eval]

deltas = ensemble.exact_adv_example(X_test, y_test)
avg_db_dist = np.abs(deltas).max(1).mean(0)
print('avg dist to db: {:.3f}'.format(avg_db_dist))


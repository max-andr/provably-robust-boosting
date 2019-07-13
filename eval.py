import argparse
import numpy as np
import data
from tree_ensemble import TreeEnsemble
from stump_ensemble import StumpEnsemble

np.random.seed(1)
np.set_printoptions(precision=10)

parser = argparse.ArgumentParser(description='Define hyperparameters.')
parser.add_argument('--dataset', type=str, default='mnist_2_6', help='Dataset: toy2d, mnist_2_6, har.')
parser.add_argument('--weak_learner', type=str, default='tree', help='Weak learner: stump or tree.')
parser.add_argument('--n_eval', type=int, default='-1', help='On how many points to evaluate.')
parser.add_argument('--model_path', type=str, default='2019-07-07 10:05:48 dataset=mnist_2_6 weak_learner=tree model=robust_bound n_train=-1 n_trials_coord=100 eps=0.300 max_depth=4 lr=1.0',
                    help='Model name.')
args = parser.parse_args()

X_train, y_train, X_test, y_test, eps = data.all_datasets_dict[args.dataset]()

# the hyperparameters of recreated models do not matter (they matter only for training)
if args.weak_learner == 'stump':
    ensemble = StumpEnsemble('stump', 0, 0)
elif args.weak_learner == 'tree':
    ensemble = TreeEnsemble('tree', 0, 0, 0, 0, 0)
else:
    raise ValueError('wrong weak learner')

ensemble.load('models/{}.model.npy'.format(args.model_path))

test_err = np.mean(y_test * ensemble.predict(X_test) < 0.0)
print('test err: {:.2%}'.format(test_err))

# if args.n_eval != -1:
#     X_test, y_test = X_test[:args.n_eval], y_test[:args.n_eval]
#
# deltas = ensemble.exact_adv_example(X_test, y_test)
# avg_db_dist = np.abs(deltas).max(1).mean(0)
# print('avg dist to db: {:.3f}'.format(avg_db_dist))


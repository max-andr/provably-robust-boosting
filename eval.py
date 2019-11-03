import argparse
import numpy as np
import time
import data
import attacks
from stump_ensemble import StumpEnsemble
from tree_ensemble import TreeEnsemble
from utils import extract_hyperparam
from classifiers import OneVsAllClassifier


np.random.seed(1)
np.set_printoptions(precision=10)

parser = argparse.ArgumentParser(description='Define hyperparameters.')
parser.add_argument('--n_eval', type=int, default='-1', help='On how many points to eval.')
parser.add_argument('--iter', type=int, default=-1, help='Which iteration (i.e. number of trees) to take.')
parser.add_argument('--n_iter_attack', type=int, default=1, help='Which iteration (i.e. number of trees) to take.')
parser.add_argument('--exp_folder', type=str, default='models/models_trees_multiclass', help='Experiment name')
parser.add_argument('--model_path', type=str, default='2019-08-06 14:59:51 dataset=fmnist weak_learner=tree model=robust_bound n_train=-1 n_trials_coord=784 eps=0.100 max_depth=30 lr=0.05',
                    help='Model name.')
args = parser.parse_args()
exp_folder = args.exp_folder

# the info about dataset, weak_learner is already encoded in the model path
dataset = extract_hyperparam(args.model_path, 'dataset=')
weak_learner = extract_hyperparam(args.model_path, 'weak_learner=')
max_depth = extract_hyperparam(args.model_path, 'max_depth=')
model = extract_hyperparam(args.model_path, 'model=')

X_train, y_train, X_test, y_test, eps = data.all_datasets_dict[dataset]()
X_train, X_test = data.convert_to_float32(X_train), data.convert_to_float32(X_test)
n_cls = int(y_train.max()) + 1
y_train, _, y_test = data.transform_labels_one_vs_all(y_train, y_train, y_test)

metrics = np.loadtxt(exp_folder + '/' + args.model_path + '.metrics')
if args.iter == -1:
    valid_errs, valid_adv_errs_lb, valid_adv_errs = metrics[:, 8], metrics[:, 9], metrics[:, 10]
    # Model selection
    if model == 'plain':
        iter_to_take = np.argmin(valid_errs)
    elif model in ['at_cube', 'robust_bound', 'robust_exact']:
        iter_to_take = np.argmin(valid_adv_errs)
    else:
        raise ValueError('wrong model name')
else:
    iter_to_take = args.iter

ensembles = []
n_classifiers = n_cls if n_cls > 2 else 1
for i_clsf in range(n_classifiers):
    if weak_learner == 'stump':
        # the hyperparameters of recreated models do not matter (they matter only for training)
        ensemble = StumpEnsemble(weak_learner, 0, 0, 0, 0, 0)
    elif weak_learner == 'tree':
        ensemble = TreeEnsemble(weak_learner, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    else:
        raise ValueError('wrong weak learner')
    ensembles.append(ensemble)

model_ova = OneVsAllClassifier(ensembles)
model_ova.load('{}/{}.model.npy'.format(exp_folder, args.model_path), iteration=iter_to_take)

if args.n_eval != -1:
    X_test, y_test = X_test[:args.n_eval], y_test[:, :args.n_eval]

time_te_start = time.time()
fmargin = model_ova.fmargin(X_test, y_test)
test_err = 1 - np.mean(fmargin > 0.0)
time_te = time.time() - time_te_start
print('te={:.2%} ({:.4f}s)'.format(test_err, time_te))

time_lrte_start = time.time()
fmargin_attack = attacks.cube_attack(model_ova, X_test, y_test, eps, args.n_iter_attack, p=0.15)[0]
lrte = 1 - (fmargin_attack > 0.0).mean()
time_lrte = time.time() - time_lrte_start
print('lrte={:.2%} ({:.4f}s)'.format(lrte, time_lrte))

if weak_learner == 'stump':
    cert_f = model_ova.fmargin_exact
elif weak_learner == 'tree':
    cert_f = model_ova.fmargin_treewise
else:
    raise ValueError('wrong weak learner')
# the first time numba takes some time to compile, thus we need this line to properly measure the certification speed
_ = 1 - (cert_f(X_train[:1000], y_train[:, :1000], eps) > 0.0).mean()

time_urte_start = time.time()
fmargin_cert = cert_f(X_test, y_test, eps)
urte = 1 - (fmargin_cert > 0.0).mean()
time_urte = time.time() - time_urte_start
print('urte={:.2%} ({:.5f}s)'.format(urte, time_urte))

print('TE: {:.2%} ({:.4f}s)  LRTE: {:.2%} ({:.4f}s)  URTE: {:.2%} ({:.5f}s)'.format(test_err, time_te, lrte, time_lrte, urte, time_urte))

import argparse
import numpy as np
import data
import time
from multiprocessing import Pool
from datetime import datetime
from utils import Logger
from stump_ensemble import StumpEnsemble
from tree_ensemble import Tree, TreeEnsemble
from attacks import cube_attack
from classifiers import OneVsAllClassifier


def eval_metrics(model_ova, X, y, pred, cert_tw, time_cert, deltas, weak_learner, eps_eval, log, n_trials_attack, check_bounds=True):
    """ Evaluation metrics for validation and test sets. """
    if X.shape[0] == 0:  # if no examples provided (e.g., the validation set is empty)
        return 1.0, 1.0, 1.0, 1.0, pred, cert_tw, time_cert, deltas

    # To save some computations, in particular for `find_min_yf()` which is slow for deep trees
    for i_clsf in range(len(model_ova.models)):
        pred[i_clsf] += model_ova.models[i_clsf].trees[-1].predict(X)
        time_before_cert = time.time()
        cert_tw[i_clsf] += model_ova.models[i_clsf].trees[-1].find_min_yf(X, y[i_clsf], eps_eval)
        time_cert += time.time() - time_before_cert

    yf = model_ova.fmargin(X, y, fx_vals=pred)
    min_yf_ub, deltas = cube_attack(model_ova, X, y, eps_eval, n_trials_attack, deltas_init=deltas)
    min_yf_lb = model_ova.fmargin_treewise(X, y, eps_eval, fx_vals=cert_tw)
    if weak_learner == 'stump':
        time_before_cert = time.time()
        min_yf_exact = model_ova.fmargin_exact(X, y, eps_eval)
        time_cert = time.time() - time_before_cert
    else:  # for trees, yf_exact just gets assigned min_yf_lb
        min_yf_exact = min_yf_lb

    is_correct = yf > 0.0
    is_rob_ub = min_yf_lb > 0.0
    is_rob_lb = min_yf_ub > 0.0
    is_rob_exact = min_yf_exact > 0.0

    err = 1 - is_correct.mean()
    adv_err_lb = 1 - is_rob_lb.mean()
    adv_err = 1 - is_rob_exact.mean()
    adv_err_ub = 1 - is_rob_ub.mean()

    if check_bounds:
        if np.sum(is_correct < is_rob_lb) > 0:
            log.print('Number pts violated correct < attack: {} ({})'.format(
                np.sum(is_correct < is_rob_lb), np.where(is_correct < is_rob_lb)[0]))
        if np.sum(is_rob_lb < is_rob_exact) > 0:
            log.print('Number pts violated attack < exact: {} ({})'.format(
                np.sum(is_rob_lb < is_rob_exact), np.where(is_rob_lb < is_rob_exact)[0]))
        if np.sum(is_rob_exact < is_rob_ub) > 0:
            log.print('Number pts violated exact < rob_ub: {} ({})'.format(
                np.sum(is_rob_exact < is_rob_ub), np.where(is_rob_exact < is_rob_ub)[0]))

    return err, adv_err_lb, adv_err, adv_err_ub, pred, cert_tw, time_cert, deltas


def update_margin(ensemble_new, X_train, y_train, margin, gamma, model, eps_train):
    if model in ['plain', 'da_uniform', 'at_cube']:
        yf = y_train * ensemble_new.trees[-1].predict(X_train)
        margin += yf
        gamma *= np.exp(-yf)
    elif model == 'robust_bound':
        min_yf_lb = ensemble_new.trees[-1].find_min_yf(X_train, y_train, eps_train)
        margin += min_yf_lb
        gamma *= np.exp(-min_yf_lb)
    elif model == 'robust_exact':
        margin = ensemble_new.certify_exact(X_train, y_train, eps_train)
        gamma = np.exp(-margin)
    else:
        raise ValueError('wrong model')
    return margin, gamma


def perturb_dataset(X_train, y_train, model_ova, model, eps_train, kantchelian_at):
    n_iter_at = 10
    num = X_train.shape[0]

    X_train_fit = np.copy(X_train)
    # Note: da_uniform in the current form (continuous noise) can lead to a significant slowdown since we have to
    # check much more thresholds than usually (n instead of 256 for image datasets)
    if model == 'da_uniform':  # or (model == 'at_cube' and model_ova.models[0].trees == []):
        deltas = np.random.uniform(-eps_train, eps_train, size=X_train.shape)
        X_train_fit = np.clip(X_train + deltas, 0.0, 1.0)  # preserve the valid data range
    elif model == 'at_cube':
        if kantchelian_at:
            _, deltas_at = cube_attack(model_ova, X_train[num // 2:], y_train[:, num // 2:], eps_train,
                                       n_trials=n_iter_at, independent_delta=True)
            X_train_fit[num // 2:] = X_train[num // 2:] + deltas_at
        else:
            _, deltas_at = cube_attack(model_ova, X_train, y_train, eps_train, n_trials=n_iter_at,
                                       independent_delta=True)
            X_train_fit = X_train + deltas_at
    return X_train_fit


def train_iter_binary_clsf(ensemble_prev, X_train, y_train, gamma, margin, model, weak_learner_type, eps_train, i_clsf):
    if model in ['da_uniform', 'at_cube']:  # we recalculate gammas if the training set changes every iteration
        margin = y_train * ensemble_prev.predict(X_train)
        gamma = np.exp(-margin)
    ensemble_new = ensemble_prev.copy()
    gamma_prev, margin_prev = np.copy(gamma), np.copy(margin)
    loss_prev = np.mean(gamma_prev)

    if weak_learner_type == 'stump':
        weak_learner = ensemble_prev.fit_stumps_over_coords(X_train, y_train, gamma, model, eps_train)
        ensemble_new.add_weak_learner(weak_learner)
        tree_depth, tree_n_nodes = 1, 1
    elif weak_learner_type == 'tree':
        # depth=1 means that we start counting from 1 (i.e. decision stumps are counted as trees of depth=1)
        weak_learner = ensemble_prev.fit_tree(X_train, y_train, gamma, model, eps_train, depth=1)
        # add a new weak learner to a new ensemble without modifying yet the main ensemble
        ensemble_new.add_weak_learner(weak_learner)
        print('Starting pruning for class {}...'.format(i_clsf))
        ensemble_new.prune_last_tree(X_train, y_train, margin, eps_train, model)
        print('Finished pruning for class {}...'.format(i_clsf))
        tree_depth, tree_n_nodes = ensemble_new.trees[-1].get_depth(), ensemble_new.trees[-1].get_n_nodes()
    else:
        raise ValueError('wrong weak learner')

    margin, gamma = update_margin(ensemble_new, X_train, y_train, margin, gamma, model, eps_train)

    loss = np.mean(gamma)
    if model not in ['da_uniform', 'at_cube'] and loss >= loss_prev:  # we return the new ensemble only if it reduces the loss
        ensemble_prev.add_weak_learner(Tree())
        print('Added empty weak learner (loss_new={:.4} >= loss_prev={:.4})'.format(loss, loss_prev))
        return ensemble_prev, gamma_prev, margin_prev, 0, 0
    else:  # to make `# weak learners` == `n_iter`, just add a constant stump/tree
        return ensemble_new, gamma, margin, tree_depth, tree_n_nodes


def robust_boost(model_ova, X_train, y_train, X_valid, y_valid, X_test, y_test, weak_learner_type, n_trees,
                 eps_train, eps_eval, n_trials_attack, cb_weights, model, log, model_path, metrics_path, debug):
    n_clsf = len(model_ova.models)
    parallel = True if n_clsf > 1 else False
    # If AT is applied, then it's done as in Kantchelian et al (i.e. 50% clean + 50% adversarial) => works better
    kantchelian_at = True
    if model == 'at_cube' and kantchelian_at:
        X_train = np.vstack([X_train, X_train])
        y_train = np.hstack([y_train, y_train])

    n_eval_train = min(X_train.shape[0], 5000)  # number of training examples to use for evaluation (not too critical, but helps for speed-up)
    time_start = time.time()
    n_train, n_valid, n_test = X_train.shape[0], X_valid.shape[0], X_test.shape[0]
    time_cert_train, time_cert_valid, time_cert_test = 0.0, 0.0, 0.0
    deltas_at, deltas_train = np.zeros_like(X_train), np.zeros_like(X_train)
    deltas_valid, deltas_test = np.zeros_like(X_valid), np.zeros_like(X_test)

    metrics = []  # all metrics are collected in this list
    margin = np.zeros([n_clsf, n_train])
    pred_train, pred_valid, pred_test = np.zeros([n_clsf, n_eval_train]), np.zeros([n_clsf, n_valid]), np.zeros([n_clsf, n_test])
    cert_tw_train, cert_tw_valid, cert_tw_test = np.zeros([n_clsf, n_eval_train]), np.zeros([n_clsf, n_valid]), np.zeros([n_clsf, n_test])
    gamma = np.ones([n_clsf, n_train])  # note: no normalization since it is unnecessary and ambiguous for robust training
    if cb_weights:  # class-balancing weights
        for i_clsf in range(n_clsf):
            gamma[i_clsf][y_train[i_clsf] == 1] *= (y_train[i_clsf] == -1).sum() / (y_train[i_clsf] == 1).sum()

    X_train_fit = X_train
    if parallel:
        proc_pool = Pool(n_clsf)
    for it in range(1, n_trees + 1):
        tree_depths, tree_ns_nodes = np.zeros(n_clsf), np.zeros(n_clsf)
        procs = []

        # # changing the dataset at every iteration doesn't seem to work very well with boosting
        # X_train_fit = data.data_augment(X_train, dataset) if data_augm and dataset in data.datasets_img_shapes else X_train
        X_train_fit = perturb_dataset(X_train_fit, y_train, model_ova, model, eps_train, kantchelian_at) if model in ['da_uniform', 'at_cube'] else X_train
        for i_clsf in range(n_clsf):  # start all the processes in parallel
            ensemble = model_ova.models[i_clsf]
            if parallel:
                train_iter_args = (ensemble, X_train_fit, y_train[i_clsf], gamma[i_clsf], margin[i_clsf], model,
                                   weak_learner_type, eps_train, i_clsf)
                procs.append(proc_pool.apply_async(train_iter_binary_clsf, args=train_iter_args))
            else:
                model_ova.models[i_clsf], gamma[i_clsf], margin[i_clsf], tree_depths[i_clsf], tree_ns_nodes[i_clsf] = train_iter_binary_clsf(
                    ensemble, X_train_fit, y_train[i_clsf], gamma[i_clsf], margin[i_clsf], model, weak_learner_type, eps_train, i_clsf)
        if parallel:
            for i_clsf in range(n_clsf):  # wait until the results are done and fetch them
                model_ova.models[i_clsf], gamma[i_clsf], margin[i_clsf], tree_depths[i_clsf], tree_ns_nodes[i_clsf] = procs[i_clsf].get()

        # Evaluations: currently designed in a way that we neeed to do it *every* iteration
        print('starting evaluation ({:.2f}s)'.format(time.time() - time_start))
        tree_depth, tree_n_nodes = np.mean(tree_depths), np.mean(tree_ns_nodes)
        train_loss = np.mean(gamma)  # mean over classes (axis=0) and examples (axis=1)
        if it > 1 and train_loss > metrics[-1][7] + 1e-7:
            log.print('The train loss increases: prev {:.5f} now {:.5f}'.format(metrics[-1][7], train_loss))

        train_err, train_adv_err_lb, train_adv_err, train_adv_err_ub, pred_train, cert_tw_train, time_cert_train, deltas_train = eval_metrics(
            model_ova, X_train[:n_eval_train], y_train[:, :n_eval_train], pred_train, cert_tw_train, time_cert_train,
            deltas_train[:n_eval_train], weak_learner_type, eps_eval, log, n_trials_attack=0, check_bounds=False)
        valid_err, valid_adv_err_lb, valid_adv_err, valid_adv_err_ub, pred_valid, cert_tw_valid, time_cert_valid, deltas_valid = eval_metrics(
            model_ova, X_valid, y_valid, pred_valid, cert_tw_valid, time_cert_valid, deltas_valid, weak_learner_type, eps_eval, log, n_trials_attack)
        test_err, test_adv_err_lb, test_adv_err, test_adv_err_ub, pred_test, cert_tw_test, time_cert_test, deltas_test = eval_metrics(
            model_ova, X_test, y_test, pred_test, cert_tw_test, time_cert_test, deltas_test, weak_learner_type, eps_eval, log, n_trials_attack)

        train_str = '[train] err {:.2%} adv_err {:.2%} loss {:.5f}'.format(
            train_err, train_adv_err, train_loss)
        valid_str = '[valid] err {:.2%} adv_err {:.2%}'.format(valid_err, valid_adv_err)
        str_adv_err = 'adv_err {:.2%} '.format(test_adv_err) if weak_learner_type == 'stump' else ''
        test_str = '[test] err {:.2%} adv_err_lb {:.2%} {}adv_err_ub {:.2%}'.format(
            test_err, test_adv_err_lb, str_adv_err, test_adv_err_ub)

        if weak_learner_type == 'tree':
            tree_info_str = '[tree] depth {:.2f} nodes {:.2f}'.format(tree_depth, tree_n_nodes)
        else:
            tree_info_str = ''
        time_elapsed = time.time() - time_start

        log.print('iter: {} {} | {} | {} | {} ({:.3f}s, {:.3f}s)'.format(it, test_str, valid_str, train_str, tree_info_str, time_elapsed, time_cert_test))
        metrics.append([it, test_err, test_adv_err_lb, test_adv_err, test_adv_err_ub, train_err, train_adv_err,
                        train_loss, valid_err, valid_adv_err_lb, valid_adv_err, valid_adv_err_ub, time_elapsed, time_cert_test,
                        tree_depth, tree_n_nodes])

        if not debug and (it % 5 == 0 or it == n_trees) and metrics_path != '':
            model_ova.save(model_path)
            np.savetxt(metrics_path, metrics)

    log.print('(done in {:.2f} min)'.format((time.time() - time_start) / 60))
    if not debug:
        log.print('Model path: {}.npy'.format(model_path))
        log.print('Metrics path: {}'.format(metrics_path))


def main():
    np.random.seed(1)
    np.set_printoptions(precision=10)

    parser = argparse.ArgumentParser(description='Define hyperparameters.')
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='breast_cancer, diabetes, cod_rna, mnist_1_5, mnist_2_6, fmnist_sandal_sneaker, gts_30_70,'
                             ' gts_100_roadworks')
    parser.add_argument('--model', type=str, default='plain',
                        help='plain, da_uniform, at_cube, robust_exact, robust_bound.')
    parser.add_argument('--weak_learner', type=str, default='tree', help='stump or tree')
    parser.add_argument('--max_depth', type=int, default=4, help='Depth of trees (only used when weak_learner==tree).')
    parser.add_argument('--max_weight', type=float, default=1.0, help='The maximum leaf weight.')
    parser.add_argument('--n_bins', type=int, default=-1, help='By default we check all thresholds.')
    parser.add_argument('--lr', type=float, default=0.2, help='Shrinkage parameter (aka learning rate).')
    parser.add_argument('--eps', type=float, default=-1, help='Linf epsilon. -1 means to use the default epsilons.')
    parser.add_argument('--n_train', type=int, default=-1, help='Number of training points to take.')
    parser.add_argument('--debug', action='store_true', help='Debugging mode: not many samples for the attack.')
    args = parser.parse_args()

    if args.weak_learner == 'stump' or (args.weak_learner == 'tree' and args.max_depth == 1):
        n_iter = 300
    elif args.weak_learner == 'tree':
        depth_iters_map = {2: 300, 4: 150, 6: 100, 8: 75}
        if args.max_depth in depth_iters_map:
            n_iter = depth_iters_map[args.max_depth]
        else:
            n_iter = 300
    else:
        raise ValueError('wrong weak learner')

    # max value of the leaf weights; has an important regularization effect similar to the learning rate
    max_weight = args.max_weight
    # to prevent extreme overfitting to a few points
    min_samples_split = 10 if args.dataset not in ['mnist', 'fmnist', 'cifar10'] else 200
    min_samples_leaf = 5
    n_trials_attack = 20 if args.dataset not in ['mnist', 'fmnist', 'cifar10'] else 10
    n_trials_attack = n_trials_attack if args.weak_learner == 'tree' else 1  # 1 iter is more of a sanity check
    frac_valid = 0.2 if args.dataset not in ['mnist', 'fmnist', 'cifar10'] else 0.0
    extend_dataset = True if args.dataset in ['mnist', 'fmnist', 'cifar10'] else False

    X_train, y_train, X_test, y_test, eps_dataset = data.all_datasets_dict[args.dataset]()
    X_train, X_test = data.convert_to_float32(X_train), data.convert_to_float32(X_test)
    X_train, y_train, X_valid, y_valid = data.split_train_validation(X_train, y_train, frac_valid, shuffle=True)
    if args.n_train != -1:
        X_train, y_train = X_train[:args.n_train], y_train[:args.n_train]

    n_cls = int(y_train.max()) + 1
    cb_weights = True if n_cls > 2 else False  # helps to convergence speed and URTE (especially, on MNIST)
    y_train, y_valid, y_test = data.transform_labels_one_vs_all(y_train, y_valid, y_test)

    if extend_dataset:
        X_train = data.extend_dataset(X_train, args.dataset)
        y_train = np.tile(y_train, [1, X_train.shape[0] // y_train.shape[1]])

    n_trials_coord = X_train.shape[1]  # we check all coordinates for every split

    if args.eps == -1:  # then use the default one if not specified from cmd
        eps_train = eps_eval = eps_dataset if args.model != 'plain' else 0.0  # not strictly needed
    else:
        eps_train = eps_eval = args.eps

    cur_timestamp = str(datetime.now())[:-7]
    hps_str_full = 'dataset={} weak_learner={} model={} n_train={} n_iter={} n_trials_coord={} eps={:.3f} min_samples_split={} ' \
                   'min_samples_leaf={} max_depth={} max_weight={} lr={} n_trials_attack={} cb_weights={} max_weight={} n_bins={} ' \
                   'expand_train_set={}'.format(
        args.dataset, args.weak_learner, args.model, args.n_train, n_iter, n_trials_coord, eps_train, min_samples_split,
        min_samples_leaf, args.max_depth, max_weight, args.lr, n_trials_attack, cb_weights, max_weight, args.n_bins, extend_dataset)
    hps_str_short = 'dataset={} weak_learner={} model={} n_train={} n_trials_coord={} eps={:.3f} max_depth={} max_weight={} lr={}'.format(
        args.dataset, args.weak_learner, args.model, args.n_train, n_trials_coord, eps_train, args.max_depth, max_weight, args.lr)

    exp_folder = 'exps_test'
    log_path = '{}/{} {}.log'.format(exp_folder, cur_timestamp, hps_str_short)
    model_path = '{}/{} {}.model'.format(exp_folder, cur_timestamp, hps_str_short)
    metrics_path = '{}/{} {}.metrics'.format(exp_folder, cur_timestamp, hps_str_short)

    log = Logger(log_path)
    log.print('Boosting started: {} {}'.format(cur_timestamp, hps_str_full))

    ensembles = []
    n_classifiers = n_cls if n_cls > 2 else 1
    for i_clsf in range(n_classifiers):
        if args.weak_learner == 'stump':
            ensemble = StumpEnsemble(args.weak_learner, n_trials_coord, args.lr, i_clsf, args.n_bins, max_weight)
        elif args.weak_learner == 'tree':
            ensemble = TreeEnsemble(args.weak_learner, n_trials_coord, args.lr, min_samples_split, min_samples_leaf, i_clsf,
                                    args.max_depth, gamma_hp=0.0, n_bins=args.n_bins, max_weight=max_weight)
        else:
            raise ValueError('wrong weak learner')
        ensembles.append(ensemble)
    model_ova = OneVsAllClassifier(ensembles)

    robust_boost(model_ova, X_train, y_train, X_valid, y_valid, X_test, y_test, args.weak_learner, n_iter, eps_train,
                 eps_eval, n_trials_attack, cb_weights, args.model, log, model_path, metrics_path,
                 args.debug)


if __name__ == '__main__':
    main()

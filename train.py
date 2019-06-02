import argparse
import numpy as np
import data
import time
import ipdb as pdb
from datetime import datetime
from utils import Logger
from stump_ensemble import StumpEnsemble
from tree_ensemble import TreeEnsemble


def robust_boost(ensemble, X_train, y_train, X_valid, y_valid, X_test, y_test, n_trees,
                 eps, eps_eval, model, log, model_path, metrics_path):
    time_start = time.time()
    num, dim = X_train.shape

    metrics = []  # all metrics are collected in this list
    gamma = np.ones(num)  # note: no normalization since it is unnecessary and ambiguous for robust training
    for it in range(1, n_trees + 1):
        if ensemble.weak_learner == 'stump':
            weak_learner = ensemble.fit_stump(X_train, y_train, gamma, model, eps)
            ensemble.add_weak_learner(weak_learner)
        elif ensemble.weak_learner == 'tree':
            weak_learner = ensemble.fit_tree(X_train, y_train, gamma, model, eps, depth=1)
            ensemble.add_weak_learner(weak_learner)
            ensemble.prune_last_tree(X_train, y_train, eps, model)
        else:
            raise ValueError('wrong weak learner')

        Fx_y = y_train * ensemble.predict(X_train)
        min_Fx_y_treewise = ensemble.certify_treewise_bound(X_train, y_train, eps)
        min_Fx_y_exact = ensemble.certify_exact(X_train, y_train, eps)
        if model == 'plain':
            gamma = np.exp(-Fx_y)
        elif model == 'robust_bound':
            gamma = np.exp(-min_Fx_y_treewise)
        elif model == 'robust_exact':  # min_d y*F(x+d) is taken jointly over the old ensemble + new weak learner
            gamma = np.exp(-min_Fx_y_exact)
        else:
            raise ValueError('wrong model')

        if it % 1 == 0:  # creates some overhead for plain training
            is_correct = y_test * ensemble.predict(X_test) > 0
            if ensemble.weak_learner == 'stump' or dim <= 2:  # or low-dimensional dataset such as toy 2d datasets
                n_trials_sampling = 20  # for stumps it's just a sanity check since we know the exact RTE
            else:
                n_trials_sampling = 250  # but for trees it's really important since we don't know the exact RTE
            is_robust_attack = ensemble.attack_by_sampling(X_test, y_test, eps_eval, n_trials_sampling) > 0.0
            is_robust_exact = ensemble.certify_exact(X_test, y_test, eps_eval) > 0.0
            is_rob_ub = ensemble.certify_treewise_bound(X_test, y_test, eps_eval) > 0.0
            test_err = 1 - is_correct.mean()
            test_adv_err_lb = 1 - is_robust_attack.mean()
            test_adv_err = 1 - is_robust_exact.mean()
            test_adv_err_ub = 1 - is_rob_ub.mean()

            valid_err = 1 - (y_valid * ensemble.predict(X_valid) > 0).mean()
            valid_adv_err_lb = 1 - (
                ensemble.attack_by_sampling(X_valid, y_valid, eps_eval, n_trials_sampling) > 0.0).mean()
            valid_adv_err = 1 - (ensemble.certify_exact(X_valid, y_valid, eps_eval) > 0.0).mean()
            valid_adv_err_ub = 1 - (ensemble.certify_treewise_bound(X_valid, y_valid, eps_eval) > 0.0).mean()

            train_err = np.mean(Fx_y <= 0)  # important to have <= since when lr->0, all preds = 0
            train_adv_err = np.mean(min_Fx_y_exact <= 0)
            train_loss = np.mean(gamma)

            time_elapsed = time.time() - time_start

            # Various sanity checks
            # print('max diff yf', np.max(np.abs(ensemble.certify_exact(X_test, y_test, args.eps_eval) -
            #                             ensemble.certify_treewise_bound(X_test, y_test, args.eps_eval))))
            if np.sum(is_correct < is_robust_attack) > 0:
                log.print('Number pts violated correct < attack: {} ({})'.format(
                    np.sum(is_correct < is_robust_attack), np.where(is_correct < is_robust_attack)[0]))
            if np.sum(is_robust_attack < is_robust_exact) > 0:
                log.print('Number pts violated attack < exact: {} ({})'.format(
                    np.sum(is_robust_attack < is_robust_exact), np.where(is_robust_attack < is_robust_exact)[0]))
            if np.sum(is_robust_exact < is_rob_ub) > 0:
                log.print('Number pts violated exact < rob_ub: {} ({})'.format(
                    np.sum(is_robust_exact < is_rob_ub), np.where(is_robust_exact < is_rob_ub)[0]))
            if it > 1 and train_loss > metrics[-1][7] + 1e-7:
                log.print('The train loss increases: prev {:.5f} now {:.5f}'.format(metrics[-1][7], train_loss))
            # log.print('New {}'.format(ensemble.trees[-1]))
            # coord_lengths = [(coord, len(ensemble.coords_trees[coord])) for coord in ensemble.coords_trees]
            # coord_lengths = sorted(coord_lengths, key=lambda t: t[1], reverse=True)
            # coord_most_freq = coord_lengths[0][0]
            # print('Most frequent coord {} ({} times) {}'.format(
            #     coord_most_freq, len(most_freq_trees), most_freq_trees))
            str_adv_err = 'adv_err {:.2%} '.format(test_adv_err) if ensemble.weak_learner == 'stump' else ''
            test_str = 'iter: {}  [test] err {:.2%} adv_err_lb {:.2%} {}adv_err_ub {:.2%}'.format(
                it, test_err, test_adv_err_lb, str_adv_err, test_adv_err_ub)
            valid_str = '[valid] err {:.2%} adv_err {:.2%}'.format(valid_err, valid_adv_err)
            train_str = '[train] err {:.2%} adv_err {:.2%} loss {:.5f}'.format(
                train_err, train_adv_err, train_loss)
            log.print('{} | {} | {} ({:.2f} sec)'.format(test_str, valid_str, train_str, time_elapsed))

            metrics.append([it, test_err, test_adv_err_lb, test_adv_err, test_adv_err_ub, train_err, train_adv_err,
                            train_loss, valid_err, valid_adv_err_lb, valid_adv_err, valid_adv_err_ub, time_elapsed])

        if (it % 5 == 0 or it == n_trees) and metrics_path != '':
            ensemble.save(model_path)
            np.savetxt(metrics_path, metrics)

    log.print('(done in {:.2f} min)'.format((time.time() - time_start) / 60))


if __name__ == '__main__':
    np.random.seed(1)
    np.set_printoptions(precision=10)

    # Example: python train.py --dataset=fmnist_sandal_sneaker --weak_learner=stump --model=robust_exact
    parser = argparse.ArgumentParser(description='Define hyperparameters.')
    parser.add_argument('--dataset', type=str, default='mnist_2_6',
                        help='breast_cancer, diabetes, cod_rna, mnist_2_6, fmnist_sandal_sneaker, gts_30_70, '
                             'gts_100_roadworks')
    parser.add_argument('--model', type=str, default='robust_exact', help='plain, robust_exact or robust_bound.')
    parser.add_argument('--weak_learner', type=str, default='stump', help='stump or tree')
    parser.add_argument('--n_train', type=int, default=-1, help='Number of training points to take.')
    args = parser.parse_args()

    # always 1.0 in all experiments
    lr = 1.0
    if args.weak_learner == 'stump':
        n_iter = 500
        n_trials_coord = 10
    elif args.weak_learner == 'tree':
        n_iter = 50
        n_trials_coord = 100
    else:
        raise ValueError('wrong weak learner')

    min_samples_split = 10  # to prevent extreme overfitting to a few points
    min_samples_leaf = 5
    max_depth = 4

    X_train, y_train, X_test, y_test, eps_dataset = data.all_datasets_dict[args.dataset]()
    X_train, y_train, X_valid, y_valid = data.split_train_validation(X_train, y_train, shuffle=True)
    if args.n_train != -1:
        X_train, y_train = X_train[:args.n_train], y_train[:args.n_train]
    eps_train = eps_dataset if args.model != 'plain' else 0.0  # not strictly needed, but just for consistency

    cur_timestamp = str(datetime.now())[:-7]
    hps_str_full = 'dataset={} weak_learner={} model={} n_train={} n_trials_coord={} eps={:.3f} min_samples_split={} min_samples_leaf={} ' \
                   'max_depth={} lr={}'.format(args.dataset, args.weak_learner, args.model, args.n_train, n_trials_coord,
                                               eps_dataset, min_samples_split, min_samples_leaf, max_depth, lr)
    hps_str_short = 'dataset={} weak_learner={} model={} n_train={} n_trials_coord={} eps={:.3f} max_depth={} lr={}'.format(
        args.dataset, args.weak_learner, args.model, args.n_train, n_trials_coord, eps_dataset, max_depth, lr)

    log_path = 'exps/{} {}.log'.format(cur_timestamp, hps_str_short)
    model_path = 'exps/{} {}.model'.format(cur_timestamp, hps_str_short)
    metrics_path = 'exps/{} {}.metrics'.format(cur_timestamp, hps_str_short)

    log = Logger(log_path)
    log.print('Boosting started: {} {}'.format(cur_timestamp, hps_str_full))

    if args.weak_learner == 'stump':
        ensemble = StumpEnsemble(args.weak_learner, n_trials_coord, lr)
    elif args.weak_learner == 'tree':
        ensemble = TreeEnsemble(args.weak_learner, n_trials_coord, lr, min_samples_split, min_samples_leaf, max_depth)
    else:
        raise ValueError('wrong weak learner')

    robust_boost(ensemble, X_train, y_train, X_valid, y_valid, X_test, y_test, n_iter, eps_train, eps_dataset, args.model, log, model_path, metrics_path)


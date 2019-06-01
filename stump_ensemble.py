import numpy as np
import ipdb as pdb
from numba import jit, prange
from collections import OrderedDict
from robust_boosting import exp_loss_robust, coord_descent_exp_loss, bisect_coord_descent, calc_h, \
    basic_case_two_intervals, dtype
from utils import minimum, get_contiguous_indices


class Stump:
    def __init__(self, w_l, w_r, b, coord):
        self.w_l, self.w_r, self.b, self.coord = w_l, w_r, b, coord
        self.left, self.right = None, None

    def predict(self, X):
        value = self.w_l + self.w_r * (X[:, self.coord] >= self.b)
        return value

    def find_min_yf(self, X, y, eps):
        split_lbs, split_ubs = X[:, self.coord] - eps, X[:, self.coord] + eps
        lval, rval = self.w_l, self.w_r + self.w_l

        # Fast vectorized version
        guaranteed_left = split_ubs < self.b
        guaranteed_right = split_lbs > self.b
        uncertain = (split_lbs <= self.b) * (split_ubs >= self.b)

        lbs = y*lval * guaranteed_left + y*rval * guaranteed_right + np.minimum(y*lval, y*rval) * uncertain

        return lbs

    def __repr__(self):
        lval, rval, threshold = self.w_l, self.w_r + self.w_l, self.b
        return 'Tree: if x[{}] < {:.4f}: {:.4f} else {:.4f}'.format(self.coord, lval, threshold, rval)


class StumpEnsemble:
    def __init__(self, weak_learner, n_trials_coord, lr):
        self.weak_learner = weak_learner
        self.n_trials_coord = n_trials_coord
        self.lr = lr
        self.trees = []
        self.coords_trees = OrderedDict()

    def __repr__(self):
        sorted_trees = sorted(self.trees, key=lambda tree: tree.coord)
        return '\n'.join([str(t) for t in sorted_trees])

    def load(self, path, iteration=-1):
        if iteration == -1:  # take all
            ensemble_arr = np.loadtxt(path)
        else:  # take up to some iteration
            ensemble_arr = np.loadtxt(path)[:iteration+1]
        for i in range(ensemble_arr.shape[0]):
            w_l, w_r, b, coord = ensemble_arr[i, :]
            coord = int(coord)
            tree = Stump(w_l, w_r, b, coord)
            self.add_weak_learner(tree)
        print('Ensemble of {} learners restored: {}'.format(ensemble_arr.shape[0], path))

    def save(self, path):
        if path != '':
            ensemble_arr = np.zeros([len(self.trees), 4])
            for i, tree in enumerate(self.trees):
                ensemble_arr[i, :] = [tree.w_l, tree.w_r, tree.b, tree.coord]
            np.savetxt(path, ensemble_arr)

    def add_weak_learner(self, tree):
        tree.w_l, tree.w_r = tree.w_l*self.lr, tree.w_r*self.lr
        self.trees.append(tree)
        if tree.coord not in self.coords_trees:
            self.coords_trees[tree.coord] = []
        self.coords_trees[tree.coord].append(tree)
        print(tree)

    def predict(self, X):
        Fx = np.zeros(X.shape[0])
        for tree in self.trees:
            Fx += tree.predict(X)
        return Fx

    def attack_by_sampling(self, X, y, eps, n_trials):
        """ A simple attack just by sampling in the Linf-box around the points. More of a sanity check. """
        num, dim = X.shape
        f_x_vals = np.zeros((num, n_trials))
        # Note: for efficiency, we sample the same random direction for all points, but this does influence matter
        deltas = np.random.uniform(-eps, eps, size=(dim, n_trials))
        for i in range(n_trials-1):
            # let's keep them as real images, although not strictly needed
            perturbed_pts = np.clip(X + deltas[:, i], 0.0, 1.0)
            f_x_vals[:, i] = self.predict(perturbed_pts)
        # maybe in some corner cases, the predictions at the original point is more worst-case than the sampled points
        f_x_vals[:, n_trials-1] = self.predict(X)

        f_x_min = np.min(y[:, None] * f_x_vals, axis=1)
        return f_x_min

    def certify_treewise_bound(self, X, y, eps):
        lb_ensemble = np.zeros(X.shape[0])

        # The naive tree-wise bounded on the merged trees
        for tree in self.trees:
            lb_ensemble += tree.find_min_yf(X, y, eps)
        return lb_ensemble

    @staticmethod
    @jit(nopython=True)
    def find_min_coord_diff(X_proj, y, thresholds, w_r_values, eps):
        # parallel=True doesn't help here; not sure if jit here is helpful at all. maybe if there are many thresholds
        num = X_proj.shape[0]
        idx = np.argsort(thresholds)
        sorted_thresholds = thresholds[idx]
        sorted_w_r = w_r_values[idx]
        f_x_min_coord_diff, f_x_cumsum = np.zeros(num), np.zeros(num)
        for i_t in range(len(sorted_thresholds)):
            # consider the threshold if it belongs to (x-eps, x+eps] (x-eps is excluded since already evaluated)
            idx_x_eps_close_to_threshold = (X_proj - eps < sorted_thresholds[i_t]) * (sorted_thresholds[i_t] <= X_proj + eps)
            f_diff = y * sorted_w_r[i_t] * idx_x_eps_close_to_threshold
            f_x_cumsum += f_diff
            f_x_min_coord_diff = minimum(f_x_cumsum, f_x_min_coord_diff)
        return f_x_min_coord_diff

    def certify_exact(self, X, y, eps, coords_to_ignore=()):
        # Idea: iterate over all thresholds, and then check if they are in (x-eps, x+eps]
        num, dim = X.shape
        f_x_min = np.zeros(num)

        # Fast, vectorized version
        for coord in self.coords_trees.keys():
            if coord in coords_to_ignore:
                continue
            trees_current_coord = self.coords_trees[coord]

            f_x_min_coord_base = np.zeros(num)
            thresholds, w_r_values = np.zeros(len(trees_current_coord)), np.zeros(len(trees_current_coord))
            for i in range(len(trees_current_coord)):
                tree = trees_current_coord[i]
                f_x_min_coord_base += y * tree.predict(X - eps)
                thresholds[i], w_r_values[i] = tree.b, tree.w_r

            f_x_min += f_x_min_coord_base + self.find_min_coord_diff(X[:, coord], y, thresholds, w_r_values, eps)
        return f_x_min

    def exact_adv_example(self, X, y):
        min_val = 1e-7
        num, dim = X.shape
        deltas = np.zeros([num, dim])
        db_dists = np.full(num, np.inf)

        for i in range(num):
            # 0.0 means we just check whether the point is originally misclassified; if yes  =>  db_dist=0
            eps_all_i = np.array([0.0] + [np.abs(tree.b - X[i, tree.coord] + min_val*np.sign(tree.b - X[i, tree.coord]))
                                          for tree in self.trees])
            eps_sorted = np.sort(eps_all_i)
            for eps in eps_sorted:
                # Vectorized but obscure version; just a sanity check for eps; doesn't return deltas
                # f_x_min = self.certify_exact(X[None, i], y[None, i], eps)

                # Clear unvectorized version
                yf_min = 0.0
                delta = np.zeros(dim)
                for coord in self.coords_trees.keys():
                    trees_current_coord = self.coords_trees[coord]

                    yf_min_coord_base, yf_orig_pt = 0.0, 0.0
                    for tree in trees_current_coord:
                        yf_min_coord_base += y[i] * tree.predict(X[None, i] - eps)
                        yf_orig_pt += y[i] * tree.predict(X[None, i])

                    unstable_thresholds, unstable_wr_values = [X[i, coord] - eps], [0.0]
                    for tree in trees_current_coord:
                        # excluding the left equality since we have already evaluated it
                        if X[i, coord] - eps < tree.b <= X[i, coord] + eps:
                            unstable_thresholds.append(tree.b)
                            unstable_wr_values.append(tree.w_r)
                    unstable_thresholds = np.array(unstable_thresholds)
                    unstable_wr_values = np.array(unstable_wr_values)
                    idx = np.argsort(unstable_thresholds)
                    unstable_thresholds = unstable_thresholds[idx]

                    sorted_y_wr = (y[i] * np.array(unstable_wr_values))[idx]
                    yf_coord_interval_vals = np.cumsum(sorted_y_wr)
                    yf_min_coord = yf_min_coord_base + yf_coord_interval_vals.min()
                    yf_min += yf_min_coord

                    i_opt_threshold = yf_coord_interval_vals.argmin()
                    # if the min value is attained at the point itself, take it instead; so that we do not take
                    # unnecessary -eps deltas (which would not anyway influence Linf size, but would bias the picture)
                    if yf_min_coord == yf_orig_pt:
                        opt_threshold = X[i, coord]  # i.e. the final delta is 0.0
                    else:
                        opt_threshold = unstable_thresholds[i_opt_threshold]
                    delta[coord] = opt_threshold - X[i, coord]

                x_adv_clipped = np.clip(X[i] + delta, 0, 1)  # make sure that the images are valid
                delta = x_adv_clipped - X[i]

                yf = float(y[i] * self.predict(X[None, i] + delta[None]))
                print('eps_max={:.3f}, eps_delta={:.3f}, yf={:.3f}, nnz={}'.format(
                    eps, np.abs(delta).max(), yf, (delta != 0.0).sum()))
                if yf_min < 0:
                    db_dists[i] = eps
                    deltas[i] = delta
                    break
            print()
            yf = y[i] * self.predict(X[None, i] + deltas[None, i])
            if yf >= 0.0:
                print('The class was not changed! Some bug!')
                import ipdb;ipdb.set_trace()
        return deltas

    @staticmethod
    @jit(nopython=True, parallel=True)  # parallel=True really matters, especially with independent iterations
    def fit_plain_stumps(X_proj, y, gamma, b_vals):
        n_thresholds = b_vals.shape[0]

        losses = np.full(n_thresholds, np.inf, dtype=dtype)
        w_l_vals = np.full(n_thresholds, np.inf, dtype=dtype)
        w_r_vals = np.full(n_thresholds, np.inf, dtype=dtype)
        sum_1, sum_m1 = np.sum((y == 1) * gamma), np.sum((y == -1) * gamma)
        for i in prange(n_thresholds):
            ind = X_proj >= b_vals[i]

            sum_1_1, sum_1_m1 = np.sum(ind * (y == 1) * gamma), np.sum(ind * (y == -1) * gamma)
            sum_0_1, sum_0_m1 = sum_1 - sum_1_1, sum_m1 - sum_1_m1
            w_l, w_r = coord_descent_exp_loss(sum_1_1, sum_1_m1, sum_0_1, sum_0_m1)

            fmargin = y*w_l + y*w_r*ind
            loss = np.mean(gamma * np.exp(-fmargin))
            losses[i], w_l_vals[i], w_r_vals[i] = loss, w_l, w_r

        return losses, w_l_vals, w_r_vals, b_vals

    @staticmethod
    @jit(nopython=True, parallel=True)  # parallel=True really matters, especially with independent iterations
    def fit_robust_bound_stumps(X_proj, y, gamma, b_vals, eps):
        n_thresholds = b_vals.shape[0]

        losses = np.full(n_thresholds, np.inf, dtype=dtype)
        w_l_vals = np.full(n_thresholds, np.inf, dtype=dtype)
        w_r_vals = np.full(n_thresholds, np.inf, dtype=dtype)
        sum_1, sum_m1 = np.sum((y == 1) * gamma), np.sum((y == -1) * gamma)
        for i in prange(n_thresholds):
            # Certification for the previous ensemble O(n)
            split_lbs, split_ubs = X_proj - eps, X_proj + eps
            guaranteed_right = split_lbs > b_vals[i]
            uncertain = (split_lbs <= b_vals[i]) * (split_ubs >= b_vals[i])

            loss, w_l, w_r = basic_case_two_intervals(y, gamma, guaranteed_right, uncertain, sum_1, sum_m1)
            losses[i], w_l_vals[i], w_r_vals[i] = loss, w_l, w_r

        return losses, w_l_vals, w_r_vals, b_vals

    @staticmethod
    @jit(nopython=True, parallel=True)  # parallel=True really matters, especially with independent iterations
    def fit_robust_exact_stumps(X_proj, y, gamma, b_vals, eps, w_rs, bs):
        n_thresholds = b_vals.shape[0]

        losses = np.full(n_thresholds, np.inf, dtype=dtype)
        w_l_vals = np.full(n_thresholds, np.inf, dtype=dtype)
        w_r_vals = np.full(n_thresholds, np.inf, dtype=dtype)
        sum_1, sum_m1 = np.sum((y == 1) * gamma), np.sum((y == -1) * gamma)
        for i in prange(n_thresholds):
            # Certification for the previous ensemble O(n)
            split_lbs, split_ubs = X_proj - eps, X_proj + eps
            guaranteed_right = split_lbs > b_vals[i]
            uncertain = (split_lbs <= b_vals[i]) * (split_ubs >= b_vals[i])

            h_l, h_r = calc_h(X_proj, y, w_rs, bs, b_vals[i], eps)
            # there should be quite many useless coordinates which do not have any stumps in the ensemble
            # thus h_l=h_r=0  =>  suffices to check just 2 regions without applying bisection
            if np.sum(h_l) == 0.0 and np.sum(h_r) == 0.0:
                loss, w_l, w_r = basic_case_two_intervals(y, gamma, guaranteed_right, uncertain, sum_1, sum_m1)
            else:  # general case; happens only when `coord` was already splitted in the previous iterations
                loss, w_l, w_r = bisect_coord_descent(y, gamma, h_l, h_r, guaranteed_right, uncertain)

            losses[i], w_l_vals[i], w_r_vals[i] = loss, w_l, w_r

        return losses, w_l_vals, w_r_vals, b_vals

    def fit_stump(self, X, y, gamma_global, model, eps):
        n_trials_coord = self.n_trials_coord
        X, y, gamma_global = X.astype(dtype), y.astype(dtype), gamma_global.astype(dtype)

        num, dim = X.shape
        params, min_losses = np.zeros((n_trials_coord, 4)), np.full(n_trials_coord, np.inf)

        # 151 features are always 0.0 on MNIST 2 vs 6. Doesn't even makes sense to consider them.
        idx_non_trivial = np.abs(X).sum(axis=0) > 0.0
        features_to_check = list(np.arange(dim)[idx_non_trivial])
        np.random.shuffle(features_to_check)  # shuffles in-place
        for trial in prange(n_trials_coord):
            if len(features_to_check) > 0:
                coord = features_to_check.pop()  # takes the last element
            else:
                self.n_trials_coord = trial
                break
            X_proj = X[:, coord]

            # Needed for exact robust optimization with stumps
            trees_current_coord = self.coords_trees[coord] if coord in self.coords_trees else []
            w_rs, bs = np.zeros(len(trees_current_coord)), np.zeros(len(trees_current_coord))
            for i in range(len(trees_current_coord)):
                w_rs[i] = trees_current_coord[i].w_r
                bs[i] = trees_current_coord[i].b

            if model == 'robust_exact' and trees_current_coord != []:  # note: the previous gamma is just ignored
                min_Fx_y_exact_without_j = self.certify_exact(X, y, eps, coords_to_ignore=(coord, ))
                w_ls = np.sum([tree.w_l for tree in trees_current_coord])
                gamma = np.exp(-min_Fx_y_exact_without_j - y*w_ls)
            else:
                gamma = gamma_global

            min_val = 1e-7
            if model not in ['robust_exact', 'robust_bound'] or eps == 0.0:  # plain training
                b_vals = np.copy(X_proj)
                b_vals += min_val  # to break the ties
            else:  # robust training
                b_vals = np.concatenate((X_proj - eps, X_proj + eps), axis=0)  # 2n thresholds
                # to make in the overlapping case |---x-|--|-x---| output 2 different losses in the middle
                b_vals += np.concatenate((-np.full(num, min_val), np.full(num, min_val)), axis=0)
            b_vals = np.unique(b_vals)  # use only unique b's
            b_vals = np.sort(b_vals)  # still important to sort because of the final threshold selection

            if model == 'plain':
                losses, w_l_vals, w_r_vals, b_vals = self.fit_plain_stumps(X_proj, y, gamma, b_vals)
            elif model == 'robust_bound':
                losses, w_l_vals, w_r_vals, b_vals = self.fit_robust_bound_stumps(X_proj, y, gamma, b_vals, eps)
            elif model == 'robust_exact':
                losses, w_l_vals, w_r_vals, b_vals = self.fit_robust_exact_stumps(X_proj, y, gamma, b_vals, eps, w_rs, bs)
            else:
                raise ValueError('wrong model')

            min_loss = np.min(losses)
            # probably, they are already sorted, but to be 100% sure since it is not explicitly mentioned in the docs
            indices_opt_init = np.sort(np.where(losses == min_loss)[0])
            indices_opt = get_contiguous_indices(indices_opt_init)
            id_opt = indices_opt[len(indices_opt) // 2]

            idx_prev = np.clip(indices_opt[0]-1, 0, len(b_vals)-1)  # to prevent stepping out of the array
            idx_next = np.clip(indices_opt[-1]+1, 0, len(b_vals)-1)  # to prevent stepping out of the array
            b_prev, w_l_prev, w_r_prev = b_vals[idx_prev], w_l_vals[idx_prev], w_r_vals[idx_prev]
            b_next, w_l_next, w_r_next = b_vals[idx_next], w_l_vals[idx_next], w_r_vals[idx_next]
            # initialization
            b_leftmost, b_rightmost = b_vals[indices_opt[0]], b_vals[indices_opt[-1]]
            # more involved, since with +-eps, an additional check of the loss is needed
            if model == 'plain':
                b_rightmost = b_next
            elif model in ['robust_bound', 'robust_exact']:
                h_flag = False if model == 'robust_bound' else True

                b_prev_half = (b_prev + b_vals[indices_opt[0]]) / 2
                loss_prev_half = exp_loss_robust(X_proj, y, gamma, w_l_prev, w_r_prev, w_rs, bs, b_prev_half, eps, h_flag)

                b_next_half = (b_vals[indices_opt[-1]] + b_next) / 2
                loss_next_half = exp_loss_robust(X_proj, y, gamma, w_l_next, w_r_next, w_rs, bs, b_next_half, eps, h_flag)

                # we extend the interval of the constant loss to the left and to the right if there the loss is
                # the same at b_prev_half or b_next_half
                if loss_prev_half == losses[id_opt]:
                    b_leftmost = b_prev
                if loss_next_half == losses[id_opt]:
                    b_rightmost = b_next
            else:
                raise ValueError('wrong model')
            # we put in the middle of the interval of the constant loss
            b_opt = (b_leftmost + b_rightmost) / 2

            # For the chosen threshold, we need to calculate w_l, w_r
            # Some of w_l, w_r that correspond to min_loss may not be optimal anymore
            b_val_final = np.array([b_opt])
            if model == 'plain':
                loss, w_l_opt, w_r_opt, _ = self.fit_plain_stumps(X_proj, y, gamma, b_val_final)
            elif model == 'robust_bound':
                loss, w_l_opt, w_r_opt, _ = self.fit_robust_bound_stumps(X_proj, y, gamma, b_val_final, eps)
            elif model == 'robust_exact':
                loss, w_l_opt, w_r_opt, _ = self.fit_robust_exact_stumps(X_proj, y, gamma, b_val_final, eps, w_rs, bs)
            else:
                raise ValueError('wrong model')
            loss, w_l_opt, w_r_opt = loss[0], w_l_opt[0], w_r_opt[0]
            # recalculation of w_l, w_r shouldn't change the min loss

            if np.abs(loss - min_loss) > 1e7:
                print('New loss: {:.5f}, min loss before: {:.5f}'.format(loss, min_loss))

            min_losses[trial] = losses[id_opt]
            params[trial, :] = [w_l_opt, w_r_opt, b_opt, coord]

        id_best_coord = min_losses[:n_trials_coord].argmin()
        best_coord = int(params[id_best_coord][3])  # float to int is necessary for a coordinate
        w_l, w_r, b, coord = params[id_best_coord][0], params[id_best_coord][1], params[id_best_coord][2], best_coord
        stump = Stump(w_l, w_r, b, coord)
        return stump


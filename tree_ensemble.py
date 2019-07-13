import numpy as np
import ipdb as pdb
import copy
from numba import jit, prange
from collections import OrderedDict
from robust_boosting import exp_loss_robust, coord_descent_exp_loss, basic_case_two_intervals, dtype
from utils import minimum, get_contiguous_indices


class Tree:
    def __init__(self, id_, left, right, w_l, w_r, b, coord, loss):
        # (left == None and right == None)  =>  leaf
        # else  =>  intermediate node
        self.id, self.left, self.right = id_, left, right
        # Note: w_l/w_r can have some values, but if left AND right is not None, then w_l/w_r are just ignored.
        # However, we still may need them because of pruning - if a leaf node was pruned, then its parent kicks in.
        self.w_l, self.w_r, self.b, self.coord, self.loss = w_l, w_r, b, coord, loss

    def __repr__(self):
        lval, rval, threshold = self.w_l, self.w_r + self.w_l, self.b

        if self.left is None and self.right is None:
            return 'Tree: if x[{}] < {:.4f}: {:.4f} else {:.4f}    '.format(self.coord, threshold, lval, rval)
        if self.left is None:
            return 'Tree: if x[{}] < {:.4f}: {:.4f}  '.format(self.coord, threshold, lval) + self.right.__repr__()
        if self.right is None:
            return self.left.__repr__() + 'Tree: if x[{}] >= {:.4f}: {:.4f}  '.format(self.coord, threshold, rval)

        s = ''
        if self.left is not None:
            s += self.left.__repr__()
        if self.right is not None:
            s += self.right.__repr__()

        return s

    def __eq__(self, other):
        """ Overrides the default equality comparison operator == """
        if isinstance(other, Tree):
            return (self.left == other.left and self.right == other.right and self.w_l == other.w_l and
                    self.w_r == other.w_r and self.b == other.b and self.coord == other.coord)
        return False

    def to_list(self):
        tree_lst_left, tree_lst_right = [], []
        id_left, id_right = -1, -1
        if self.left is not None:
            tree_lst_left = self.left.to_list()
            id_left = self.left.id
        if self.right is not None:
            tree_lst_right = self.right.to_list()
            id_right = self.right.id
        curr_node = (self.id, id_left, id_right, self.w_l, self.w_r, self.b, self.coord, self.loss)
        return [curr_node] + tree_lst_left + tree_lst_right  # concatenate both lists

    def predict(self, X):
        f = np.zeros(X.shape[0])

        # route some points to the left and some to the right nodes
        idx_left = X[:, self.coord] < self.b
        if self.left is None:
            f[idx_left] = self.w_l
        else:
            f[idx_left] = self.left.predict(X[idx_left])

        idx_right = X[:, self.coord] >= self.b
        if self.right is None:
            f[idx_right] = self.w_l + self.w_r
        else:
            f[idx_right] = self.right.predict(X[idx_right])
        return f

    def get_some_leaf(self):
        if self.left is None and self.right is None:
            return self
        if self.left is not None:
            return self.left.get_some_leaf()
        if self.right is not None:
            return self.right.get_some_leaf()

    def get_some_leaf_except(self, checked_leaves):
        if self.left is None and self.right is None:
            if self not in checked_leaves:
                return self
        if self.left is not None:
            some_left_leaf = self.left.get_some_leaf_except(checked_leaves)
            if some_left_leaf not in checked_leaves and some_left_leaf is not None:
                return some_left_leaf
        if self.right is not None:
            some_right_leaf = self.right.get_some_leaf_except(checked_leaves)
            if some_right_leaf not in checked_leaves and some_right_leaf is not None:
                return some_right_leaf
        # None should be returned only in the end after the whole tree is checked
        return None

    def rm_leaf(self, leaf_to_rm):
        if self.left == leaf_to_rm:
            self.left = None
        if self.right == leaf_to_rm:
            self.right = None

        left_first = np.random.choice([False, True])
        if left_first:
            if self.left is not None:
                self.left.rm_leaf(leaf_to_rm)
            if self.right is not None:
                self.right.rm_leaf(leaf_to_rm)
        else:
            if self.right is not None:
                self.right.rm_leaf(leaf_to_rm)
            if self.left is not None:
                self.left.rm_leaf(leaf_to_rm)

    def get_empty_leaf(self):
        if self.left is not None:
            return self.left.get_empty_leaf()
        if self.right is not None:
            return self.right.get_empty_leaf()
        if self.left is None and self.right is None and self.w_l == 0.0 and self.w_r == 0.0:
            return self

    def find_min_yf(self, X, y, eps):
        split_lbs, split_ubs = X[:, self.coord] - eps, X[:, self.coord] + eps
        lval, rval = self.w_l, self.w_r + self.w_l

        guaranteed_left = split_ubs < self.b
        guaranteed_right = split_lbs > self.b
        uncertain = (split_lbs <= self.b) * (split_ubs >= self.b)

        if self.left is None:
            left_min_yf = y[guaranteed_left]*lval
            uleft_min_yf = y[uncertain]*lval
        else:
            left_min_yf = self.left.find_min_yf(X[guaranteed_left], y[guaranteed_left], eps)
            uleft_min_yf = self.left.find_min_yf(X[uncertain], y[uncertain], eps)

        if self.right is None:
            right_min_yf = y[guaranteed_right]*rval
            uright_min_yf = y[uncertain]*rval
        else:
            right_min_yf = self.right.find_min_yf(X[guaranteed_right], y[guaranteed_right], eps)
            uright_min_yf = self.right.find_min_yf(X[uncertain], y[uncertain], eps)

        min_yf = np.zeros(X.shape[0])
        min_yf[guaranteed_left] = left_min_yf
        min_yf[guaranteed_right] = right_min_yf
        min_yf[uncertain] = np.minimum(uleft_min_yf, uright_min_yf)

        return min_yf


class TreeEnsemble:
    def __init__(self, weak_learner, n_trials_coord, lr, min_samples_split, min_samples_leaf, max_depth):
        self.weak_learner = weak_learner
        self.n_trials_coord = n_trials_coord
        self.lr = lr
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.trees = []
        self.max_tree_node_id = 0
        self.coords_trees = OrderedDict()

    def __repr__(self):
        sorted_trees = sorted(self.trees, key=lambda tree: tree.coord)
        return '\n'.join([str(t) for t in sorted_trees])

    def load(self, path, iteration=-1):
        if iteration == -1:  # take all
            ensemble_arr = np.load(path)
        else:  # take up to some iteration
            ensemble_arr = np.load(path)[:iteration+1]
        for i in range(ensemble_arr.shape[0]):
            # first create all tree nodes and maintain a dictionary with all nodes (for easier look-up later on)
            node_dict = {}
            for i_node in range(len(ensemble_arr[i])):
                if not np.all(ensemble_arr[i][i_node] == 0):
                    id_, id_left, id_right, w_l, w_r, b, coord, loss = ensemble_arr[i, i_node]
                    id_, id_left, id_right, coord = int(id_), int(id_left), int(id_right), int(coord)
                    # create a node, but without any connections to its children
                    tree = Tree(id_, None, None, w_l, w_r, b, coord, loss)
                    node_dict[id_] = (tree, id_left, id_right)
            # then establish the right connections between the nodes of the tree
            for node in node_dict:
                tree, id_left, id_right = node_dict[node]
                if id_left != -1:
                    tree.left = node_dict[id_left][0]
                if id_right != -1:
                    tree.right = node_dict[id_right][0]
            # add the root as the next element of the ensemble
            root = node_dict[ensemble_arr[i][0][0]][0]
            self.add_weak_learner(root, apply_lr=False)
        print('Ensemble of {} learners restored: {}'.format(ensemble_arr.shape[0], path))

    def save(self, path):
        if path != '':
            # note: every tree has potentially a different number of nodes.
            ensemble_arr = np.zeros([len(self.trees), 2**self.max_depth, 8])
            for i, tree in enumerate(self.trees):
                tree_list = tree.to_list()
                ensemble_arr[i, :len(tree_list), :] = tree_list  # all tree nodes are in this list
            np.save(path, ensemble_arr, allow_pickle=False)

    def add_weak_learner(self, tree, apply_lr=True):
        def adjust_lr(tree, lr):
            """ Recursively goes over all node values and scales the weights by a the learning rate. """
            tree.w_l, tree.w_r = tree.w_l*lr, tree.w_r*lr
            if tree.left is not None:
                adjust_lr(tree.left, lr)
            if tree.right is not None:
                adjust_lr(tree.right, lr)
            return tree

        if apply_lr:
            tree = adjust_lr(tree, self.lr)
        self.trees.append(tree)
        if tree.coord not in self.coords_trees:
            self.coords_trees[tree.coord] = []
        self.coords_trees[tree.coord].append(tree)

    def predict(self, X):
        Fx = np.zeros(X.shape[0])
        for tree in self.trees:
            Fx += tree.predict(X)
        return Fx

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
            idx_x_eps_close_to_threshold = (X_proj - eps < sorted_thresholds[i_t]) * (
                    sorted_thresholds[i_t] <= X_proj + eps)
            f_diff = y * sorted_w_r[i_t] * idx_x_eps_close_to_threshold
            f_x_cumsum += f_diff
            f_x_min_coord_diff = minimum(f_x_cumsum, f_x_min_coord_diff)
        return f_x_min_coord_diff

    def attack_by_sampling(self, X, y, eps, n_trials):
        """ A simple attack just by sampling in the Linf-box around the points. More of a sanity check. """
        num, dim = X.shape
        f_x_vals = np.zeros((num, n_trials))
        # Note: for efficiency, we sample the same random direction for all points, but this does influence matter
        deltas = np.random.uniform(-eps, eps, size=(dim, n_trials))
        for i in range(n_trials - 1):
            # let's keep them as real images, although not strictly needed
            perturbed_pts = np.clip(X + deltas[:, i], 0.0, 1.0)
            f_x_vals[:, i] = self.predict(perturbed_pts)
        # maybe in some corner cases, the predictions at the original point is more worst-case than the sampled points
        f_x_vals[:, n_trials - 1] = self.predict(X)

        f_x_min = np.min(y[:, None] * f_x_vals, axis=1)
        return f_x_min

    def certify_exact(self, X, y, eps, coords_to_ignore=()):
        """
        Note: this is clearly not exact certification.
        We do it just to be compatible with robust_boost() function that requires certify_exact() to output some
        meaningful numbers that do not violate the other bounds on the exact minimum of the adversarial opt problem.
        """
        return self.certify_treewise_bound(X, y, eps)

    def fit_tree(self, X, y, gamma, model, eps, depth):
        """Recursive procedure for building a single tree.

        Note: this function belongs to the tree, and not to the ensemble because the ensemble doesn't matter anymore
        once the vector gamma is fixed.
        """
        if depth > self.max_depth:
            return None  # so that tree.left or tree.right is set to None
        if X.shape[0] < self.min_samples_split:
            return None  # so that tree.left or tree.right is set to None
        if (y == -1).all() or (y == 1).all():  # if already pure, don't branch anymore
            return None

        # create a new tree that will become a node (if further splits are needed)
        # or a leaf (if max_depth or min_samples_leaf is reached)
        w_l, w_r, b, coord, loss = self.fit_stump(X, y, gamma, model, eps)
        tree = Tree(self.max_tree_node_id, None, None, w_l, w_r, b, coord, loss)
        self.max_tree_node_id += 1

        if tree.coord == -1:  # no further splits because min_samples_leaf is reached
            return None

        if model == 'plain':
            idx_left = (X[:, tree.coord] < tree.b)
            idx_right = (X[:, tree.coord] >= tree.b)
        elif model == 'robust_bound':
            idx_left = (X[:, tree.coord] < tree.b + eps)
            idx_right = (X[:, tree.coord] >= tree.b - eps)
        else:
            raise ValueError('wrong model type')

        # print("left subtree: {:d} examples".format(np.sum(idx_left)))
        tree.left = self.fit_tree(X[idx_left, :], y[idx_left], gamma[idx_left], model, eps, depth+1)

        # print("right subtree: {:d} examples".format(np.sum(idx_right)))
        tree.right = self.fit_tree(X[idx_right, :], y[idx_right], gamma[idx_right], model, eps, depth+1)

        return tree

    def prune_last_tree(self, X, y, eps, model):
        """Recursive procedure for building a single tree.

        Note: this function belongs to the tree, and not to the ensemble because the ensemble doesn't matter anymore
        once the vector gamma is fixed.
        """
        # The naive tree-wise bounded on trees
        lb_ensemble = np.zeros(X.shape[0])
        for tree in self.trees[:-1]:
            lb_ensemble += tree.find_min_yf(X, y, eps)
        gamma = np.exp(-lb_ensemble)

        best_tree = copy.deepcopy(self.trees[-1])  # copy the whole tree since we will change its leaves
        if model == 'plain':
            best_loss = np.mean(gamma * np.exp(-y*best_tree.predict(X)))
        elif model == 'robust_bound':
            best_loss = np.mean(gamma * np.exp(-best_tree.find_min_yf(X, y, eps)))
        else:
            raise ValueError('wrong model type')
        curr_tree = copy.deepcopy(best_tree)
        while curr_tree.left is not None or curr_tree.right is not None:
            some_leaf = curr_tree.get_some_leaf_except([])
            curr_tree.rm_leaf(some_leaf)
            if model == 'plain':
                loss_pruned = np.mean(gamma * np.exp(-y * curr_tree.predict(X)))
            elif model == 'robust_bound':
                loss_pruned = np.mean(gamma * np.exp(-curr_tree.find_min_yf(X, y, eps)))
            else:
                raise ValueError('wrong model type')
            # print('{:.4f} {:.4f} {}'.format(loss_pruned, best_loss, curr_tree))
            if loss_pruned < best_loss:
                best_loss = loss_pruned
                best_tree = copy.deepcopy(curr_tree)
        # print('best loss: {:.4f}, best tree: {}'.format(best_loss, best_tree))
        self.trees[-1] = best_tree

        # while 1:
        #     some_leaf = curr_tree.get_some_leaf_except(checked_leaves)
        #     if curr_tree.left is None and curr_tree.right is None:
        #         print('break1', curr_tree)
        #         break
        #     else:
        #         # checked_leaves.append(copy.deepcopy(some_leaf))
        #         curr_tree.rm_leaf(some_leaf)
        #
        #         loss_pruned = np.mean(gamma * np.exp(-curr_tree.find_min_yf(X, y, eps)))
        #         print(loss_pruned, best_loss, curr_tree)
        #         if loss_pruned < best_loss:
        #             print('pruned successfully')
        #             self.trees[-1] = last_tree
        #         losses_trees.append((robust_loss_pruned, last_tree))
        #
        #         self.trees[-1] = last_tree
        #         if last_tree.left is None and last_tree.right is None:  # root
        #             print('break2', curr_tree)
        #             pdb.set_trace()
        #             break
        # self.trees[-1] = best_tree

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

            fmargin = y * w_l + y * w_r * ind
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

    def fit_stump(self, X, y, gamma, model, eps):
        n_trials_coord = self.n_trials_coord
        X, y, gamma = X.astype(dtype), y.astype(dtype), gamma.astype(dtype)

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
                n_trials_coord = trial
                break
            X_proj = X[:, coord]

            min_val = 1e-7
            threshold_candidates = np.sort(np.copy(X_proj))
            if self.min_samples_leaf > 0:
                threshold_candidates = threshold_candidates[self.min_samples_leaf:-self.min_samples_leaf]
            if len(threshold_candidates) == 0:  # if no samples left according to min_samples_leaf
                min_losses[trial] = np.inf
                params[trial, :] = [0.0, 0.0, 0.0, -1]
                continue

            if model not in ['robust_bound'] or eps == 0.0:  # plain training
                b_vals = np.copy(threshold_candidates)
                b_vals += min_val  # to break the ties
            else:  # robust training
                b_vals = np.concatenate((threshold_candidates - eps, threshold_candidates + eps), axis=0)
                # to make in the overlapping case |---x-|--|-x---| output 2 different losses in the middle
                n_bs = len(threshold_candidates)
                b_vals += np.concatenate((-np.full(n_bs, min_val), np.full(n_bs, min_val)), axis=0)
            b_vals = np.unique(b_vals)  # use only unique b's
            b_vals = np.sort(b_vals)  # still important to sort because of the final threshold selection

            if model == 'plain':
                losses, w_l_vals, w_r_vals, b_vals = self.fit_plain_stumps(X_proj, y, gamma, b_vals)
            elif model == 'robust_bound':
                losses, w_l_vals, w_r_vals, b_vals = self.fit_robust_bound_stumps(X_proj, y, gamma, b_vals, eps)
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
            elif model in ['robust_bound']:
                b_prev_half = (b_prev + b_vals[indices_opt[0]]) / 2
                loss_prev_half = exp_loss_robust(X_proj, y, gamma, w_l_prev, w_r_prev, [], [], b_prev_half, eps, False)

                b_next_half = (b_vals[indices_opt[-1]] + b_next) / 2
                loss_next_half = exp_loss_robust(X_proj, y, gamma, w_l_next, w_r_next, [], [], b_next_half, eps, False)

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
            # note: now inf can easily happen if e.g. all examples at some subtree are < eps (happens on MNIST)
            # if (losses == np.nan).sum() > 0:
            #     pdb.set_trace()

            # w_l_opt, w_r_opt, b_opt = w_l_vals[id_opt], w_r_vals[id_opt], b_vals[id_opt]

            # For the chosen threshold, we need to calculate w_l, w_r
            # Some of w_l, w_r that correspond to min_loss may not be optimal anymore
            b_val_final = np.array([b_opt])
            if model == 'plain':
                loss, w_l_opt, w_r_opt, _ = self.fit_plain_stumps(X_proj, y, gamma, b_val_final)
            elif model == 'robust_bound':
                loss, w_l_opt, w_r_opt, _ = self.fit_robust_bound_stumps(X_proj, y, gamma, b_val_final, eps)
            else:
                raise ValueError('wrong model')
            loss, w_l_opt, w_r_opt = loss[0], w_l_opt[0], w_r_opt[0]

            # recalculation of w_l, w_r shouldn't change the min loss
            if np.abs(loss - min_loss) > 1e7:
                print('New loss: {:.5f}, min loss before: {:.5f}'.format(loss, min_loss))

            min_losses[trial] = losses[id_opt]
            params[trial, :] = [w_l_opt, w_r_opt, b_opt, coord]

        id_best_coord = min_losses[:n_trials_coord].argmin()
        min_loss = min_losses[id_best_coord]
        best_coord = int(params[id_best_coord][3])  # float to int is necessary for a coordinate
        return params[id_best_coord][0], params[id_best_coord][1], params[id_best_coord][2], best_coord, min_loss


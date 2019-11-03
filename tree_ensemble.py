import numpy as np
import copy
from collections import OrderedDict
from numba import njit, prange
from robust_boosting import exp_loss_robust, dtype, fit_plain_stumps, fit_robust_bound_stumps
from utils import get_contiguous_indices, get_n_proc
from concurrent.futures import ThreadPoolExecutor


@njit(nogil=True)
def find_min_yf_point(nodes, x, y, eps):
    # Every node is: (self.id, id_left, id_right, self.w_l, self.w_r, self.b, self.coord, self.loss)
    node_ids_to_explore = [0]  # root node id
    min_val = np.inf
    while len(node_ids_to_explore) > 0:
        node = nodes[node_ids_to_explore.pop()]
        id_left, id_right, w_l, w_r, b, coord = int(node[1]), int(node[2]), node[3], node[4], node[5], int(node[6])
        if x[coord] <= b + eps:
            if id_left != -1:
                node_ids_to_explore.append(int(nodes[id_left][0]))
            else:
                min_val = min(min_val, y * w_l)
        if x[coord] >= b - eps:
            if id_right != -1:
                node_ids_to_explore.append(int(nodes[id_right][0]))
            else:
                min_val = min(min_val, y * (w_l + w_r))
    return min_val


@njit(parallel=True, nogil=True)
def find_min_yf_tree_par(nodes, X, y, eps):
    # == works as expected only if all numbers are in float32; float32 is the preferred choice due to less memory
    eps = np.float32(eps)
    f = np.zeros(X.shape[0])
    for i in prange(X.shape[0]):
        f[i] = find_min_yf_point(nodes, X[i], y[i], eps)
    return f


@njit(nogil=True)
def predict_point(nodes, x):
    # Every node is: (self.id, id_left, id_right, self.w_l, self.w_r, self.b, self.coord, self.loss)
    node = nodes[0]  # take the root node
    while True:
        id_left, id_right, w_l, w_r, b, coord = int(node[1]), int(node[2]), node[3], node[4], node[5], int(node[6])
        if x[coord] < b:
            if id_left != -1:
                node = nodes[id_left]
            else:
                return w_l
        else:
            if id_right != -1:
                node = nodes[id_right]
            else:
                return w_l + w_r


@njit(parallel=True, nogil=True)
def predict_tree_par(nodes, X):
    f = np.zeros(X.shape[0])
    for i in prange(X.shape[0]):
        f[i] = predict_point(nodes, X[i])
    return f


class Tree:
    def __init__(self, id_=-1, left=None, right=None, w_l=0.0, w_r=0.0, b=0.0, coord=0, loss=0.0):
        # (left == None and right == None)  =>  leaf
        # else  =>  intermediate node
        self.id, self.left, self.right = id_, left, right
        # Note: w_l/w_r can have some values, but if left AND right is not None, then w_l/w_r are just ignored.
        # However, we still may need them because of pruning - if a leaf node was pruned, then its parent kicks in.
        self.w_l, self.w_r, self.b, self.coord, self.loss = w_l, w_r, b, coord, loss
        self.node_list = []

    def __repr__(self):
        lval, rval, threshold = self.w_l, self.w_r + self.w_l, self.b

        if self.left is None and self.right is None:
            return 'if x[{}] < {:.4f}: {:.4f} else {:.4f}    '.format(self.coord, threshold, lval, rval)
        if self.left is None:
            return 'if x[{}] < {:.4f}: {:.4f}  '.format(self.coord, threshold, lval) + self.right.__repr__()
        if self.right is None:
            return self.left.__repr__() + 'if x[{}] >= {:.4f}: {:.4f}  '.format(self.coord, threshold, rval)

        s = ''
        if self.left is not None:
            s += 'if x[{}] < {:.4f} and '.format(self.coord, threshold) + self.left.__repr__()
        if self.right is not None:
            s += 'if x[{}] >= {:.4f} and '.format(self.coord, threshold) + self.right.__repr__()

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

    def to_array_contiguous(self):
        """ Make ids correspond to node positions in the array. """
        nodes = np.array(self.to_list())
        max_node_id = int(nodes[:, 0].max())
        nodes_new = np.zeros([max_node_id+1, len(nodes[0])])
        for node in nodes:
            nodes_new[int(node[0])] = node
        return nodes_new

    def predict(self, X):
        parallel = True
        if parallel and len(self.node_list) > 0:  # 2nd condition is needed to prevent an error in predict_tree_par()
            return predict_tree_par(self.node_list, X)
        else:
            return self.predict_native(X)

    def predict_native(self, X):
        def predict_recursive(curr_tree, idx):
            """ To avoid copying the whole matrix X many times, we use global indices `idx` to directly use
            the single matrix X as a closure variable. The only overhead is that the threshold comparison is done
            for *all* examples.

            Note: the parallel version using numba should be preferred.
            """
            # route some points to the left and some to the right nodes
            idx_left_superset = X[:, curr_tree.coord] < curr_tree.b
            idx_left = idx * idx_left_superset
            idx_right = idx * ~idx_left_superset

            if curr_tree.left is None:
                f[idx_left] = curr_tree.w_l
            else:
                predict_recursive(curr_tree.left, idx_left)

            if curr_tree.right is None:
                f[idx_right] = curr_tree.w_l + curr_tree.w_r
            else:
                predict_recursive(curr_tree.right, idx_right)

        idx = np.full(X.shape[0], True)
        f = np.zeros(len(idx))
        predict_recursive(self, idx)  # modifies the closure variable `f` in-place
        return f

    def find_min_yf(self, X, y, eps):
        parallel = True  # really crucial; 1-2x orders of magnitude speed-up over the native python version
        if parallel and len(self.node_list) > 0:  # 2nd condition is needed to prevent an error in predict_tree_par()
            return find_min_yf_tree_par(self.node_list, X, y, eps)
        else:
            return self.find_min_yf_native(X, y, eps)

    def find_min_yf_native(self, X, y, eps):
        split_lbs, split_ubs = X[:, self.coord] - eps, X[:, self.coord] + eps
        lval, rval = self.w_l, self.w_r + self.w_l

        guaranteed_left = split_ubs < self.b
        guaranteed_right = split_lbs > self.b
        uncertain = (split_lbs <= self.b) * (split_ubs >= self.b)

        if self.left is None:
            left_min_yf = y[guaranteed_left] * lval
            uleft_min_yf = y[uncertain] * lval
        else:
            left_min_yf = self.left.find_min_yf(X[guaranteed_left], y[guaranteed_left], eps)
            uleft_min_yf = self.left.find_min_yf(X[uncertain], y[uncertain], eps)

        if self.right is None:
            right_min_yf = y[guaranteed_right] * rval
            uright_min_yf = y[uncertain] * rval
        else:
            right_min_yf = self.right.find_min_yf(X[guaranteed_right], y[guaranteed_right], eps)
            uright_min_yf = self.right.find_min_yf(X[uncertain], y[uncertain], eps)

        min_yf = np.zeros(X.shape[0])
        min_yf[guaranteed_left] = left_min_yf
        min_yf[guaranteed_right] = right_min_yf
        min_yf[uncertain] = np.minimum(uleft_min_yf, uright_min_yf)

        return min_yf

    def get_n_nodes(self):
        left_n, right_n = 0, 0
        if self.left is not None:
            left_n = self.left.get_n_nodes()
        if self.right is not None:
            right_n = self.right.get_n_nodes()
        subtree_n = left_n + right_n  # n nodes of the subtree rooted at the current node
        return subtree_n + 1  # which means that a decision stump is a tree of depth=1

    def get_depth(self):
        left_depth, right_depth = 0, 0
        if self.left is not None:
            left_depth = self.left.get_depth()
        if self.right is not None:
            right_depth = self.right.get_depth()
        subtree_depth = max(left_depth, right_depth)  # depth of the subtree rooted at the current node
        return subtree_depth + 1  # which means that a decision stump is a tree of depth=1

    def get_some_leaf(self):
        if self.left is None and self.right is None:
            return self
        if self.left is not None:
            return self.left.get_some_leaf()
        if self.right is not None:
            return self.right.get_some_leaf()

    def rm_leaf(self, leaf_to_rm):
        if self.left == leaf_to_rm:
            self.left = None
        if self.right == leaf_to_rm:
            self.right = None

        # Left-first search
        if self.left is not None:
            self.left.rm_leaf(leaf_to_rm)
        if self.right is not None:
            self.right.rm_leaf(leaf_to_rm)

    def rm_bottom_layer(self, depth, max_depth):
        if depth + 1 == max_depth:
            # print('rm a node from depth {} (max_depth={})'.format(depth+1, max_depth))
            self.left = None
            self.right = None
        if self.left is not None:
            self.left.rm_bottom_layer(depth+1, max_depth)
        if self.right is not None:
            self.right.rm_bottom_layer(depth+1, max_depth)

    def get_empty_leaf(self):
        if self.left is not None:
            return self.left.get_empty_leaf()
        if self.right is not None:
            return self.right.get_empty_leaf()
        if self.left is None and self.right is None and self.w_l == 0.0 and self.w_r == 0.0:
            return self

    def get_json_dict(self, counter_terminal_nodes):
        """
        counter_terminal_nodes: needed to assign nodeid's to terminal nodes (negative to prevent collisions)
        """
        precision = 5

        children_list = []
        if self.left is None:
            id_left = counter_terminal_nodes
            counter_terminal_nodes -= 1
            children_list.append({'nodeid': id_left, 'leaf': round(self.w_l, precision)})  # end node
        else:
            id_left = self.left.id
            children, counter_terminal_nodes = self.left.get_json_dict(counter_terminal_nodes)
            children_list.append(children)

        if self.right is None:
            id_right = counter_terminal_nodes
            counter_terminal_nodes -= 1
            children_list.append({'nodeid': id_right, 'leaf': round(self.w_l + self.w_r, precision)})  # end node
        else:
            id_right = self.right.id
            children, counter_terminal_nodes = self.right.get_json_dict(counter_terminal_nodes)
            children_list.append(children)

        tree_dict = {'nodeid': self.id, 'split': 'f' + str(self.coord), 'split_condition': round(self.b, precision),
                     'yes': id_left, 'no': id_right, 'children': children_list}

        return tree_dict, counter_terminal_nodes


class TreeEnsemble:
    def __init__(self, weak_learner, n_trials_coord, lr, min_samples_split, min_samples_leaf, idx_clsf, max_depth,
                 gamma_hp=0.0, n_bins=-1, max_weight=1.0):
        self.weak_learner = weak_learner
        self.n_trials_coord = n_trials_coord
        self.lr = lr
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.gamma_hp = gamma_hp  # depth pruning coefficient
        self.n_bins = n_bins
        self.idx_clsf = idx_clsf  # class index that this ensemble correspond to in the one-vs-all scheme
        self.max_weight = max_weight
        self.trees = []
        self.coords_trees = OrderedDict()
        self.ens_nodes_array = []
        self.max_tree_node_id = 0

    def __repr__(self):
        sorted_trees = sorted(self.trees, key=lambda tree: tree.coord)
        return '\n'.join([str(t) for t in sorted_trees])

    def copy(self):
        ensemble_new = TreeEnsemble(self.weak_learner, self.n_trials_coord, self.lr, self.min_samples_split,
                                    self.min_samples_leaf, self.idx_clsf, self.max_depth, self.gamma_hp, self.n_bins,
                                    self.max_weight)
        for tree in self.trees:
            ensemble_new.add_weak_learner(tree, apply_lr=False)
        return ensemble_new

    def load(self, ensemble_dict, iteration):
        tree_indices = np.sort(list(ensemble_dict.keys()))  # just a list of contiguous indices [0, 1, ..., n_trees]
        if iteration != -1:  # take only the tree ensemble up to a certain iteration
            tree_indices = tree_indices[tree_indices <= iteration]
        for i_tree in tree_indices:
            # first create all tree nodes and maintain a dictionary with all nodes (for easier look-up later on)
            node_dict = {}
            for i_node in range(len(ensemble_dict[i_tree])):
                if not np.all(ensemble_dict[i_tree][i_node] == 0):
                    id_, id_left, id_right, w_l, w_r, b, coord, loss = ensemble_dict[i_tree][i_node]
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
            if ensemble_dict[i_tree] != []:
                root = node_dict[ensemble_dict[i_tree][0][0]][0]
                self.add_weak_learner(root, apply_lr=False)
                root.node_list = root.to_array_contiguous()

    def export_model(self):
        # note: every tree has potentially a different number of nodes, thus we save it in a dictionary
        ensemble_dict = {}
        for i, tree in enumerate(self.trees):
            ensemble_dict[i] = np.array(tree.node_list)  # all tree nodes are in this array
        return ensemble_dict

    def add_weak_learner(self, tree, apply_lr=True):
        def adjust_lr(tree, lr):
            """ Recursively goes over all node values and scales the weights by the learning rate. """
            tree.w_l, tree.w_r = tree.w_l * lr, tree.w_r * lr
            if tree.node_list != []:  # i.e. if root
                for node_tuple in tree.node_list:
                    node_tuple[3], node_tuple[4] = node_tuple[3] * lr, node_tuple[4] * lr
            if tree.left is not None:
                adjust_lr(tree.left, lr)
            if tree.right is not None:
                adjust_lr(tree.right, lr)
            return tree

        if tree is None:  # can happen if no splits whatsoever were made
            tree = Tree()
        if apply_lr:
            tree = adjust_lr(tree, self.lr)
        self.trees.append(tree)

        if tree.coord not in self.coords_trees:
            self.coords_trees[tree.coord] = []
        self.coords_trees[tree.coord].append(tree)

    def predict(self, X):
        f = np.zeros(X.shape[0])
        for tree in self.trees:
            f += tree.predict(X)
        return f

    def certify_treewise(self, X, y, eps):
        lb_ensemble = np.zeros(X.shape[0])
        for tree in self.trees:
            lb_ensemble += tree.find_min_yf(X, y, eps)
        return lb_ensemble

    def prune_last_tree(self, X, y, margin_prev, eps, model):
        """
        Recursive procedure for building a single tree.

        Note: this function belongs to the tree, and not to the ensemble because the ensemble doesn't matter anymore
        once the vector gamma is fixed.
        """
        gamma = np.exp(-margin_prev)
        loss_prev_ensemble = np.mean(gamma)

        best_tree = copy.deepcopy(self.trees[-1])  # copy the whole tree since we will change its leaves
        if model in ['plain', 'da_uniform', 'at_cube']:
            best_loss = np.mean(gamma * np.exp(-y*best_tree.predict(X)))
        elif model == 'robust_bound':
            best_loss = np.mean(gamma * np.exp(-best_tree.find_min_yf(X, y, eps)))
        else:
            raise ValueError('wrong model type')
        best_loss += self.gamma_hp * best_tree.get_depth()  # introduce depth penalization
        if best_loss < loss_prev_ensemble:
            return

        curr_tree = copy.deepcopy(best_tree)
        # stop when best_loss is better than the previous loss or curr_tree became just a stump
        while best_loss >= loss_prev_ensemble and not (curr_tree.left is None and curr_tree.right is None):
            curr_tree.rm_leaf(curr_tree.get_some_leaf())  # gradual pruning
            # curr_tree.rm_bottom_layer(depth=1, max_depth=curr_tree.get_depth())  # agressive pruning
            curr_tree.node_list = curr_tree.to_array_contiguous()
            if model in ['plain', 'da_uniform', 'at_cube']:
                loss_pruned = np.mean(gamma * np.exp(-y * curr_tree.predict(X)))
            elif model == 'robust_bound':
                loss_pruned = np.mean(gamma * np.exp(-curr_tree.find_min_yf(X, y, eps)))
            else:
                raise ValueError('wrong model type')
            loss_pruned += self.gamma_hp * curr_tree.get_depth()  # introduce depth penalization
            # print('{:.4f} {:.4f} {}'.format(loss_pruned, best_loss, curr_tree))
            if loss_pruned < best_loss:
                best_loss = loss_pruned
                best_tree = copy.deepcopy(curr_tree)
        # print('best loss: {:.4f}, best tree: {}'.format(best_loss, best_tree))
        self.trees[-1] = best_tree

    def fit_tree(self, X, y, gamma, model, eps, depth):
        """
        Recursive procedure for building a single tree.
        Returning None means that tree.left or tree.right will be set to None, i.e. no child.

        TODO: the problem currently is that there is a minor memory leak in the current implementation. One can try to
        get rid of it by rewriting this function in a non-recursive way (similarly to, e.g. how predict_point() is done)
        """
        parallel = True  # causes a minor memory leak; disable if the memory is limited

        if depth == 1:
            self.max_tree_node_id = 0  # if we start a new tree, set the counter to 0 (needed for efficient predict())
        if depth > self.max_depth:  # and (X.shape[0] <= 10000 or depth > 2*self.max_depth):  # adaptive depth
            return None
        if X.shape[0] < self.min_samples_split:
            return None
        if (y == -1).all() or (y == 1).all():  # if already pure, don't branch anymore
            return None

        # create a new tree that will become a node (if further splits are needed)
        # or a leaf (if max_depth or min_samples_leaf is reached)
        w_l, w_r, b, coord, loss = self.fit_stumps_over_coords(X, y, gamma, model, eps, depth)

        if coord == -1:  # no further splits because min_samples_leaf is reached
            return None
        if loss >= np.mean(gamma):  # if the stump doesn't help, don't add it at all; very unlikely situation
            # print('Did not make this split since old_loss={:.4} <= new_loss={:.4}'.format(np.mean(gamma), loss))
            return None

        tree = Tree(self.max_tree_node_id, None, None, w_l, w_r, b, coord, loss)
        self.max_tree_node_id += 1  # increment the counter

        if model in ['plain', 'da_uniform', 'at_cube']:
            idx_left = (X[:, tree.coord] < tree.b)
            idx_right = (X[:, tree.coord] >= tree.b)
        elif model == 'robust_bound':
            idx_left = (X[:, tree.coord] < tree.b + eps)
            idx_right = (X[:, tree.coord] >= tree.b - eps)
        else:
            raise ValueError('wrong model type')

        if parallel and depth <= 4:
            with ThreadPoolExecutor(max_workers=2) as executor:
                proc_left = executor.submit(self.fit_tree, X[idx_left, :], y[idx_left], gamma[idx_left], model, eps, depth+1)
                proc_right = executor.submit(self.fit_tree, X[idx_right, :], y[idx_right], gamma[idx_right], model, eps, depth+1)
                tree.left = proc_left.result()
                tree.right = proc_right.result()
        else:
            # print("left subtree: {:d} examples".format(np.sum(idx_left)))
            tree.left = self.fit_tree(X[idx_left, :], y[idx_left], gamma[idx_left], model, eps, depth+1)
            # print("right subtree: {:d} examples".format(np.sum(idx_right)))
            tree.right = self.fit_tree(X[idx_right, :], y[idx_right], gamma[idx_right], model, eps, depth+1)

        if depth == 1:
            # a list of all nodes at the root is needed for fast parallel predictions
            tree.node_list = tree.to_array_contiguous()
        return tree

    def fit_stumps_over_coords(self, X, y, gamma, model, eps, depth):
        verbose = False
        parallel = True
        n_ex = X.shape[0]
        X, y, gamma = X.astype(dtype), y.astype(dtype), gamma.astype(dtype)
        prev_loss = np.mean(gamma)

        # 151 features are always 0.0 on MNIST 2 vs 6. And this number is even higher for smaller subsets of MNIST,
        # i.e. subsets of examples partitioned by tree splits.
        idx_non_trivial = np.abs(X).sum(axis=0) > 0.0
        features_to_check = np.random.permutation(np.where(idx_non_trivial)[0])[:self.n_trials_coord]

        n_coords = len(features_to_check)
        params, min_losses = np.zeros((n_coords, 4)), np.full(n_coords, np.inf)

        if parallel:
            n_proc = get_n_proc(n_ex)
            n_proc = min(n_coords, min(100, n_proc))
            batch_size = n_coords // n_proc
            n_batches = n_coords // batch_size + 1

            with ThreadPoolExecutor(max_workers=n_proc) as executor:
                procs = []
                for i_batch in range(n_batches):
                    coords = features_to_check[i_batch*batch_size:(i_batch+1)*batch_size]
                    args = (X[:, coords], y, gamma, model, eps, coords, self.n_bins, self.min_samples_leaf, self.max_weight)
                    procs.append(executor.submit(fit_stump_batch, *args))

                # Process the results
                i_coord = 0
                for i_batch in range(n_batches):
                    res_many = procs[i_batch].result()
                    for res in res_many:
                        min_losses[i_coord], *params[i_coord, :] = res
                        i_coord += 1
        else:
            for i_coord, coord in enumerate(features_to_check):
                min_losses[i_coord], *params[i_coord, :] = fit_stump(
                    X[:, coord], y, gamma, model, eps, coord, self.n_bins, self.min_samples_leaf, self.max_weight)

        id_best_coord = min_losses.argmin()
        min_loss = min_losses[id_best_coord]
        best_coord = int(params[id_best_coord][3])  # float to int is necessary for a coordinate
        best_wl, best_wr, best_b = params[id_best_coord][0], params[id_best_coord][1], np.float32(params[id_best_coord][2])
        if verbose:
            print('[{}-vs-all] depth {}: n_ex {}, n_coords {} -- loss {:.5f}->{:.5f}, b={:.3f} wl={:.3f} wr={:.3f} at coord {}'.format(
                self.idx_clsf, depth, n_ex, n_coords, prev_loss, min_loss, best_b, best_wl, best_wr, best_coord))
        return best_wl, best_wr, best_b, best_coord, min_loss


def fit_stump_batch(Xs, y, gamma, model, eps, coords, n_bins, min_samples_leaf, max_weight):
    res = np.zeros([len(coords), 5])
    for i, coord in enumerate(coords):
        res[i] = fit_stump(Xs[:, i], y, gamma, model, eps, coord, n_bins, min_samples_leaf, max_weight)
    return res


def fit_stump(X_proj, y, gamma, model, eps, coord, n_bins, min_samples_leaf, max_weight):
    min_prec_val = 1e-7
    min_val, max_val = 0.0, 1.0  # can be changed if the features are in a different range

    if n_bins > 0:
        if model == 'robust_bound':
            # e.g. that's the thresholds that one gets with n_bins=10: [0.31, 0.41, 0.5, 0.59, 0.69]
            b_vals = np.arange(eps*n_bins, n_bins - eps*n_bins + 1) / n_bins
            # to have some margin to make the thresholds not adversarially reachable from 0 or 1
            b_vals[b_vals < 0.5] += 0.1 * 1/n_bins
            b_vals[b_vals > 0.5] -= 0.1 * 1/n_bins
        else:
            b_vals = np.arange(1, n_bins) / n_bins
    else:
        threshold_candidates = np.sort(X_proj)
        if min_samples_leaf > 0:
            threshold_candidates = threshold_candidates[min_samples_leaf:-min_samples_leaf]
        if len(threshold_candidates) == 0:  # if no samples left according to min_samples_leaf
            return [np.inf, 0.0, 0.0, 0.0, -1]
        if model not in ['robust_bound'] or eps == 0.0:  # plain or da_uniform training
            b_vals = np.copy(threshold_candidates)
            b_vals += min_prec_val  # to break the ties
        else:  # robust training
            b_vals = np.concatenate((threshold_candidates - eps, threshold_candidates + eps), axis=0)
            b_vals = np.clip(b_vals, min_val, max_val)  # save computations (often goes 512 -> 360 thresholds on MNIST)
            # to make in the overlapping case [---x-[--]-x---] output 2 different losses in the middle
            n_bs = len(threshold_candidates)
            b_vals += np.concatenate((-np.full(n_bs, min_prec_val), np.full(n_bs, min_prec_val)), axis=0)
        b_vals = np.unique(b_vals)  # use only unique b's
        b_vals = np.sort(b_vals)  # still important to sort because of the final threshold selection

    if model in ['plain', 'da_uniform', 'at_cube']:
        losses, w_l_vals, w_r_vals, b_vals = fit_plain_stumps(X_proj, y, gamma, b_vals, max_weight)
    elif model == 'robust_bound':
        losses, w_l_vals, w_r_vals, b_vals = fit_robust_bound_stumps(X_proj, y, gamma, b_vals, eps, max_weight)
    else:
        raise ValueError('wrong model')

    min_loss = np.min(losses)
    # probably, they are already sorted, but to be 100% sure since it is not explicitly mentioned in the docs
    indices_opt_init = np.sort(np.where(losses == min_loss)[0])
    indices_opt = get_contiguous_indices(indices_opt_init)
    id_opt = indices_opt[len(indices_opt) // 2]

    idx_prev = np.clip(indices_opt[0] - 1, 0, len(b_vals) - 1)  # to prevent stepping out of the array
    idx_next = np.clip(indices_opt[-1] + 1, 0, len(b_vals) - 1)  # to prevent stepping out of the array
    b_prev, w_l_prev, w_r_prev = b_vals[idx_prev], w_l_vals[idx_prev], w_r_vals[idx_prev]
    b_next, w_l_next, w_r_next = b_vals[idx_next], w_l_vals[idx_next], w_r_vals[idx_next]
    # initialization
    b_leftmost, b_rightmost = b_vals[indices_opt[0]], b_vals[indices_opt[-1]]

    if n_bins > 0:  # note that one shouldn't average thresholds since it's unpredictable what is in between
        return [min_loss, w_l_vals[id_opt], w_r_vals[id_opt], b_vals[id_opt], coord]

    # more involved, since with +-eps, an additional check of the loss is needed
    if model in ['plain', 'da_uniform', 'at_cube']:
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

    # For the chosen threshold, we need to calculate w_l, w_r
    # Some of w_l, w_r that correspond to min_loss may not be optimal anymore
    b_val_final = np.array([b_opt])
    if model in ['plain', 'da_uniform', 'at_cube']:
        loss, w_l_opt, w_r_opt, _ = fit_plain_stumps(X_proj, y, gamma, b_val_final, max_weight)
    elif model == 'robust_bound':
        loss, w_l_opt, w_r_opt, _ = fit_robust_bound_stumps(X_proj, y, gamma, b_val_final, eps, max_weight)
    else:
        raise ValueError('wrong model')
    loss, w_l_opt, w_r_opt = loss[0], w_l_opt[0], w_r_opt[0]

    # recalculation of w_l, w_r shouldn't change the min loss
    if np.abs(loss - min_loss) > 1e7:
        print('New loss: {:.5f}, min loss before: {:.5f}'.format(loss, min_loss))

    best_loss = losses[id_opt]
    return [best_loss, w_l_opt, w_r_opt, b_opt, coord]

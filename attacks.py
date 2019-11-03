import numpy as np


def sampling_attack(f, X, y, eps, n_trials):
    """ A simple attack just by sampling in the Linf-box around the points. More of a sanity check.
        `f` is any function that has f.predict() method that returns class scores.
    """
    num, dim = X.shape
    f_x_vals = np.zeros((num, n_trials))
    # Note: for efficiency, we sample the same random direction for all points
    deltas = np.random.uniform(-eps, eps, size=(dim, n_trials))
    for i in range(n_trials - 1):
        # let's keep them as real images, although not strictly needed
        perturbed_pts = np.clip(X + deltas[:, i], 0.0, 1.0)
        f_x_vals[:, i] = f.fmargin(perturbed_pts)
    # maybe in some corner cases, the predictions at the original point is more worst-case than the sampled points
    f_x_vals[:, n_trials - 1] = f.fmargin(X, np.ones(X.shape[0]))

    idx_min = np.argmin(y[:, None] * f_x_vals, axis=1)
    f_x_min = (y[:, None] * f_x_vals)[idx_min]
    deltas = deltas[:, idx_min]
    return f_x_min, deltas


def cube_attack(f, X, y, eps, n_trials, p=0.5, deltas_init=None, independent_delta=False, min_val=0.0, max_val=1.0):
    """ A simple, but efficient black-box attack that just adds random steps of values in {-2eps, 0, 2eps}
    (i.e., the considered points are always corners). The random change is added if the loss decreases for a
    particular point. The only disadvantage of this method is that it will never find decision regions inside the
    Linf-ball which do not intersect any corner. But tight LRTE (compared to RTE/URTE) suggest that this doesn't happen.
        `f` is any function that has f.fmargin() method that returns class scores.
        `eps` can be a scalar or a vector of size X.shape[0].
        `min_val`, `max_val` are min/max allowed values for values in X (e.g. 0 and 1 for images). This can be adjusted
        depending on the feature range of the data. It's also possible to specify the as numpy vectors.
    """
    assert type(eps) is float or type(eps) is np.ndarray

    p_neg_eps = p/2  # probability of sampling -2eps
    p_pos_eps = p/2  # probability of sampling +2eps
    p_zero = 1 - p  # probability of not doing an update
    num, dim = X.shape
    # independent deltas work better for adv. training but slow down attacks
    size_delta = (num, dim) if independent_delta else (1, dim)

    if deltas_init is None:
        deltas_init = np.zeros(size_delta)
    # this init is important, s.t. there is no violation of bounds
    f_x_vals_min = f.fmargin(X, y)

    if deltas_init is not None:  # evaluate the provided deltas and take them if they are better
        X_adv = np.clip(X + deltas_init, np.maximum(min_val, X - eps), np.minimum(max_val, X + eps))
        deltas = X_adv - X  # because of the projection above, the new delta vector is not just +-eps
        f_x_vals = f.fmargin(X_adv, y)
        idx_improved = f_x_vals < f_x_vals_min
        f_x_vals_min = idx_improved * f_x_vals + ~idx_improved * f_x_vals_min
        deltas = idx_improved[:, None] * deltas_init + ~idx_improved[:, None] * deltas
    else:
        deltas = deltas_init

    i_trial = 0
    while i_trial < n_trials:
        # +-2*eps is *very* important to escape local minima; +-eps has very unstable performance
        new_deltas = np.random.choice([-1, 0, 1], p=[p_neg_eps, p_zero, p_pos_eps], size=size_delta)
        new_deltas = 2 * eps * new_deltas  # if eps is a vector, then it's an outer product num x 1 times 1 x dim
        X_adv = np.clip(X + deltas + new_deltas, np.maximum(min_val, X - eps), np.minimum(max_val, X + eps))
        new_deltas = X_adv - X  # because of the projection above, the new delta vector is not just +-eps
        f_x_vals = f.fmargin(X_adv, y)
        idx_improved = f_x_vals < f_x_vals_min
        f_x_vals_min = idx_improved * f_x_vals + ~idx_improved * f_x_vals_min
        deltas = idx_improved[:, None] * new_deltas + ~idx_improved[:, None] * deltas
        i_trial += 1

    return f_x_vals_min, deltas


def binary_search_attack(attack, f, X, y, n_trials_attack, cleanup=True):
    """
    Binary search to find the minimal perturbation that changes the class using `attack`.
    Supports a single eps only.
    """
    n_iter_bs = 10  # precision up to the 4th digit
    num, dim = X.shape
    deltas = np.zeros([num, dim])
    eps = np.ones((num, 1))
    eps_step = 1.0
    for i_iter_bs in range(n_iter_bs):
        f_x_vals, new_deltas = attack(f, X, y, eps, n_trials_attack, p=0.5, deltas_init=deltas)
        print('iter_bs {}: yf={}, eps={}'.format(i_iter_bs, f_x_vals, eps.flatten()))
        idx_adv = f_x_vals[:, None] < 0.0  # if adversarial, reduce the eps
        eps = idx_adv * (eps - eps_step/2) + ~idx_adv * (eps + eps_step/2)
        deltas = idx_adv * new_deltas + ~idx_adv * deltas
        eps_step /= 2

    yf = f.fmargin(X + deltas, y)
    print('yf after binary search: yf={}, Linf={}'.format(yf, np.abs(deltas).max(1)))
    if np.any(yf >= 0.0):
        print('The class was not changed (before cleanup)! Some bug apparently!')

    if cleanup:
        # If some eps/-eps do not change the prediction for a particular example, use delta_i = 0 instead.
        # Better for interpretability. Caution: needs num * dim function evaluations, thus advisable to use only
        # for visualizations, but not for LRTE.
        for i in range(dim):
            deltas_i_zeroed = np.copy(deltas)
            deltas_i_zeroed[:, i] = 0.0
            f_x_vals = f.fmargin(X + deltas_i_zeroed, y)
            idx_adv = f_x_vals < 0.0
            deltas = idx_adv[:, None] * deltas_i_zeroed + ~idx_adv[:, None] * deltas

    yf = f.fmargin(X + deltas, y)
    print('yf after cleanup: yf={}, Linf={}'.format(yf, np.abs(deltas).max(1)))
    if np.any(yf >= 0.0):
        print('The class was not changed (after cleanup)! Some bug apparently!')

    return deltas


def coord_descent_attack_trees(f, X, y, eps, n_trials, deltas=None):
    """ A simple, but relatively efficient (if multiple passes through the coordinates are allowed) white-box attack
    just by iterating over coordinates (in the importance order) and checking whether -eps, 0 or eps is better.
    Needs 2 function evaluations per coordinate.
        `f` is a TreeEnsemble object.
    """
    num, dim = X.shape
    if deltas is None:
        deltas = np.zeros((num, dim))
    # this init is important, s.t. there is no violation of bounds
    f_x_vals_min = y * f.fmargin(np.clip(X + deltas, np.maximum(0.0, X - eps), np.minimum(1.0, X + eps)))

    coords_per_tree = np.zeros(dim)
    for tree in f.trees:
        coords_curr_tree = np.array(tree.to_list(), dtype=int)[:, 6]
        for coord in coords_curr_tree:  # 6 is coord, 7 is min_loss
            coords_per_tree[coord] += 1
    idx_coords_sorted = np.argsort(-coords_per_tree)  # sort in the reverse order
    coords_nnz_usage = np.where(coords_per_tree[idx_coords_sorted] != 0)[0]
    coords_to_consider = idx_coords_sorted[coords_nnz_usage]
    # print('The most important coords:', coords_to_consider[:20])

    i_trial, id_coord = 0, 0
    X_adv = X
    while i_trial < n_trials:
        # if len(coords_to_consider) < n_trials, then we do more than 1 cycle of the coordinate descent scheme
        coord = coords_to_consider[id_coord % len(coords_to_consider)]
        for new_delta in [-eps, eps]:
            X_adv_new = X + deltas
            # because of multiple cycles of coordinate descent, we also need to consider +-eps constraints
            X_adv_new[:, coord] = np.clip(X_adv_new[:, coord] + new_delta, np.maximum(0.0, X[:, coord] - eps),
                                          np.minimum(1.0, X[:, coord] + eps))
            # because of constraint projections, the new delta vector is not just +-eps
            new_delta_vector = X_adv_new[:, coord] - X_adv[:, coord]
            f_x_vals = y * f.fmargin(X_adv_new)
            improved = (f_x_vals < f_x_vals_min)
            f_x_vals_min = improved * f_x_vals + ~improved * f_x_vals_min
            deltas[:, coord] = improved * new_delta_vector + ~improved * deltas[:, coord]
            i_trial += 1
        id_coord += 1

    return f_x_vals_min, deltas


def exact_attack_stumps(f, X, y):
    """ Fast exact adv. examples for boosted stumps.
        `f` is a StumpEnsemble object.
    """
    min_val = 1e-7
    num, dim = X.shape
    deltas = np.zeros([num, dim])
    db_dists = np.full(num, np.inf)

    for i in range(num):
        # 0.0 means we just check whether the point is originally misclassified; if yes  =>  db_dist=0
        eps_all_i = np.array([0.0] + [np.abs(tree.b - X[i, tree.coord] + min_val*np.sign(tree.b - X[i, tree.coord]))
                                      for tree in f.trees])
        eps_sorted = np.sort(eps_all_i)
        for eps in eps_sorted:
            # Vectorized but obscure version that doesn't return deltas; just a sanity check for eps
            # f_x_min = self.certify_exact(X[None, i], y[None, i], eps)

            # Clear unvectorized version
            yf_min = 0.0
            delta = np.zeros(dim)
            for coord in f.coords_trees.keys():
                trees_current_coord = f.coords_trees[coord]

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

            yf = float(y[i] * f.predict(X[None, i] + delta[None]))
            print('eps_max={:.3f}, eps_delta={:.3f}, yf={:.3f}, nnz={}'.format(
                eps, np.abs(delta).max(), yf, (delta != 0.0).sum()))
            if yf_min < 0:
                db_dists[i] = eps
                deltas[i] = delta
                break
        print()
        yf = y[i] * f.predict(X[None, i] + deltas[None, i])
        if yf >= 0.0:
            print('The class was not changed! Some bug apparently!')
    return deltas


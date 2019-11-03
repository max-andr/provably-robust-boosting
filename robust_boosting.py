import numpy as np
import math
from numba import jit, prange
from utils import minimum, clip


dtype = np.float32  # float32 is much faster than float64 because of exp
parallel = False  # and then it also depends on NUMBA_NUM_THREADS
nogil = True


@jit(nopython=True, nogil=nogil)
def fit_plain_stumps_iter(X_proj, y, gamma, b_vals_i, sum_1, sum_m1, max_weight):
    ind = X_proj >= b_vals_i
    sum_1_1, sum_1_m1 = np.sum(ind * (y == 1) * gamma), np.sum(ind * (y == -1) * gamma)
    sum_0_1, sum_0_m1 = sum_1 - sum_1_1, sum_m1 - sum_1_m1
    w_l, w_r = coord_descent_exp_loss(sum_1_1, sum_1_m1, sum_0_1, sum_0_m1, max_weight)

    fmargin = y * w_l + y * w_r * ind
    loss = np.mean(gamma * np.exp(-fmargin))
    return loss, w_l, w_r


@jit(nopython=True, nogil=nogil, parallel=parallel)  # really matters, especially with independent iterations
def fit_plain_stumps(X_proj, y, gamma, b_vals, max_weight):
    n_thresholds = b_vals.shape[0]

    losses = np.full(n_thresholds, np.inf, dtype=dtype)
    w_l_vals = np.full(n_thresholds, np.inf, dtype=dtype)
    w_r_vals = np.full(n_thresholds, np.inf, dtype=dtype)
    sum_1, sum_m1 = np.sum((y == 1) * gamma), np.sum((y == -1) * gamma)
    for i in prange(n_thresholds):
        # due to a numba bug, if we don't use a separate function inside a prange-loop, we experience a memory leak
        losses[i], w_l_vals[i], w_r_vals[i] = fit_plain_stumps_iter(
            X_proj, y, gamma, b_vals[i], sum_1, sum_m1, max_weight)
    return losses, w_l_vals, w_r_vals, b_vals


@jit(nopython=True, nogil=nogil)
def fit_robust_bound_stumps_iter(X_proj, y, gamma, b_vals_i, sum_1, sum_m1, eps, max_weight):
    # Certification for the previous ensemble O(n)
    split_lbs, split_ubs = X_proj - eps, X_proj + eps
    guaranteed_right = split_lbs > b_vals_i
    uncertain = (split_lbs <= b_vals_i) * (split_ubs >= b_vals_i)

    loss, w_l, w_r = basic_case_two_intervals(y, gamma, guaranteed_right, uncertain, sum_1, sum_m1, max_weight)
    return loss, w_l, w_r


@jit(nopython=True, nogil=nogil, parallel=parallel)  # parallel=True really matters, especially with independent iterations
def fit_robust_bound_stumps(X_proj, y, gamma, b_vals, eps, max_weight):
    n_thresholds = b_vals.shape[0]

    losses = np.full(n_thresholds, np.inf, dtype=dtype)
    w_l_vals = np.full(n_thresholds, np.inf, dtype=dtype)
    w_r_vals = np.full(n_thresholds, np.inf, dtype=dtype)
    sum_1, sum_m1 = np.sum((y == 1) * gamma), np.sum((y == -1) * gamma)
    for i in prange(n_thresholds):
        losses[i], w_l_vals[i], w_r_vals[i] = fit_robust_bound_stumps_iter(
            X_proj, y, gamma, b_vals[i], sum_1, sum_m1, eps, max_weight)

    return losses, w_l_vals, w_r_vals, b_vals


@jit(nopython=True, nogil=nogil)
def fit_robust_exact_stumps_iter(X_proj, y, gamma, w_rs, bs, b_vals_i, sum_1, sum_m1, eps, max_weight):
    # Certification for the previous ensemble O(n)
    split_lbs, split_ubs = X_proj - eps, X_proj + eps
    guaranteed_right = split_lbs > b_vals_i
    uncertain = (split_lbs <= b_vals_i) * (split_ubs >= b_vals_i)

    h_l, h_r = calc_h(X_proj, y, w_rs, bs, b_vals_i, eps)
    # there should be quite many useless coordinates which do not have any stumps in the ensemble
    # thus h_l=h_r=0  =>  suffices to check just 2 regions without applying bisection
    if np.sum(h_l) == 0.0 and np.sum(h_r) == 0.0:
        loss, w_l, w_r = basic_case_two_intervals(y, gamma, guaranteed_right, uncertain, sum_1, sum_m1, max_weight)
    else:  # general case; happens only when `coord` was already splitted in the previous iterations
        loss, w_l, w_r = bisect_coord_descent(y, gamma, h_l, h_r, guaranteed_right, uncertain, max_weight)
    return loss, w_l, w_r


@jit(nopython=True, nogil=nogil, parallel=parallel)  # parallel=True really matters, especially with independent iterations
def fit_robust_exact_stumps(X_proj, y, gamma, b_vals, eps, w_rs, bs, max_weight):
    n_thresholds = b_vals.shape[0]

    losses = np.full(n_thresholds, np.inf, dtype=dtype)
    w_l_vals = np.full(n_thresholds, np.inf, dtype=dtype)
    w_r_vals = np.full(n_thresholds, np.inf, dtype=dtype)
    sum_1, sum_m1 = np.sum((y == 1) * gamma), np.sum((y == -1) * gamma)
    for i in prange(n_thresholds):
        losses[i], w_l_vals[i], w_r_vals[i] = fit_robust_exact_stumps_iter(
            X_proj, y, gamma, w_rs, bs, b_vals[i], sum_1, sum_m1, eps, max_weight)

    return losses, w_l_vals, w_r_vals, b_vals


@jit(nopython=True, nogil=nogil)  # almost 2 times speed-up by njit for this loop!
def coord_descent_exp_loss(sum_1_1, sum_1_m1, sum_0_1, sum_0_m1, max_weight):
    m = 1e-10
    # if sum_0_1 + sum_0_m1 == 0 or sum_1_1 + sum_1_m1 == 0:
    #     return np.inf, np.inf
    # w_l = (sum_0_1 - sum_0_m1) / (sum_0_1 + sum_0_m1)
    # w_r = (sum_1_1 - sum_1_m1) / (sum_1_1 + sum_1_m1) - w_l

    # 1e-4 up to 20-50 iters; 1e-6 up to 100-200 iters which leads to a significant slowdown in practice
    eps_precision = 1e-4

    # We have to properly handle the cases when the optimal leaf value is +-inf.
    if sum_1_m1 < m and sum_0_1 < m:
        w_l, w_r = -max_weight, 2 * max_weight
    elif sum_1_1 < m and sum_0_m1 < m:
        w_l, w_r = max_weight, -2 * max_weight
    elif sum_1_m1 < m:
        w_r = max_weight
        w_l = 0.5 * math.log((math.exp(-w_r) * sum_1_1 + sum_0_1) / (math.exp(w_r) * sum_1_m1 + sum_0_m1))
    elif sum_1_1 < m:
        w_r = -max_weight
        w_l = 0.5 * math.log((math.exp(-w_r) * sum_1_1 + sum_0_1) / (math.exp(w_r) * sum_1_m1 + sum_0_m1))
    elif sum_0_1 < m:
        w_l = -max_weight
        w_r = 0.5 * math.log(sum_1_1 / sum_1_m1) - w_l
    elif sum_0_m1 < m:
        w_l = max_weight
        w_r = 0.5 * math.log(sum_1_1 / sum_1_m1) - w_l
    else:  # main case
        w_r = 0.0
        w_l = 0.0
        w_r_prev, w_l_prev = np.inf, np.inf
        i = 0
        # Note: ideally one has to calculate the loss, but O(n) factor would slow down everything here
        while (np.abs(w_r - w_r_prev) > eps_precision) or (np.abs(w_l - w_l_prev) > eps_precision):
            i += 1
            w_r_prev, w_l_prev = w_r, w_l
            w_r = 0.5 * math.log(sum_1_1 / sum_1_m1) - w_l
            w_l = 0.5 * math.log((math.exp(-w_r) * sum_1_1 + sum_0_1) / (math.exp(w_r) * sum_1_m1 + sum_0_m1))
            if i == 50:
                break
    left_leaf = clip(w_l, -max_weight, max_weight)
    right_leaf = clip(left_leaf + w_r, -max_weight, max_weight)
    w_l, w_r = left_leaf, right_leaf - left_leaf
    return w_l, w_r


@jit(nopython=True, nogil=nogil)
def calc_h(X_proj, y, w_rs, bs, b_curr, eps):
    num = X_proj.shape[0]
    h_l_base, h_r_base = np.zeros(num), np.zeros(num)
    if len(bs) == 0:
        return h_l_base, h_r_base

    # Has to be calculated inside of the loop since depends on the current b
    for i in range(len(w_rs)):
        # idea: accumulate all the thresholds that preceed the leftmost point
        h_l_base += y * w_rs[i] * (X_proj - eps >= bs[i])  # leftmost point is `X_proj - eps`
        h_r_base += y * w_rs[i] * (np.maximum(b_curr, X_proj - eps) >= bs[i])  # leftmost point is max(b_curr, x-eps)
    # check all thresholds, and afterwards check if they are in (x-eps, x+eps]
    idx = np.argsort(bs)
    sorted_thresholds = bs[idx]
    sorted_w_r = w_rs[idx]

    min_left, min_right = np.zeros(num), np.zeros(num)
    cumsum_left, cumsum_right = np.zeros(num), np.zeros(num)
    for i_t in range(len(sorted_thresholds)):
        # consider the threshold if it belongs to (x-eps, min(b, x+eps)] (x-eps is excluded since already evaluated)
        idx_x_left = (X_proj - eps < sorted_thresholds[i_t]) * (sorted_thresholds[i_t] <= b_curr) * (
                sorted_thresholds[i_t] <= X_proj + eps)
        # consider the threshold if it belongs to (max(b, x-eps), x+eps] (b is excluded since already evaluated)
        idx_x_right = (b_curr < sorted_thresholds[i_t]) * (X_proj - eps < sorted_thresholds[i_t]) * (
                sorted_thresholds[i_t] <= X_proj + eps)
        assert np.sum(idx_x_left * idx_x_right) == 0  # mutually exclusive  =>  cannot be True at the same time
        diff_left = y * sorted_w_r[i_t] * idx_x_left
        diff_right = y * sorted_w_r[i_t] * idx_x_right
        # Note: numba doesn't support cumsum over axis=1 nor min over axis=1
        cumsum_left += diff_left
        cumsum_right += diff_right
        min_left = minimum(cumsum_left, min_left)
        min_right = minimum(cumsum_right, min_right)
    h_l = h_l_base + min_left
    h_r = h_r_base + min_right
    # That was the case when b is in [x-eps, x+eps]. If not, then:
    h_l = h_l * (b_curr >= X_proj - eps)  # zero out if b_curr < X_proj - eps
    h_r = h_r * (b_curr <= X_proj + eps)  # zero out if b_curr > X_proj + eps
    return h_l, h_r


@jit(nopython=True, nogil=nogil)
def bisection(w_l, y, gamma, h_l, h_r, guaranteed_right, uncertain, max_weight):
    # bisection to find w_r* for the current w_l
    eps_precision = 1e-5  # 1e-5: 21 steps, 1e-4: 18 steps (assuming max_weight=10)
    w_r = 0.0
    w_r_lower, w_r_upper = -max_weight, max_weight
    loss_best = np.inf
    i = 0
    while i == 0 or np.abs(w_r_upper - w_r_lower) > eps_precision:
        w_r = (w_r_lower + w_r_upper) / 2
        ind = guaranteed_right + (y * w_r < h_l - h_r) * uncertain

        # Calculate the indicator function based on the known h_l - h_r
        fmargin = y * w_l + h_l + (h_r - h_l + y * w_r) * ind
        losses_per_pt = gamma * np.exp(-fmargin)
        loss = np.mean(losses_per_pt)  # also O(n)
        # derivative wrt w_r for bisection
        derivative = np.mean(-losses_per_pt * y * ind)

        if loss < loss_best:
            w_r_best, loss_best = w_r, loss
        if derivative >= 0:
            w_r_upper = w_r
        else:
            w_r_lower = w_r

        i += 1
    return w_r


@jit(nopython=True, nogil=nogil)
def bisect_coord_descent(y, gamma, h_l, h_r, guaranteed_right, uncertain, max_weight):
    eps_precision = 1e-5
    w_l_prev, w_r_prev = np.inf, np.inf
    w_l, w_r = 0.0, 0.0
    i = 0
    while np.abs(w_l - w_l_prev) > eps_precision or np.abs(w_r - w_r_prev) > eps_precision:
        w_r_prev = w_r
        w_r = bisection(w_l, y, gamma, h_l, h_r, guaranteed_right, uncertain, max_weight)

        ind = guaranteed_right + (y * w_r < h_l - h_r) * uncertain
        gamma_with_h = gamma * np.exp(-(~ind * h_l + ind * h_r))  # only for the coord descent step
        sum_1_1, sum_1_m1 = np.sum(ind * (y == 1) * gamma_with_h), np.sum(ind * (y == -1) * gamma_with_h)
        sum_0_1, sum_0_m1 = np.sum(~ind * (y == 1) * gamma_with_h), np.sum(~ind * (y == -1) * gamma_with_h)
        w_l_prev = w_l
        w_l = 0.5 * math.log((math.exp(-w_r) * sum_1_1 + sum_0_1) / (math.exp(w_r) * sum_1_m1 + sum_0_m1))
        i += 1
        if i == 10:
            break

    ind = guaranteed_right + (y * w_r < h_l - h_r) * uncertain
    fmargin = y * w_l + h_l + (h_r - h_l + y * w_r) * ind
    loss = np.mean(gamma * np.exp(-fmargin))

    return loss, w_l, w_r


def exp_loss_robust(X_proj, y, gamma, w_l, w_r, w_rs, bs, b_curr, eps, h_flag):
    num = X_proj.shape[0]
    if h_flag:
        h_l, h_r = calc_h(X_proj, y, w_rs, bs, b_curr, eps)
    else:
        h_l, h_r = np.zeros(num), np.zeros(num)

    split_lbs, split_ubs = X_proj - eps, X_proj + eps
    guaranteed_right = split_lbs > b_curr
    uncertain = (split_lbs <= b_curr) * (split_ubs >= b_curr)

    ind = guaranteed_right + (y * w_r < h_l - h_r) * uncertain
    fmargin = y * w_l + h_l + (h_r - h_l + y * w_r) * ind
    loss = np.mean(gamma * np.exp(-fmargin))
    loss = dtype(loss)  # important for the proper selection of the final threshold
    return loss


@jit(nopython=True, nogil=nogil)
def basic_case_two_intervals(y, gamma, guaranteed_right, uncertain, sum_1, sum_m1, max_weight):
    loss_best, w_r_best, w_l_best = np.inf, np.inf, np.inf
    for sign_w_r in (-1, 1):
        # Calculate the indicator function based on the known `sign_w_r`
        ind = guaranteed_right + (y * sign_w_r < 0) * uncertain

        # Calculate all partial sums
        sum_1_1, sum_1_m1 = np.sum(ind * (y == 1) * gamma), np.sum(ind * (y == -1) * gamma)
        sum_0_1, sum_0_m1 = sum_1 - sum_1_1, sum_m1 - sum_1_m1
        # Minimizer of w_l, w_r on the current interval
        w_l, w_r = coord_descent_exp_loss(sum_1_1, sum_1_m1, sum_0_1, sum_0_m1, max_weight)
        # if w_r is on the different side from 0, then sign_w_r*w_r < 0  =>  w_r:=0
        w_r = sign_w_r * max(sign_w_r * w_r, 0)

        # If w_r now become 0, we need to readjust w_l
        if sum_1_m1 != 0 and sum_0_m1 != 0:
            w_l = 0.5 * math.log((math.exp(-w_r) * sum_1_1 + sum_0_1) / (math.exp(w_r) * sum_1_m1 + sum_0_m1))
            w_l = clip(w_l, -max_weight, max_weight)
        else:  # to prevent a division over zero
            w_l = max_weight * math.copysign(1, 0.5 * math.log((math.exp(-w_r) * sum_1_1 + sum_0_1)))

        preds_adv = w_l + w_r * ind

        loss = np.mean(gamma * np.exp(-y * preds_adv))  # also O(n)
        if loss < loss_best:
            loss_best, w_l_best, w_r_best = loss, w_l, w_r
    return loss_best, w_l_best, w_r_best

import numpy as np
import scipy.stats
import sys
sys.path.append('/home/yilingh/SI-Interaction')

from selectinf.Simulation.spline_instance import (
    generate_gaussian_instance_nonlinear_interaction_simple,
    generate_gaussian_instance_nonlinear_interaction)
from selectinf.group_lasso_query import group_lasso
from selectinf.reluctant_interaction import (SPAM, split_SPAM)
from selectinf.base import selected_targets_interaction
import seaborn as sns
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial
import copy
from scipy.stats import norm as ndist

def calculate_F1_score_interactions(true_set, selected_list):
    selected_set = set(selected_list)
    # print(true_set, selected_set)

    # precision & recall
    if len(selected_set) > 0:
        precision = len(true_set & selected_set) / len(selected_set)
    else:
        precision = 0
    recall = len(true_set & selected_set) / len(true_set)

    if precision + recall > 0:
        return 2 * precision * recall / (precision + recall)
    else:
        return 0
def interaction_t_test_single(X_E, Y, Y_mean, interaction, level=0.9,
                              p_val=False, return_pivot=True):
    interaction = interaction.reshape(-1, 1)
    X_aug = np.concatenate((X_E, interaction), axis=1)
    n, p_prime = X_aug.shape

    S = np.linalg.inv(X_aug.T @ X_aug)
    H = X_aug @ S @ X_aug.T
    e = Y - H @ Y
    sigma_hat = np.sqrt(e.T @ e / (n - p_prime))
    sd = sigma_hat * np.sqrt(S[p_prime - 1, p_prime - 1])

    beta_hat = S @ X_aug.T @ Y
    beta_targets = S @ X_aug.T @ Y_mean

    # Normal quantiles
    qt_low = scipy.stats.t.ppf((1 - level) / 2, df=n - p_prime)
    qt_up = scipy.stats.t.ppf(1 - (1 - level) / 2, df=n - p_prime)
    assert np.abs(np.abs(qt_low) - np.abs(qt_up)) < 10e-6

    # Construct confidence intervals
    interval_low = beta_hat[p_prime - 1] + qt_low * sd
    interval_up = beta_hat[p_prime - 1] + qt_up * sd

    target = beta_targets[p_prime - 1]

    coverage = (target > interval_low) * (target < interval_up)

    if p_val:
        pivot = (beta_hat[p_prime - 1] / sd)
        if not return_pivot:
            p_inter = 2 * scipy.stats.norm.sf(np.abs(pivot))
        else:
            p_inter = ndist.cdf((beta_hat[p_prime - 1] - target) / sd)

        return (coverage, interval_up - interval_low,
                (interval_up * interval_low > 0), p_inter, target)

    return (coverage, interval_up - interval_low,
            (interval_up * interval_low > 0), target)

# T-test for all interaction terms
def interaction_t_tests_all(X_E, Y, Y_mean, n_features, active_vars_flag,
                            interactions, selection_idx=None,
                            level=0.9, mode="allpairs",
                            p_val=False, return_pivot=True,
                            target_ids=None):
    #print(active_vars_flag.sum())
    coverage_list = []
    length_list = []
    selected_interactions = []
    p_value_list = []
    target_list = []

    if target_ids is None:
        task_idx = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if mode == "allpairs":
                    task_idx.append((i, j))
                elif mode == 'weakhierarchy':
                    if active_vars_flag[i] or active_vars_flag[j]:
                        task_idx.append((i, j))
                elif mode == 'stronghierarchy':
                    if active_vars_flag[i] and active_vars_flag[j]:
                        task_idx.append((i, j))
    else:
        task_idx = target_ids

    for pair in task_idx:
        i, j = pair
        if selection_idx is not None:
            interaction_ij = interactions[(i, j)][~selection_idx]
        else:
            interaction_ij = interactions[(i, j)]
        if not p_val:
            coverage, length, selected, target \
                = interaction_t_test_single(X_E, Y, Y_mean,
                                            interaction_ij,
                                            level=level, p_val=p_val,
                                            return_pivot=return_pivot)
            coverage_list.append(coverage)
            length_list.append(length)
            target_list.append(target)
        else:
            coverage, length, selected, p_inter, target \
                = interaction_t_test_single(X_E, Y, Y_mean,
                                            interaction_ij,
                                            level=level, p_val=p_val,
                                            return_pivot=return_pivot)
            coverage_list.append(coverage)
            length_list.append(length)
            p_value_list.append(p_inter)
            target_list.append(target)
        if selected:
            selected_interactions.append((i, j))

    if not p_val:
        return (np.array(coverage_list), np.array(length_list),
                selected_interactions, target_list)
    else:
        return (np.array(coverage_list), np.array(length_list),
                selected_interactions, p_value_list, target_list)


def interaction_t_test_single_parallel(ij_pair, X_E, Y, Y_mean,
                                       interactions, selection_idx,
                                       level=0.9, p_val=False, return_pivot=True):
    i, j = ij_pair
    if selection_idx is not None:
        interaction = interactions[(i, j)][~selection_idx]
    else:
        interaction = interactions[(i, j)]

    interaction = interaction.reshape(-1, 1)
    X_aug = np.concatenate((X_E, interaction), axis=1)
    n, p_prime = X_aug.shape

    S = np.linalg.inv(X_aug.T @ X_aug)
    H = X_aug @ S @ X_aug.T
    e = Y - H @ Y
    sigma_hat = np.sqrt(e.T @ e / (n - p_prime))
    sd = sigma_hat * np.sqrt(S[p_prime - 1, p_prime - 1])

    beta_hat = S @ X_aug.T @ Y
    beta_targets = S @ X_aug.T @ Y_mean

    # Normal quantiles
    qt_low = scipy.stats.t.ppf((1 - level) / 2, df=n - p_prime)
    qt_up = scipy.stats.t.ppf(1 - (1 - level) / 2, df=n - p_prime)
    assert np.abs(np.abs(qt_low) - np.abs(qt_up)) < 10e-6

    # Construct confidence intervals
    interval_low = beta_hat[p_prime - 1] + qt_low * sd
    interval_up = beta_hat[p_prime - 1] + qt_up * sd

    target = beta_targets[p_prime - 1]

    coverage = (target > interval_low) * (target < interval_up)

    if p_val:
        pivot = (beta_hat[p_prime - 1] / sd)
        if not return_pivot:
            p_inter = 2 * scipy.stats.norm.sf(np.abs(pivot))
        else:
            p_inter = ndist.cdf((beta_hat[p_prime - 1] - target)/ sd)

        return (coverage, interval_up - interval_low,
                (interval_up * interval_low > 0), p_inter, target)

    return (coverage, interval_up - interval_low,
            (interval_up * interval_low > 0), target)


# T-test for all interaction terms
def interaction_t_tests_all_parallel(X_E, Y, Y_mean, n_features, active_vars_flag,
                                     interactions, selection_idx=None,
                                     level=0.9, mode="allpairs", ncores=8,
                                     p_val=False, return_pivot=True,
                                     target_ids=None):
    if target_ids is None:
        task_idx = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if mode == "allpairs":
                    task_idx.append((i, j))
                elif mode == 'weakhierarchy':
                    if active_vars_flag[i] or active_vars_flag[j]:
                        task_idx.append((i, j))
                elif mode == 'stronghierarchy':
                    if active_vars_flag[i] and active_vars_flag[j]:
                        task_idx.append((i, j))
    else:
        task_idx = target_ids

    with Pool(ncores) as pool:
        results = pool.map(partial(interaction_t_test_single_parallel,
                                   X_E=X_E, Y=Y, Y_mean=Y_mean,
                                   interactions=interactions,
                                   selection_idx=selection_idx,
                                   level=level, p_val=p_val, return_pivot=return_pivot),
                           task_idx)

    coverage_list = np.array(results)[:, 0].astype(bool)
    length_list = np.array(results)[:, 1]
    selected = np.array(results)[:, 2].astype(bool)
    selected_interactions = [pair for pair, sel in zip(task_idx, selected) if sel]

    if p_val:
        p_value_list = np.array(results)[:, 3]
        target_list = np.array(np.array(results)[:, 4])
        return (np.array(coverage_list), np.array(length_list),
                selected_interactions, p_value_list.tolist(), target_list)

    target_list = np.array(np.array(results)[:, 3])
    return (np.array(coverage_list), np.array(length_list),
            selected_interactions, target_list)


def naive_inference_inter(X, Y, groups, Y_mean, const,
                          n_features, interactions, intercept=False,
                          weight_frac=1.25, level=0.9, mode="allpairs",
                          parallel=False, ncores=8,
                          solve_only=False, continued=False,
                          nonzero_cont=None, selected_groups_cont=None,
                          p_val=False, return_pivot=True,
                          target_ids=None
                          ):
    """
    Naive inference post-selection for interaction filtering
        X: design matrix, with/without intercept, depending on the value of intercept
        Y: response
        Y_mean: True mean of Y given X
        const: LASSO/Group LASSO solver
        n_features: Number of features,
            p in the case of linear main effects
            |G| in the case of basis expansion
        interactions: Dictionary of interactions, keys are of form (i,j), i>j
    """
    n, p = X.shape

    if not continued:
        ##estimate noise level in data

        sigma_ = np.std(Y)
        if n > p:
            dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)
        else:
            dispersion = sigma_ ** 2

        sigma_ = np.sqrt(dispersion)

        ##solve group LASSO with group penalty weights = weights
        weights = dict([(i, weight_frac * sigma_ * np.sqrt(2 * np.log(p))) for i in np.unique(groups)])
        # Don't penalize intercept
        if intercept:
            weights[0] = 0

        conv = const(X=X,
                     Y=Y,
                     groups=groups,
                     weights=weights,
                     useJacobian=True,
                     perturb=np.zeros(p),
                     ridge_term=0.)

        signs, _ = conv.fit()
        nonzero = signs != 0

        selected_groups = conv.selection_variable['active_groups']
        G_E = len(selected_groups)

        if solve_only:
            return nonzero, selected_groups
    else:
        nonzero = nonzero_cont
        selected_groups = selected_groups_cont
        G_E = len(selected_groups)

    if G_E > (1 + intercept):
        print("Naive Selected Groups:", G_E)
        # E: nonzero flag
        X_E = X[:, nonzero]

        active_flag = np.zeros(np.unique(groups).shape[0])
        active_flag[selected_groups] = 1.

        if intercept:
            active_vars_flag = active_flag[1:]
        else:
            active_vars_flag = active_flag

        task_idx = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if mode == "allpairs":
                    task_idx.append((i, j))
                elif mode == 'weakhierarchy':
                    if active_vars_flag[i] or active_vars_flag[j]:
                        task_idx.append((i, j))
                elif mode == 'stronghierarchy':
                    if active_vars_flag[i] and active_vars_flag[j]:
                        task_idx.append((i, j))
        if target_ids is not None:
            inference_flag = np.any([p in task_idx for p in target_ids])
            target_ids = [p for p in target_ids if p in task_idx]
        else:
            inference_flag = True
        if not inference_flag:
            if not p_val:
                return None, None, None, None, None
            else:
                return None, None, None, None, None, None

        if parallel:
            if not p_val:
                coverages, lengths, selected_interactions, targets \
                    = interaction_t_tests_all_parallel(X_E, Y, Y_mean, n_features,
                                                       active_vars_flag, interactions,
                                                       level=level, mode=mode, ncores=ncores,
                                                       p_val=False, target_ids=target_ids)
            else:
                coverages, lengths, selected_interactions, p_values, targets \
                    = interaction_t_tests_all_parallel(X_E, Y, Y_mean, n_features,
                                                       active_vars_flag, interactions,
                                                       level=level, mode=mode, ncores=ncores,
                                                       p_val=True, return_pivot=return_pivot,
                                                       target_ids=target_ids)
        else:
            if not p_val:
                coverages, lengths, selected_interactions, targets \
                    = interaction_t_tests_all(X_E, Y, Y_mean, n_features,
                                              active_vars_flag, interactions,
                                              level=level, mode=mode, p_val=False,
                                              target_ids=target_ids)
            else:
                coverages, lengths, selected_interactions, p_values, targets \
                    = interaction_t_tests_all(X_E, Y, Y_mean, n_features,
                                              active_vars_flag, interactions,
                                              level=level, mode=mode, p_val=True,
                                              return_pivot=return_pivot,
                                              target_ids=target_ids)
        #print("Naive Selection Size:", len(selected_interactions))
        if not p_val:
            return coverages, lengths, selected_interactions, targets, task_idx#target_ids
        else:
            return coverages, lengths, selected_interactions, p_values, targets, task_idx#target_ids
    if not p_val:
        return None, None, None, None, None
    else:
        return None, None, None, None, None, None


def data_splitting_inter(X, Y, groups, Y_mean, const,
                         n_features, interactions, intercept=False,
                         weight_frac=1.25, level=0.9,
                         proportion=0.5, mode="allpairs",
                         parallel=False, ncores=8,
                         solve_only=False, continued=False,
                         nonzero_cont=None, selected_groups_cont=None, subset_cont=None,
                         p_val=False, return_pivot=True,
                         target_ids=None
                         ):
    """
    Naive inference post-selection for interaction filtering
        X: design matrix, with/without intercept, depending on the value of intercept
        Y: response
        Y_mean: True mean of Y given X
        const: LASSO/Group LASSO solver
        n_features: Number of features,
            p in the case of linear main effects
            |G| in the case of basis expansion
        interactions: Dictionary of interactions, keys are of form (i,j), i>j
    """
    n, p = X.shape
    if not continued:
        pi_s = proportion
        subset_select = np.zeros(n, np.bool_)
        subset_select[:int(pi_s * n)] = True
        n1 = subset_select.sum()
        n2 = n - n1
        np.random.shuffle(subset_select)
    else:
        subset_select = subset_cont
    X_S = X[subset_select, :]
    Y_S = Y[subset_select]
    X_notS = X[~subset_select, :]
    Y_notS = Y[~subset_select]

    if not continued:
        ##estimate noise level in data
        sigma_ = np.std(Y_S)
        if n > p:
            dispersion = np.linalg.norm(Y_S - X_S.dot(np.linalg.pinv(X_S).dot(Y_S))) ** 2 / (n1 - p)
        else:
            dispersion = sigma_ ** 2

        sigma_ = np.sqrt(dispersion)
        weight_frac *= n1 / n

        ##solve group LASSO with group penalty weights = weights
        weights = dict([(i, weight_frac * sigma_ * np.sqrt(2 * np.log(p))) for i in np.unique(groups)])
        # Don't penalize intercept
        if intercept:
            weights[0] = 0

        conv = const(X=X_S,
                     Y=Y_S,
                     groups=groups,
                     weights=weights,
                     useJacobian=True,
                     perturb=np.zeros(p),
                     ridge_term=0.)

        signs, _ = conv.fit()
        nonzero = signs != 0

        selected_groups = conv.selection_variable['active_groups']
        G_E = len(selected_groups)

        if solve_only:
            return nonzero, selected_groups, subset_select
    else:
        nonzero = nonzero_cont
        selected_groups = selected_groups_cont
        G_E = len(selected_groups)

    if G_E > (1 + intercept):
        print("DS Selected Groups:", G_E)
        # E: nonzero flag
        X_E = X_notS[:, nonzero]

        active_flag = np.zeros(np.unique(groups).shape[0])
        active_flag[selected_groups] = 1.

        if intercept:
            active_vars_flag = active_flag[1:]
        else:
            active_vars_flag = active_flag

        task_idx = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if mode == "allpairs":
                    task_idx.append((i, j))
                elif mode == 'weakhierarchy':
                    if active_vars_flag[i] or active_vars_flag[j]:
                        task_idx.append((i, j))
                elif mode == 'stronghierarchy':
                    if active_vars_flag[i] and active_vars_flag[j]:
                        task_idx.append((i, j))
        if target_ids is not None:
            inference_flag = np.any([p in task_idx for p in target_ids])
            target_ids = [p for p in target_ids if p in task_idx]
        else:
            inference_flag = True
        if not inference_flag:
            if not p_val:
                return None, None, None, None, None
            else:
                return None, None, None, None, None, None

        if parallel:
            if not p_val:
                coverages, lengths, selected_interactions, targets \
                    = interaction_t_tests_all_parallel(X_E, Y_notS,
                                                       Y_mean[~subset_select],
                                                       n_features,
                                                       active_vars_flag, interactions,
                                                       subset_select,
                                                       level=level, mode=mode, ncores=ncores,
                                                       p_val=False,
                                                       target_ids=target_ids)
            else:
                coverages, lengths, selected_interactions, p_values, targets \
                    = interaction_t_tests_all_parallel(X_E, Y_notS,
                                                       Y_mean[~subset_select],
                                                       n_features,
                                                       active_vars_flag, interactions,
                                                       subset_select,
                                                       level=level, mode=mode, ncores=ncores,
                                                       p_val=True,
                                                       return_pivot=return_pivot,
                                                       target_ids=target_ids)
        else:
            if not p_val:
                coverages, lengths, selected_interactions, targets \
                    = interaction_t_tests_all(X_E, Y_notS,
                                              Y_mean[~subset_select], n_features,
                                              active_vars_flag, interactions,
                                              selection_idx=subset_select,
                                              level=level, mode=mode, p_val=False,
                                              target_ids=target_ids)
            else:
                coverages, lengths, selected_interactions, p_values, targets \
                    = interaction_t_tests_all(X_E, Y_notS, Y_mean[~subset_select],
                                              n_features,
                                              active_vars_flag, interactions,
                                              subset_select,
                                              level=level, mode=mode, p_val=True,
                                              return_pivot=return_pivot,
                                              target_ids=target_ids)

        #print("DS Selection Size:", len(selected_interactions))
        if not p_val:
            return coverages, lengths, selected_interactions, targets, task_idx#target_ids
        else:
            return coverages, lengths, selected_interactions, p_values, targets, task_idx#target_ids

    if not p_val:
        return None, None, None, None, None
    else:
        return None, None, None, None, None, None


def interaction_selective_single(conv, dispersion, X_E, Y_mean,
                                 interaction, level=0.9, p_val=False, return_pivot=False):
    interaction = interaction.reshape(-1, 1)
    X_aug = np.concatenate((X_E, interaction), axis=1)
    n, p_prime = X_aug.shape

    S = np.linalg.inv(X_aug.T @ X_aug)
    beta_targets = S @ X_aug.T @ Y_mean

    target = beta_targets[p_prime - 1]

    conv.setup_interaction(interaction=interaction)
    conv.setup_inference(dispersion=dispersion)

    target_spec = selected_targets_interaction(conv.loglike,
                                               conv.observed_soln,
                                               leastsq=True,
                                               interaction=interaction,
                                               dispersion=dispersion)
    result, _ = conv.inference(target_spec,
                               method='selective_MLE',
                               level=level)

    if not return_pivot:
        pval = result['pvalue'][p_prime - 1]
    else:
        pval = ndist.cdf((result['MLE'][p_prime - 1] - target) / result['SE'][p_prime - 1])
        #pval = result['pivot'][p_prime - 1]
    #print("Z-values:", result['Zvalue'][p_prime - 1])

    intervals = np.asarray(result[['lower_confidence',
                                   'upper_confidence']])
    (interval_low, interval_up) = intervals[-1, :]

    coverage = (target > interval_low) * (target < interval_up)

    if p_val:
        return (coverage, interval_up - interval_low,
                (interval_up * interval_low > 0), pval, target)

    return (coverage, interval_up - interval_low,
            (interval_up * interval_low > 0), target)

def interaction_selective_single_parallel(ij_pair, conv, dispersion, X_E, Y_mean,
                                          interactions, level=0.9, p_val=False,
                                          return_pivot=False):
    conv = copy.deepcopy(conv)
    i, j = ij_pair
    interaction = interactions[(i, j)]

    interaction = interaction.reshape(-1, 1)
    X_aug = np.concatenate((X_E, interaction), axis=1)
    n, p_prime = X_aug.shape

    S = np.linalg.inv(X_aug.T @ X_aug)
    beta_targets = S @ X_aug.T @ Y_mean

    target = beta_targets[p_prime - 1]

    conv.setup_interaction(interaction=interaction)
    conv.setup_inference(dispersion=dispersion)

    target_spec = selected_targets_interaction(conv.loglike,
                                               conv.observed_soln,
                                               leastsq=True,
                                               interaction=interaction,
                                               dispersion=dispersion)
    result, _ = conv.inference(target_spec,
                               method='selective_MLE',
                               level=level)

    if not return_pivot:
        pval = result['pvalue'][p_prime - 1]
    else:
        pval = ndist.cdf((result['MLE'][p_prime - 1] - target) / result['SE'][p_prime - 1])
    intervals = np.asarray(result[['lower_confidence',
                                   'upper_confidence']])
    (interval_low, interval_up) = intervals[-1, :]

    coverage = (target > interval_low) * (target < interval_up)

    if p_val:
        return (coverage, interval_up - interval_low,
                (interval_up * interval_low > 0), pval, target)

    return (coverage, interval_up - interval_low,
            (interval_up * interval_low > 0), target)


def interaction_selective_tests_all(conv, dispersion,
                                    X_E, Y_mean, n_features, active_vars_flag,
                                    interactions,
                                    level=0.9, mode="allpairs", p_val=False,
                                    return_pivot=True,
                                    target_ids=None):
    #print(active_vars_flag.sum())
    coverage_list = []
    length_list = []
    selected_interactions = []
    p_values_list = []
    target_list = []

    if target_ids is None:
        task_idx = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if mode == "allpairs":
                    task_idx.append((i, j))
                elif mode == 'weakhierarchy':
                    if active_vars_flag[i] or active_vars_flag[j]:
                        task_idx.append((i, j))
                elif mode == 'stronghierarchy':
                    if active_vars_flag[i] and active_vars_flag[j]:
                        task_idx.append((i, j))
    else:
        task_idx = target_ids

    iter = 0
    for pair in task_idx:
        # print(iter, "th interaction out of", len(task_idx))
        iter+=1
        i, j = pair
        interaction_ij = interactions[(i, j)]
        if not p_val:
            coverage, length, selected, target \
                = interaction_selective_single(conv, dispersion, X_E, Y_mean,
                                               interaction_ij, level=level, p_val=False)
        else:
            coverage, length, selected, p_inter, target \
                = interaction_selective_single(conv, dispersion, X_E, Y_mean,
                                               interaction_ij, level=level, p_val=True,
                                               return_pivot=return_pivot)
            p_values_list.append(p_inter)
        coverage_list.append(coverage)
        length_list.append(length)
        target_list.append(target)
        if selected:
            selected_interactions.append((i, j))

    if not p_val:
        return (np.array(coverage_list), np.array(length_list),
                selected_interactions, np.array(target_list))
    else:
        return (np.array(coverage_list), np.array(length_list),
                selected_interactions, p_values_list, np.array(target_list))


def interaction_selective_tests_all_parallel(conv, dispersion,
                                             X_E, Y_mean, n_features, active_vars_flag,
                                             interactions,
                                             level=0.9, mode="allpairs", ncores=8,
                                             p_val=False, return_pivot=True,
                                             target_ids=None):
    conv = copy.deepcopy(conv)
    conv.setup_parallelization()

    if target_ids is None:
        task_idx = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if mode == "allpairs":
                    task_idx.append((i, j))
                elif mode == 'weakhierarchy':
                    if active_vars_flag[i] or active_vars_flag[j]:
                        task_idx.append((i, j))
                elif mode == 'stronghierarchy':
                    if active_vars_flag[i] and active_vars_flag[j]:
                        task_idx.append((i, j))
    else:
        task_idx = target_ids

    with Pool(ncores) as pool:
        results = pool.map(partial(interaction_selective_single_parallel,
                                   conv=conv, dispersion=dispersion,
                                   X_E=X_E, Y_mean=Y_mean,
                                   interactions=interactions,
                                   level=level, p_val=p_val,
                                   return_pivot=return_pivot),
                           task_idx)

    coverage_list = np.array(results)[:, 0].astype(bool)
    length_list = np.array(results)[:, 1]
    selected = np.array(results)[:, 2].astype(bool)
    selected_interactions = [pair for pair, sel in zip(task_idx, selected) if sel]
    target_list = np.array(results)[:, 4]
    if p_val:
        p_values = np.array(results)[:, 3]
        return (np.array(coverage_list), np.array(length_list),
                selected_interactions, p_values.tolist(), target_list)

    return (np.array(coverage_list), np.array(length_list),
            selected_interactions, target_list)


def MLE_inference_inter(X, Y, Y_mean, groups,
                        n_features, interactions, intercept=False,
                        weight_frac=1.25, level=0.9,
                        proportion=0.5, mode="allpairs",
                        parallel=False, continued=True, solve_only=False,
                        conv_cont=None, nonzero_cont=None, ncores=8,
                        p_val=False, return_pivot=True,
                        target_ids=None
                        ):
    """
    Naive inference post-selection for interaction filtering
        X: design matrix, with/without intercept, depending on the value of intercept
        Y: response
        Y_mean: True mean of Y given X
        const: LASSO/Group LASSO solver
        n_features: Number of features,
            p in the case of linear main effects
            |G| in the case of basis expansion
        interactions: Dictionary of interactions, keys are of form (i,j), i>j
    """
    n, p = X.shape

    ##estimate noise level in data
    sigma_ = np.std(Y)
    if n > p:
        dispersion \
            = (np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2
               / (n - p))
    else:
        dispersion = sigma_ ** 2

    sigma_ = np.sqrt(dispersion)

    if not continued:
        const = SPAM.gaussian

        ##solve group LASSO with group penalty weights = weights
        weights = dict([(i, weight_frac * sigma_ * np.sqrt(2 * np.log(p))) for i in np.unique(groups)])
        # Don't penalize intercept
        if intercept:
            weights[0] = 0

        prop_scalar = (1 - proportion) / proportion

        mean_diag = np.mean((X ** 2).sum(0))
        randomizer_scale = np.sqrt(mean_diag) * np.std(Y) * 1.5

        conv = const(X=X,
                     Y=Y,
                     groups=groups,
                     weights=weights,
                     useJacobian=True,
                     ridge_term=0.,
                     randomizer_scale=randomizer_scale)
        # cov_rand=X.T @ X * prop_scalar)

        signs, _ = conv.fit()
        nonzero = signs != 0
        selected_groups = conv.selection_variable['active_groups']
        G_E = len(selected_groups)

        if solve_only:
            return conv, nonzero
    else:
        conv = conv_cont
        nonzero = nonzero_cont
        selected_groups = conv.selection_variable['active_groups']
        G_E = len(selected_groups)

    if G_E > 1 + intercept:
        print("MLE Selected Groups:", G_E)
        X_E = X[:, nonzero]
        active_flag = np.zeros(np.unique(groups).shape[0])
        active_flag[selected_groups] = 1.

        if intercept:
            active_vars_flag = active_flag[1:]
        else:
            active_vars_flag = active_flag

        task_idx = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if mode == "allpairs":
                    task_idx.append((i, j))
                elif mode == 'weakhierarchy':
                    if active_vars_flag[i] or active_vars_flag[j]:
                        task_idx.append((i, j))
                elif mode == 'stronghierarchy':
                    if active_vars_flag[i] and active_vars_flag[j]:
                        task_idx.append((i, j))
        if target_ids is not None:
            inference_flag = np.any([p in task_idx for p in target_ids])
            target_ids = [p for p in target_ids if p in task_idx]
        else:
            inference_flag = True
        if not inference_flag:
            if not p_val:
                return None, None, None, None, None
            else:
                return None, None, None, None, None, None

        if parallel:
            if not p_val:
                coverages, lengths, selected_interactions, targets \
                    = interaction_selective_tests_all_parallel(conv, dispersion,
                                                               X_E, Y_mean, n_features, active_vars_flag,
                                                               interactions,
                                                               level=level, mode=mode,
                                                               ncores=ncores, p_val=False,
                                                               target_ids=target_ids)
            else:
                coverages, lengths, selected_interactions, p_values, targets \
                    = interaction_selective_tests_all_parallel(conv, dispersion,
                                                               X_E, Y_mean, n_features, active_vars_flag,
                                                               interactions,
                                                               level=level, mode=mode,
                                                               ncores=ncores, p_val=True,
                                                               return_pivot=return_pivot,
                                                               target_ids=target_ids)
        else:
            if not p_val:
                coverages, lengths, selected_interactions, targets \
                    = interaction_selective_tests_all(conv, dispersion,
                                                      X_E, Y_mean, n_features, active_vars_flag,
                                                      interactions,
                                                      level=level, mode=mode, p_val=False,
                                                      target_ids=target_ids)
            else:
                coverages, lengths, selected_interactions, p_values, targets \
                    = interaction_selective_tests_all(conv, dispersion,
                                                      X_E, Y_mean, n_features, active_vars_flag,
                                                      interactions,
                                                      level=level, mode=mode, p_val=True,
                                                      return_pivot=return_pivot,
                                                      target_ids=target_ids)

        if not p_val:
            return coverages, lengths, selected_interactions, targets, task_idx#target_ids
        return coverages, lengths, selected_interactions, p_values, targets, task_idx#target_ids

    if not p_val:
        return None, None, None, None, None
    return None, None, None, None, None, None


def plotting(oper_char_df, x_axis='p', hue='method'):
    oper_char_df_copy = oper_char_df.copy()
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(10, 6))

    # print("Mean coverage rate/length:")
    # print(oper_char_df.groupby([x_axis, hue]).mean())
    my_palette = {"MLE": "#48c072",
                  "Naive": "#fc5a50",
                  "Data Splitting": "#03719c"}

    alias = {"stronghierarchy": "Strong",
             "weakhierarchy": "Weak",
             "allpairs": "All"}
    if x_axis == 'mode':
        oper_char_df_copy['mode'] = oper_char_df_copy['mode'].map(alias)

    cov_plot = sns.boxplot(y=oper_char_df_copy["coverage rate"],
                           x=oper_char_df_copy[x_axis],
                           hue=oper_char_df_copy[hue],
                           palette=my_palette,
                           orient="v", ax=ax1,
                           showmeans=True,
                           linewidth=1)
    cov_plot.set(title='Coverage')
    cov_plot.set_ylim(0., 1.05)
    # plt.tight_layout()
    cov_plot.axhline(y=0.9, color='k', linestyle='--', linewidth=1)
    # ax1.set_ylabel("")  # remove y label, but keep ticks
    ax1.set_xlabel(x_axis)

    len_plot = sns.boxplot(y=oper_char_df_copy["avg length"],
                           x=oper_char_df_copy[x_axis],
                           hue=oper_char_df_copy[hue],
                           palette=my_palette,
                           orient="v", ax=ax2,
                           linewidth=1)
    len_plot.set(title='Length')
    # len_plot.set_ylim(0, 100)
    # len_plot.set_ylim(3.5, 7.8)
    # plt.tight_layout()
    # ax2.set_ylabel("")  # remove y label, but keep ticks
    ax2.set_xlabel(x_axis)

    handles, labels = ax2.get_legend_handles_labels()
    # fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.2)
    fig.subplots_adjust(bottom=0.15)
    fig.legend(handles, labels, loc='lower center', ncol=4)

    F1_plot = sns.boxplot(y=oper_char_df_copy["F1 score interaction"],
                          x=oper_char_df_copy[x_axis],
                          hue=oper_char_df_copy[hue],
                          palette=my_palette,
                          orient="v", ax=ax3,
                          linewidth=1)
    F1_plot.set(title='F1 score')
    ax3.set_xlabel(x_axis)

    size_plot = sns.boxplot(y=oper_char_df_copy["|G|"],
                            x=oper_char_df_copy[x_axis],
                            hue=oper_char_df_copy[hue],
                            palette=my_palette,
                            orient="v", ax=ax4,
                            linewidth=1)
    size_plot.set(title='|G|')
    ax4.set_xlabel(x_axis)

    cov_plot.legend_.remove()
    len_plot.legend_.remove()
    F1_plot.legend_.remove()
    size_plot.legend_.remove()

    # plt.suptitle("Changing n,p")
    plt.subplots_adjust(wspace=0.6, hspace=0.5)
    plt.show()


def one_sim_mode_serial(SNR, intercept_flag, p, oper_char,
                        use_MLE, mode, weight_frac, inst,
                        rho, rho_noise, full_corr):
    const = group_lasso.gaussian

    while True:  # run until we get some selection
        (design, data_interaction, Y, Y_mean, data_combined,
         groups, active, active_inter_adj, active_inter_list) \
            = inst(n=500, p_nl=p, p_l=0, s_l=0,
                   nknots=6, degree=2, SNR=SNR,
                   rho=rho, rho_noise=rho_noise, full_corr=full_corr,
                   center=False, scale=False, random_signs=True,
                   intercept=intercept_flag, structure='weakhierarchy',
                   s_interaction=10, interaction_signal=2)

        noselection = False  # flag for a certain method having an empty selected set

        if not noselection:
            # MLE inference
            coverages_ds, lengths_ds, selected_interactions_ds \
                = data_splitting_inter(X=design, Y=Y, groups=groups,
                                       Y_mean=Y_mean, const=const,
                                       n_features=p,
                                       interactions=data_interaction,
                                       weight_frac=weight_frac, level=0.9,
                                       intercept=intercept_flag,
                                       proportion=0.5,
                                       mode=mode)

            # Convert the matrix into a list of tuples
            active_inter_set = set([tuple(row) for row in active_inter_list])
            noselection = (coverages_ds is None)

        if not noselection:
            coverages, lengths, selected_interactions \
                = naive_inference_inter(X=design, Y=Y, groups=groups,
                                        Y_mean=Y_mean, const=const,
                                        n_features=p,
                                        interactions=data_interaction,
                                        weight_frac=weight_frac, level=0.9,
                                        intercept=intercept_flag,
                                        mode=mode)
            noselection = (coverages is None)

        if (not noselection) and use_MLE:
            coverages_MLE, lengths_MLE, selected_interactions_MLE \
                = MLE_inference_inter(X=design, Y=Y, groups=groups,
                                      Y_mean=Y_mean,
                                      n_features=p, interactions=data_interaction,
                                      intercept=intercept_flag,
                                      weight_frac=weight_frac, level=0.9,
                                      proportion=0.5, mode=mode, solve_only=False,
                                      continued=False, parallel=True, ncores=8)
            noselection = (coverages_MLE is None)

        if not noselection:
            F1_i_ds \
                = calculate_F1_score_interactions(true_set=active_inter_set,
                                                  selected_list=selected_interactions_ds)
            F1_i = calculate_F1_score_interactions(true_set=active_inter_set,
                                                   selected_list=selected_interactions)
            if use_MLE:
                F1_i_MLE \
                    = calculate_F1_score_interactions(true_set=active_inter_set,
                                                      selected_list=selected_interactions_MLE)
            # Naive
            oper_char["coverage rate"].append(np.mean(coverages))
            oper_char["avg length"].append(np.mean(lengths))
            oper_char["F1 score interaction"].append(F1_i)
            oper_char["method"].append('Naive')
            oper_char["|G|"].append(len(selected_interactions))
            oper_char["mode"].append(mode)
            oper_char["SNR"].append(SNR)

            # Data splitting
            oper_char["coverage rate"].append(np.mean(coverages_ds))
            oper_char["avg length"].append(np.mean(lengths_ds))
            oper_char["F1 score interaction"].append(F1_i_ds)
            oper_char["method"].append('Data Splitting')
            oper_char["|G|"].append(len(selected_interactions_ds))
            oper_char["mode"].append(mode)
            oper_char["SNR"].append(SNR)



            if use_MLE:
                # MLE
                oper_char["coverage rate"].append(np.mean(coverages_MLE))
                oper_char["avg length"].append(np.mean(lengths_MLE))
                oper_char["F1 score interaction"].append(F1_i_MLE)
                oper_char["method"].append('MLE')
                oper_char["|G|"].append(len(selected_interactions_MLE))
                oper_char["mode"].append(mode)
                oper_char["SNR"].append(SNR)

            break

def one_sim_mode(idx, SNR, intercept_flag, p,
                 use_MLE, mode, weight_frac, inst,
                 rho, rho_noise, full_corr):
    np.random.seed(idx)
    const = group_lasso.gaussian

    while True:  # run until we get some selection
        (design, data_interaction, Y, Y_mean, data_combined,
         groups, active, active_inter_adj, active_inter_list) \
            = inst(n=500, p_nl=p, p_l=0, s_l=0,
                   nknots=6, degree=2, SNR=SNR,
                   rho=rho, rho_noise=rho_noise, full_corr=full_corr,
                   center=False, scale=False, random_signs=True,
                   intercept=intercept_flag, structure='weakhierarchy',
                   s_interaction=10, interaction_signal=2)

        noselection = False  # flag for a certain method having an empty selected set

        if not noselection:
            nonzero_ds, selected_groups_ds, subset_select_ds \
                = data_splitting_inter(X=design, Y=Y, groups=groups,
                                       Y_mean=Y_mean, const=const,
                                       n_features=p,
                                       interactions=data_interaction,
                                       weight_frac=weight_frac, level=0.9,
                                       intercept=intercept_flag,
                                       proportion=0.5,
                                       mode=mode, solve_only=True, continued=False)

            # Convert the matrix into a list of tuples
            active_inter_set = set([tuple(row) for row in active_inter_list])
            noselection = (len(selected_groups_ds) <= 1 + intercept_flag)

        if not noselection:
            nonzero_naive, selected_groups_naive \
                = naive_inference_inter(X=design, Y=Y, groups=groups,
                                        Y_mean=Y_mean, const=const,
                                        n_features=p,
                                        interactions=data_interaction,
                                        weight_frac=weight_frac, level=0.9,
                                        intercept=intercept_flag,
                                        mode=mode, solve_only=True, continued=False)
            noselection = (len(selected_groups_naive) <= 1 + intercept_flag)

        if (not noselection) and use_MLE:
            conv_MLE, nonzero_MLE \
                = MLE_inference_inter(X=design, Y=Y, groups=groups,
                                      Y_mean=Y_mean,
                                      n_features=p, interactions=data_interaction,
                                      intercept=intercept_flag,
                                      weight_frac=weight_frac, level=0.9,
                                      proportion=0.5, mode=mode, solve_only=True,
                                      continued=False, parallel=False)
            selected_groups_MLE = conv_MLE.selection_variable['active_groups']
            noselection = (len(selected_groups_MLE) <= 1 + intercept_flag)

        if not noselection:
            coverages_ds, lengths_ds, selected_interactions_ds \
                = data_splitting_inter(X=design, Y=Y, groups=groups,
                                       Y_mean=Y_mean, const=const,
                                       n_features=p,
                                       interactions=data_interaction,
                                       weight_frac=2, level=0.9,
                                       intercept=intercept_flag,
                                       proportion=0.5,
                                       mode=mode, continued=True,
                                       nonzero_cont=nonzero_ds,
                                       selected_groups_cont=selected_groups_ds,
                                       subset_cont=subset_select_ds
                                       )
            coverages, lengths, selected_interactions \
                = naive_inference_inter(X=design, Y=Y, groups=groups,
                                        Y_mean=Y_mean, const=const,
                                        n_features=p,
                                        interactions=data_interaction,
                                        weight_frac=2, level=0.9,
                                        intercept=intercept_flag,
                                        mode=mode, continued=True,
                                        nonzero_cont=nonzero_naive,
                                        selected_groups_cont=selected_groups_naive)
            if use_MLE:
                coverages_MLE, lengths_MLE, selected_interactions_MLE \
                    = MLE_inference_inter(X=design, Y=Y, groups=groups,
                                          Y_mean=Y_mean,
                                          n_features=p, interactions=data_interaction,
                                          intercept=intercept_flag,
                                          weight_frac=2, level=0.9,
                                          proportion=0.5, mode=mode, solve_only=False,
                                          continued=True, parallel=False,
                                          conv_cont = conv_MLE, nonzero_cont = nonzero_MLE)

        if not noselection:
            F1_i_ds \
                = calculate_F1_score_interactions(true_set=active_inter_set,
                                                  selected_list=selected_interactions_ds)
            F1_i = calculate_F1_score_interactions(true_set=active_inter_set,
                                                   selected_list=selected_interactions)
            if use_MLE:
                F1_i_MLE \
                    = calculate_F1_score_interactions(true_set=active_inter_set,
                                                      selected_list=selected_interactions_MLE)
            # Naive
            naive_results = [np.mean(coverages), np.mean(lengths), F1_i,
                             'Naive', len(selected_interactions), mode, SNR]

            # Data splitting
            ds_results = [np.mean(coverages_ds), np.mean(lengths_ds), F1_i_ds,
                          'Data Splitting', len(selected_interactions_ds), mode, SNR]

            if use_MLE:
                # MLE
                MLE_results = [np.mean(coverages_MLE), np.mean(lengths_MLE), F1_i_MLE,
                               'MLE', len(selected_interactions_MLE), mode, SNR]
                return [naive_results, ds_results, MLE_results]

            return [naive_results, ds_results]
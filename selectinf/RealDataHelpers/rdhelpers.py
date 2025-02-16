from selectinf.group_lasso_query import group_lasso
from selectinf.reluctant_interaction import (SPAM, split_SPAM)
from selectinf.base import selected_targets_interaction
import seaborn as sns
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial
import copy
from scipy.stats import norm as ndist
import numpy as np
import scipy

def interaction_t_test_single(X_E, Y, interaction, level=0.9):
    interaction = interaction.reshape(-1, 1)
    #print(interaction)
    X_aug = np.concatenate((X_E, interaction), axis=1)
    n, p_prime = X_aug.shape

    S = np.linalg.inv(X_aug.T @ X_aug)
    H = X_aug @ S @ X_aug.T
    e = Y - H @ Y
    sigma_hat = np.sqrt(e.T @ e / (n - p_prime))
    sd = sigma_hat * np.sqrt(S[p_prime - 1, p_prime - 1])

    beta_hat = S @ X_aug.T @ Y

    # Normal quantiles
    qt_low = scipy.stats.t.ppf((1 - level) / 2, df=n - p_prime)
    qt_up = scipy.stats.t.ppf(1 - (1 - level) / 2, df=n - p_prime)
    assert np.abs(np.abs(qt_low) - np.abs(qt_up)) < 10e-6

    # Construct confidence intervals
    interval_low = beta_hat[p_prime - 1] + qt_low * sd
    interval_up = beta_hat[p_prime - 1] + qt_up * sd

    piv = (beta_hat[p_prime - 1] / sd)
    p_value = 2 * scipy.stats.norm.sf(np.abs(piv))

    return (interval_low, interval_up, beta_hat[p_prime - 1], p_value)

# T-test for all interaction terms
def interaction_t_tests_all(X_E, Y, n_features, active_vars_flag,
                            interactions, selection_idx=None,
                            level=0.9, mode="allpairs"):
    #print(active_vars_flag.sum())
    result_dict = {"i": [], "j": [], "CI_l": [], "CI_u": [],
                   "beta_hat":[], "pval": []}

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

    for pair in task_idx:
        i, j = pair
        if selection_idx is not None:
            interaction_ij = interactions[(i, j)][~selection_idx]
        else:
            interaction_ij = interactions[(i, j)]
        interval_low, interval_up, beta_hat, p_value \
            = interaction_t_test_single(X_E, Y, interaction_ij, level=level)
        result_dict["i"].append(i)
        result_dict["j"].append(j)
        result_dict["CI_l"].append(interval_low)
        result_dict["CI_u"].append(interval_up)
        result_dict["beta_hat"].append(beta_hat)
        result_dict["pval"].append(p_value)


    return result_dict

def interaction_selective_single(conv, dispersion, X_E,
                                 interaction, level=0.9):
    interaction = interaction.reshape(-1, 1)
    X_aug = np.concatenate((X_E, interaction), axis=1)
    n, p_prime = X_aug.shape

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
    pval = result['pvalue'][p_prime - 1]
    beta_hat = result['MLE'][p_prime - 1]

    intervals = np.asarray(result[['lower_confidence',
                                   'upper_confidence']])
    (interval_low, interval_up) = intervals[-1, :]

    return (interval_low, interval_up, beta_hat, pval)

def interaction_selective_tests_all(conv, dispersion,
                                    X_E, n_features, active_vars_flag,
                                    interactions,
                                    level=0.9, mode="allpairs"):
    result_dict = {"i": [], "j": [], "CI_l": [], "CI_u": [],
                   "beta_hat": [], "pval": []}

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

    iter = 0
    for pair in task_idx:
        # print(iter, "th interaction out of", len(task_idx))
        iter+=1
        i, j = pair
        interaction_ij = interactions[(i, j)]
        interval_low, interval_up, beta_hat, p_value \
            = interaction_selective_single(conv, dispersion, X_E,
                                           interaction_ij, level = level)
        result_dict["i"].append(i)
        result_dict["j"].append(j)
        result_dict["CI_l"].append(interval_low)
        result_dict["CI_u"].append(interval_up)
        result_dict["beta_hat"].append(beta_hat)
        result_dict["pval"].append(p_value)

    return result_dict


def naive_inference_real_data(X, Y, raw_data, groups, const,
                              n_features, intercept=True,
                              weight_frac=1.25, level=0.9, mode="allpairs", root_n_scaled=False
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
        dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)
    else:
        dispersion = sigma_ ** 2

    sigma_ = np.sqrt(dispersion)

    if not root_n_scaled:
        weight_frac *= np.sqrt(n)
    ##solve group LASSO with group penalty weights = weights
    weights = dict([(i, weight_frac * sigma_ * np.sqrt(2 * np.log(p))) for i in np.unique(groups)])

    # print("Naive weights:", weights)

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

    signs, soln = conv.fit()
    nonzero = signs != 0

    selected_groups = conv.selection_variable['active_groups']
    print("Selected groups:", selected_groups)
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

        data_interaction = {}
        task_idx = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if mode == "allpairs":
                    task_idx.append((i, j))
                    data_interaction[(i, j)] = raw_data[:, i] * raw_data[:, j]
                elif mode == 'weakhierarchy':
                    if active_vars_flag[i] or active_vars_flag[j]:
                        task_idx.append((i, j))
                        data_interaction[(i, j)] = raw_data[:, i] * raw_data[:, j]
                elif mode == 'stronghierarchy':
                    if active_vars_flag[i] and active_vars_flag[j]:
                        task_idx.append((i, j))
                        data_interaction[(i, j)] = raw_data[:, i] * raw_data[:, j]

        result_dict = interaction_t_tests_all(X_E, Y, n_features,
                                              active_vars_flag, data_interaction,
                                              level=level, mode=mode)

        return (result_dict, nonzero, selected_groups)
    else:
        print("Nothing selected")


def data_splitting_real_data(X_sel, Y_sel, X_inf, Y_inf, raw_data, groups, const,
                             n_features, intercept=True,
                             weight_frac=1.25, level=0.9,
                             mode="allpairs",
                             root_n_scaled=False
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

    print("raw data", raw_data.shape)
    n1, p = X_sel.shape
    ##estimate noise level in data
    dispersion = np.linalg.norm(Y_sel - X_sel.dot(np.linalg.pinv(X_sel).dot(Y_sel))) ** 2 / (n1 - p)

    sigma_ = np.sqrt(dispersion)

    if not root_n_scaled:
        weight_frac *= np.sqrt(n1)

    ##solve group LASSO with group penalty weights = weights
    weights = dict([(i, weight_frac * sigma_ * np.sqrt(2 * np.log(p))) for i in np.unique(groups)])

    # print("Data splitting weights:", weights)

    # Don't penalize intercept
    if intercept:
        weights[0] = 0

    conv = const(X=X_sel,
                 Y=Y_sel,
                 groups=groups,
                 weights=weights,
                 useJacobian=True,
                 perturb=np.zeros(p),
                 ridge_term=0.)

    signs, soln = conv.fit()
    nonzero = signs != 0

    selected_groups = conv.selection_variable['active_groups']
    G_E = len(selected_groups)

    if G_E > (1 + intercept):
        print("DS Selected Groups:", G_E)
        # E: nonzero flag
        X_E = X_inf[:, nonzero]
        print("XE:", X_E.shape)

        active_flag = np.zeros(np.unique(groups).shape[0])
        active_flag[selected_groups] = 1.

        if intercept:
            active_vars_flag = active_flag[1:]
        else:
            active_vars_flag = active_flag

        data_interaction = {}
        task_idx = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if mode == "allpairs":
                    task_idx.append((i, j))
                    data_interaction[(i, j)] = raw_data[:, i] * raw_data[:, j]
                elif mode == 'weakhierarchy':
                    if active_vars_flag[i] or active_vars_flag[j]:
                        task_idx.append((i, j))
                        data_interaction[(i, j)] = raw_data[:, i] * raw_data[:, j]
                elif mode == 'stronghierarchy':
                    if active_vars_flag[i] and active_vars_flag[j]:
                        task_idx.append((i, j))
                        data_interaction[(i, j)] = raw_data[:, i] * raw_data[:, j]

        result_dict = interaction_t_tests_all(X_E, Y_inf,
                                              n_features,
                                              active_vars_flag=active_vars_flag,
                                              interactions=data_interaction,
                                              selection_idx=None,
                                              level=level, mode=mode)
        return (result_dict, nonzero, selected_groups)

    else:
        print("Nothing selected")

def MLE_inference_real_data(X, Y, raw_data, groups,
                            n_features, intercept=True,
                            weight_frac=1.25, level=0.9,
                            proportion=None, mode="allpairs",
                            root_n_scaled=False, seed=123):

    n, p = X.shape

    ##estimate noise level in data
    sigma_ = np.std(Y)
    if n > p:
        dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)
    else:
        dispersion = sigma_ ** 2

    sigma_ = np.sqrt(dispersion)

    const = SPAM.gaussian

    ##solve group LASSO with group penalty weights = weights
    if not root_n_scaled:
        weight_frac *= np.sqrt(n)
    if proportion:
        weights = dict([(i, weight_frac / np.sqrt(proportion) * sigma_ * np.sqrt(2 * np.log(p))) for i in np.unique(groups)])
    else:
        weights = dict([(i, weight_frac * sigma_ * np.sqrt(2 * np.log(p))) for i in np.unique(groups)])

    #print("MLE weights:", weights)
    # Don't penalize intercept
    if intercept:
        weights[0] = 0

    prop_scalar = (1 - proportion) / proportion
    conv = const(X=X,
                 Y=Y,
                 groups=groups,
                 weights=weights,
                 useJacobian=True,
                 ridge_term=0.,
                 cov_rand=X.T @ X * prop_scalar * (np.std(Y)**2))
    np.random.seed(seed)
    signs, soln = conv.fit()
    nonzero = signs != 0
    selected_groups = conv.selection_variable['active_groups']
    G_E = len(selected_groups)

    if G_E > 1 + intercept:
        print("Selected groups:", selected_groups)
        print("MLE Selected Groups:", G_E)
        X_E = X[:, nonzero]
        active_flag = np.zeros(np.unique(groups).shape[0])
        active_flag[selected_groups] = 1.

        if intercept:
            active_vars_flag = active_flag[1:]
        else:
            active_vars_flag = active_flag

        data_interaction = {}
        task_idx = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if mode == "allpairs":
                    task_idx.append((i, j))
                    data_interaction[(i, j)] = raw_data[:, i] * raw_data[:, j]
                elif mode == 'weakhierarchy':
                    if active_vars_flag[i] or active_vars_flag[j]:
                        task_idx.append((i, j))
                        data_interaction[(i, j)] = raw_data[:, i] * raw_data[:, j]
                elif mode == 'stronghierarchy':
                    if active_vars_flag[i] and active_vars_flag[j]:
                        task_idx.append((i, j))
                        data_interaction[(i, j)] = raw_data[:, i] * raw_data[:, j]

        result_dict \
            = interaction_selective_tests_all(conv, dispersion,
                                              X_E, n_features, active_vars_flag,
                                              data_interaction,
                                              level=level, mode=mode)
        return (result_dict, nonzero, selected_groups)

    else:
        print("Nothing selected")
import numpy as np
import scipy.stats
import sys
sys.path.append('/home/yilingh/SI-Interaction')

from selectinf.Simulation.spline_instance import generate_gaussian_instance_nonlinear_interaction_simple
from selectinf.group_lasso_query import group_lasso
from selectinf.reluctant_interaction import (SPAM, split_SPAM)
from selectinf.base import selected_targets_interaction
import seaborn as sns
import matplotlib.pyplot as plt

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
def interaction_t_test_single(X_E, Y, Y_mean, interaction, level=0.9):
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

    ### Test
    mat = np.array([interval_up, interval_low, target])
    # print("intervals: ", mat)

    coverage = (target > interval_low) * (target < interval_up)

    return coverage, interval_up - interval_low, (interval_up * interval_low > 0)

# T-test for all interaction terms
def interaction_t_tests_all(X_E, Y, Y_mean, n_features, active_vars_flag,
                            interactions, selection_idx=None,
                            level=0.9, mode="allpairs"):
    coverage_list = []
    length_list = []
    selected_interactions = []

    if mode == "allpairs":
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if selection_idx is not None:
                    interaction_ij = interactions[(i, j)][~selection_idx]
                else:
                    interaction_ij = interactions[(i, j)]
                coverage, length, selected \
                    = interaction_t_test_single(X_E, Y, Y_mean,
                                                interaction_ij,
                                                level=level)
                coverage_list.append(coverage)
                length_list.append(length)
                if selected:
                    selected_interactions.append((i, j))
    elif mode == 'weakhierarchy':
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if active_vars_flag[i] or active_vars_flag[j]:
                    if selection_idx is not None:
                        interaction_ij = interactions[(i, j)][~selection_idx]
                    else:
                        interaction_ij = interactions[(i, j)]
                    coverage, length, selected \
                        = interaction_t_test_single(X_E, Y, Y_mean,
                                                    interaction_ij,
                                                    level=level)
                    coverage_list.append(coverage)
                    length_list.append(length)
                    if selected:
                        selected_interactions.append((i, j))
    elif mode == 'stronghierarchy':
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if active_vars_flag[i] and active_vars_flag[j]:
                    #print(i,j)
                    if selection_idx is not None:
                        interaction_ij = interactions[(i, j)][~selection_idx]
                    else:
                        interaction_ij = interactions[(i, j)]
                    coverage, length, selected \
                        = interaction_t_test_single(X_E, Y, Y_mean,
                                                    interaction_ij,
                                                    level=level)
                    coverage_list.append(coverage)
                    length_list.append(length)
                    if selected:
                        selected_interactions.append((i, j))

    return np.array(coverage_list), np.array(length_list), selected_interactions


def naive_inference_inter(X, Y, groups, Y_mean, const,
                                 n_features, interactions, intercept=False,
                                 weight_frac=1.25, level=0.9, mode="allpairs"):
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

    if G_E > (1 + intercept):
        # E: nonzero flag
        X_E = X[:, nonzero]

        active_flag = np.zeros(np.unique(groups).shape[0])
        active_flag[selected_groups] = 1.

        if intercept:
            active_vars_flag = active_flag[1:]
        else:
            active_vars_flag = active_flag

        coverages, lengths, selected_interactions \
            = interaction_t_tests_all(X_E, Y, Y_mean, n_features,
                                      active_vars_flag, interactions,
                                      level=level, mode=mode)
        print("Naive Selection Size:", len(selected_interactions))
        return coverages, lengths, selected_interactions

    return None, None, None


def data_splitting_inter(X, Y, groups, Y_mean, const,
                                n_features, interactions, intercept=False,
                                weight_frac=1.25, level=0.9,
                                proportion=0.5, mode="allpairs"):
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

    pi_s = proportion
    subset_select = np.zeros(n, np.bool_)
    subset_select[:int(pi_s * n)] = True
    n1 = subset_select.sum()
    n2 = n - n1
    np.random.shuffle(subset_select)
    X_S = X[subset_select, :]
    Y_S = Y[subset_select]
    X_notS = X[~subset_select, :]
    Y_notS = Y[~subset_select]

    ##estimate noise level in data

    sigma_ = np.std(Y_S)
    if n > p:
        dispersion = np.linalg.norm(Y_S - X_S.dot(np.linalg.pinv(X_S).dot(Y_S))) ** 2 / (n1 - p)
    else:
        dispersion = sigma_ ** 2

    sigma_ = np.sqrt(dispersion)
    weight_frac *= n1/n

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

    if G_E > (1 + intercept):
        # E: nonzero flag
        X_E = X_notS[:, nonzero]

        active_flag = np.zeros(np.unique(groups).shape[0])
        active_flag[selected_groups] = 1.

        if intercept:
            active_vars_flag = active_flag[1:]
        else:
            active_vars_flag = active_flag

        coverages, lengths, selected_interactions \
            = interaction_t_tests_all(X_E, Y_notS,
                                      Y_mean[~subset_select], n_features,
                                      active_vars_flag, interactions, subset_select,
                                      level=level, mode=mode)

        print("DS Selection Size:", len(selected_interactions))
        return coverages, lengths, selected_interactions

    return None, None, None


def interaction_selective_single(conv, dispersion, X_E, Y_mean,
                                 interaction, level=0.9):
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

    # pval = result['pvalue']
    intervals = np.asarray(result[['lower_confidence',
                                   'upper_confidence']])
    (interval_low, interval_up) = intervals[-1, :]

    coverage = (target > interval_low) * (target < interval_up)

    return coverage, interval_up - interval_low, (interval_up * interval_low > 0)


def interaction_selective_tests_all(conv, dispersion,
                                    X_E, Y_mean, n_features, active_vars_flag,
                                    interactions,
                                    level=0.9, mode="allpairs"):
    coverage_list = []
    length_list = []
    selected_interactions = []

    if mode == "allpairs":
        for i in range(n_features):
            for j in range(i + 1, n_features):
                interaction_ij = interactions[(i, j)]
                coverage, length, selected \
                    = interaction_selective_single(conv, dispersion, X_E, Y_mean,
                                                   interaction_ij, level=level)
                coverage_list.append(coverage)
                length_list.append(length)
                if selected:
                    selected_interactions.append((i, j))
    elif mode == 'weakhierarchy':
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if active_vars_flag[i] or active_vars_flag[j]:
                    interaction_ij = interactions[(i, j)]
                    coverage, length, selected \
                        = interaction_selective_single(conv, dispersion, X_E, Y_mean,
                                                       interaction_ij, level=level)
                    coverage_list.append(coverage)
                    length_list.append(length)
                    if selected:
                        selected_interactions.append((i, j))
    elif mode == 'stronghierarchy':
        #print("|E|:", active_vars_flag.sum())
        #print("n inter:", active_vars_flag * (active_vars_flag - 1) / 2)
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if active_vars_flag[i] and active_vars_flag[j]:
                    interaction_ij = interactions[(i, j)]
                    coverage, length, selected \
                        = interaction_selective_single(conv, dispersion, X_E, Y_mean,
                                                       interaction_ij, level=level)
                    coverage_list.append(coverage)
                    length_list.append(length)
                    if selected:
                        selected_interactions.append((i, j))

    return np.array(coverage_list), np.array(length_list), selected_interactions


def MLE_inference_inter(X, Y, Y_mean, groups,
                        n_features, interactions, intercept=False,
                        weight_frac=1.25, level=0.9,
                        proportion=0.5, mode="allpairs"):
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
    const = SPAM.gaussian

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

    ##solve group LASSO with group penalty weights = weights
    weights = dict([(i, weight_frac * sigma_ * np.sqrt(2 * np.log(p))) for i in np.unique(groups)])
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
                 cov_rand=X.T @ X * n * prop_scalar)

    signs, _ = conv.fit()
    nonzero = signs != 0

    selected_groups = conv.selection_variable['active_groups']
    G_E = len(selected_groups)

    if G_E > 1 + intercept:
        X_E = X[:, nonzero]
        selected_groups = conv.selection_variable['active_groups']
        active_flag = np.zeros(np.unique(groups).shape[0])
        active_flag[selected_groups] = 1.

        if intercept:
            active_vars_flag = active_flag[1:]
        else:
            active_vars_flag = active_flag

        coverages, lengths, selected_interactions \
            = interaction_selective_tests_all(conv, dispersion,
                                              X_E, Y_mean, n_features, active_vars_flag,
                                              interactions,
                                              level=level, mode=mode)

        return coverages, lengths, selected_interactions

    return None, None, None


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
                        use_MLE, mode):
    inst, const = (generate_gaussian_instance_nonlinear_interaction_simple,
                   group_lasso.gaussian)

    while True:  # run until we get some selection
        (design, data_interaction, Y, Y_mean, data_combined,
         groups, active, active_inter_adj, active_inter_list) \
            = inst(n=500, p_nl=p, p_l=0, s_l=0,
                   nknots=6, degree=2, SNR=SNR,
                   rho=0.5, rho_noise=0.5, full_corr=False,
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
                                       weight_frac=2, level=0.9,
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
                                        weight_frac=2, level=0.9,
                                        intercept=intercept_flag,
                                        mode=mode)
            noselection = (coverages is None)

        if (not noselection) and use_MLE:
            coverages_MLE, lengths_MLE, selected_interactions_MLE \
                = MLE_inference_inter(X=design, Y=Y, groups=groups,
                                      Y_mean=Y_mean,
                                      n_features=p, interactions=data_interaction,
                                      intercept=intercept_flag,
                                      weight_frac=2, level=0.9,
                                      proportion=0.5, mode=mode)
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

            # Data splitting
            oper_char["coverage rate"].append(np.mean(coverages_ds))
            oper_char["avg length"].append(np.mean(lengths_ds))
            oper_char["F1 score interaction"].append(F1_i_ds)
            oper_char["method"].append('Data Splitting')
            oper_char["|G|"].append(len(selected_interactions_ds))
            oper_char["mode"].append(mode)



            if use_MLE:
                # MLE
                oper_char["coverage rate"].append(np.mean(coverages_MLE))
                oper_char["avg length"].append(np.mean(lengths_MLE))
                oper_char["F1 score interaction"].append(F1_i_MLE)
                oper_char["method"].append('MLE')
                oper_char["|G|"].append(len(selected_interactions_MLE))
                oper_char["mode"].append(mode)

            break

def one_sim_mode(idx, SNR, intercept_flag, p,
                 use_MLE, mode, weight_frac):
    np.random.seed(idx)
    inst, const = (generate_gaussian_instance_nonlinear_interaction_simple,
                   group_lasso.gaussian)

    while True:  # run until we get some selection
        (design, data_interaction, Y, Y_mean, data_combined,
         groups, active, active_inter_adj, active_inter_list) \
            = inst(n=500, p_nl=p, p_l=0, s_l=0,
                   nknots=6, degree=2, SNR=SNR,
                   rho=0.5, rho_noise=0.5, full_corr=False,
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
                                      proportion=0.5, mode=mode)
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
import time, sys
import multiprocessing as mp

from selectinf.Simulation.simulation_helpers import (
    generate_gaussian_instance_nonlinear_interaction_simple)

from selectinf.group_lasso_query import (group_lasso)
from selectinf.Simulation.simulation_helpers import (naive_inference_inter,
                                                     data_splitting_inter,
                                                     MLE_inference_inter)
from selectinf.Simulation.plotting_helpers import *


# %%
def calculate_power(pivot, targets, level):
    pivot = np.array(pivot)
    targets = np.array(targets)
    non_null = (targets != 0)
    rejection = 2 * np.min([pivot, 1 - pivot], axis=0) < level
    true_rej = np.sum(non_null * rejection) / np.sum(non_null)

    return true_rej
# %%
def update_targets(dict, true_inter_list,
                   targets, parameter, method, idx):
    i = 0
    for id in idx:
        if true_inter_list is not None:
            if id in true_inter_list:
                dict["parameter"].append(parameter)
                dict["target"].append(targets[i])
                dict["target id"].append(str(id))
                dict["method"].append(method)
                i += 1
        else:
            dict["parameter"].append(parameter)
            dict["target"].append(targets[i])
            dict["target id"].append(str(id))
            dict["method"].append(method)
            i += 1

def vary_SNR(start=0, end=100):
    # A dictionary recording simulation results and metrics
    oper_char = {}
    oper_char["coverage rate"] = []
    oper_char["avg length"] = []
    oper_char["method"] = []
    oper_char["rho"] = []
    oper_char["signal"] = []
    oper_char["SNR"] = []
    oper_char["power"] = []

    # A dictionary recording p-values for each true interaction
    # over all simulation results.
    # Each simulation parameter (here parameter_list contain a list of SNRs
    # to be considered) has a corresponding dictionary of results
    pval_dict = {}
    parameter_list = [0.5, 1, 2, 5]
    for sig in parameter_list:
        pval_dict[sig] = {}
        for m in ['Naive', 'Data Splitting', 'MLE']:
            pval_dict[sig][m] = []

    # Group lasso solver constructor
    const = group_lasso.gaussian

    # List and array representations of true interaction indices
    active_inter_list_true = np.array([[0, 1], [1, 2], [2, 4], [1, 5], [2, 6]])
    active_inter_list_true_list = [(x[0], x[1]) for x in active_inter_list_true]

    # Dictionary of projected targets, over all simulation parameters
    target_dict = {}
    target_dict["parameter"] = []
    target_dict["target"] = []
    target_dict["target id"] = []
    target_dict["method"] = []

    # p = 50
    rho = 0.5  # Correlation of signal covariates (amongst themselves), and noise.
    sig = 0.01  # Controlling interaction vs main signals.
    # Setting it this way generates comparable main
    # and interaction signals
    weights = 2.5  # Group Lasso weights
    s_inter = 5  # Number of true interactions
    p_nl = 20  # Number of nonlinear covariates
    parallel = False

    for i in range(start, end):
        print(i, "th simulation done")
        np.random.seed(i+10000)
        for SNR in parameter_list:
            while True:
                # Generating a (X, Y) pair, and corresponding basis expansion
                # The 'weakhierarchy' argument is overridden by setting
                # `active_inter_list`.
                (design, data_interaction, Y, Y_mean, data_combined,
                 groups, active, active_inter_adj, active_inter_list, gamma) \
                    = (generate_gaussian_instance_nonlinear_interaction_simple
                       (n=500, p_nl=p_nl, rho=rho, full_corr=False, rho_noise=rho,
                        SNR=SNR, nknots=6, degree=2, interaction_signal=sig,
                        random_signs=False, scale=True, center=False,
                        structure='weakhierarchy', s_interaction=s_inter, intercept=True,
                        active_inter_list=active_inter_list_true, return_gamma=True))

                # Performing Naive inference using 'all pairs'
                coverages, lengths, selected_inter, p_values, targets, idx \
                    = naive_inference_inter(X=design, Y=Y, groups=groups,
                                            Y_mean=Y_mean, const=const,
                                            n_features=20, interactions=data_interaction,
                                            weight_frac=weights, level=0.9,
                                            mode='weakhierarchy',
                                            solve_only=False, continued=False,
                                            parallel=parallel, p_val=True,
                                            return_pivot=True, intercept=True,
                                            target_ids=None)

                noselection = coverages is None

                # Continue if Naive yields a nonempty group lasso selection
                # (this is almost always the case)
                if not noselection:
                    # Performing data splitting using 'all pairs'
                    (coverages_ds, lengths_ds, selected_inter_ds,
                     p_values_ds, targets_ds, idx_ds) \
                        = data_splitting_inter(X=design, Y=Y, groups=groups,
                                               Y_mean=Y_mean, const=const,
                                               n_features=20,
                                               interactions=data_interaction,
                                               proportion=0.5,
                                               weight_frac=weights, level=0.9,
                                               mode='weakhierarchy',
                                               solve_only=False, continued=False,
                                               parallel=parallel,
                                               p_val=True,
                                               target_ids=None)
                    noselection = coverages_ds is None

                # Continue if data splitting yields a nonempty group lasso selection
                # (this is almost always the case)
                if not noselection:
                    # Performing MLE using 'all pairs'
                    coverages_MLE, lengths_MLE, selected_inter_MLE, p_values_MLE, targets_MLE, idx_MLE \
                        = (MLE_inference_inter
                           (X=design, Y=Y, Y_mean=Y_mean, groups=groups,
                            n_features=p_nl, interactions=data_interaction,
                            intercept=True, proportion=0.5, weight_frac=weights,
                            level=0.9, mode='weakhierarchy', solve_only=False,
                            continued=False, parallel=parallel, p_val=True,
                            target_ids=None))
                    noselection = coverages_MLE is None

                # Collect results if all three methods yields
                # nonempty first-stage selection
                if not noselection:
                    # Naive
                    oper_char["coverage rate"].append(np.mean(coverages))
                    oper_char["avg length"].append(np.mean(lengths))
                    oper_char["method"].append('Naive')
                    oper_char["signal"].append(sig)
                    oper_char["rho"].append(rho)
                    oper_char["SNR"].append(SNR)
                    pval_dict[SNR]['Naive'] += (p_values)
                    oper_char["power"].append(calculate_power(p_values, targets, 0.1))
                    update_targets(dict=target_dict,
                                   true_inter_list=None,
                                   targets=targets, parameter=SNR,
                                   method="Naive", idx=idx)

                    # Data splitting
                    oper_char["coverage rate"].append(np.mean(coverages_ds))
                    oper_char["avg length"].append(np.mean(lengths_ds))
                    oper_char["method"].append('Data Splitting')
                    oper_char["signal"].append(sig)
                    oper_char["rho"].append(rho)
                    oper_char["SNR"].append(SNR)
                    pval_dict[SNR]['Data Splitting'] += (p_values_ds)
                    oper_char["power"].append(calculate_power(p_values_ds, targets_ds, 0.1))
                    update_targets(dict=target_dict,
                                   true_inter_list=None,
                                   targets=targets_ds, parameter=SNR,
                                   method="Data Splitting", idx=idx_ds)

                    # MLE
                    oper_char["coverage rate"].append(np.mean(coverages_MLE))
                    oper_char["avg length"].append(np.mean(lengths_MLE))
                    oper_char["method"].append('MLE')
                    oper_char["signal"].append(sig)
                    oper_char["rho"].append(rho)
                    oper_char["SNR"].append(SNR)
                    pval_dict[SNR]['MLE'] += (p_values_MLE)
                    oper_char["power"].append(calculate_power(p_values_MLE,
                                                              targets_MLE,
                                                              0.1))
                    update_targets(dict=target_dict,
                                   true_inter_list=None,
                                   targets=targets_MLE, parameter=SNR,
                                   method="MLE", idx=idx_MLE)

                    break
    return oper_char, pval_dict, target_dict


def combine_lists(L1):
    combined_dict = {}

    for dic in L1:
        for key, value in dic.items():
            if key not in combined_dict:
                combined_dict[key] = []
            combined_dict[key].extend(value)

    return combined_dict

def combine_nested_lists(L1):
    combined_dict = {}

    for outer_dict in L1:
        for outer_key, inner_dict in outer_dict.items():
            if outer_key not in combined_dict:
                combined_dict[outer_key] = {}
            for inner_key, value_list in inner_dict.items():
                if inner_key not in combined_dict[outer_key]:
                    combined_dict[outer_key][inner_key] = []
                combined_dict[outer_key][inner_key].extend(value_list)

    return combined_dict
import time, sys, joblib, os
sys.path.append('/home/yilingh/SI-Interaction')
import multiprocessing as mp

from selectinf.Simulation.H1.nonlinear_H1_helpers import *

def predict(beta_hat, X_test):
    return X_test.dot(beta_hat)

def generate_test(Y_mean, noise_sd):
    n = Y_mean.shape[0]
    Y_test = Y_mean + np.random.normal(size=(n,), scale=noise_sd)

    return Y_test

def vary_sparsity(start, end, dir):
    # A dictionary recording simulation results and metrics
    oper_char = {}
    oper_char["coverage rate"] = []
    oper_char["avg length"] = []
    oper_char["method"] = []
    # oper_char["MSE"] = []
    oper_char["prop"] = []
    oper_char["rho"] = []
    oper_char["sd_y"] = []
    oper_char["main signal"] = []
    oper_char["target"] = []
    oper_char["power"] = []

    MSE_dict = {}
    MSE_dict["MSE"] = []
    MSE_dict["method"] = []
    MSE_dict["sd_y"] = []

    # Dictionary of projected targets, over all simulation parameters
    target_dict = {}
    target_dict["parameter"] = []
    target_dict["target"] = []
    target_dict["target id"] = []
    target_dict["method"] = []
    target_dict["index"] = []
    target_dict["pivot"] = []
    target_dict["pval"] = []

    # A dictionary recording p-values for each true interaction
    # over all simulation results.
    # Each simulation parameter (here parameter_list contain a list of main signal strengths
    # to be considered) has a corresponding dictionary of results
    parameter_list = np.array([4, 2, 1, 0.5])  # np.array([0.5, 2, 5, 10])
    pval_dict = {}
    for x in parameter_list:
        pval_dict[x] = {}
        for m in ['Naive', 'Data Splitting', 'MLE']:
            pval_dict[x][m] = []

    # Group lasso solver constructor
    const = group_lasso.gaussian
    active_inter_list_true = np.array([[0, 1], [1, 2], [2, 4], [1, 5], [2, 6]])
    active_inter_list_true_list = [(x[0], x[1]) for x in active_inter_list_true]

    # p = 50
    rho = 0.5  # Correlation of signal covariates (amongst themselves), and noise.
    sig = 2  # Controlling interaction vs main signals.
    # Setting it this way generates comparable main
    # and interaction signals (sig = 2 works )
    weights = 0.05  # Group Lasso weights
    s_inter = 5  # Number of true interactions
    p_nl = 20  # Number of nonlinear covariates
    n = 200
    root_n_scaled = False
    main_sig = 2
    prop = 0.9

    ds_rank_def_count = {sd_y: 0 for sd_y in parameter_list}

    for sd_y in parameter_list:
        for i in range(start, end):
            np.random.seed(i + 1000)
            # MSE_set = False

            print(sd_y, i, "th simulation done")
            print("seed:", i + 1000)
            # Generating a (X, Y) pair, and corresponding basis expansion
            # The 'weakhierarchy' argument is overridden by setting
            # `active_inter_list`.
            (design, data_interaction, Y, Y_mean, data_combined,
             groups, active, active_inter_adj, active_inter_list, gamma) \
                = (generate_gaussian_instance_nonlinear_interaction_simple
                   (n=n, p_nl=p_nl, rho=rho, full_corr=False, rho_noise=rho,
                    SNR=None, main_signal=main_sig, noise_sd=sd_y,
                    nknots=6, degree=2, interaction_signal=sig,
                    random_signs=False, scale=root_n_scaled, center=False,
                    structure='weakhierarchy', s_interaction=s_inter,
                    intercept=True, active_inter_list=active_inter_list_true,
                    return_gamma=True))
            print("SD(Y): ", np.std(Y))
            Y_test = Y_mean + np.random.normal(size=(n,), scale=sd_y)

            # Performing Naive inference using 'all pairs'
            (coverages, lengths, selected_inter, p_values, pivots, targets, idx,
             beta_hat_naive) \
                = naive_inference_inter(X=design, Y=Y, groups=groups,
                                        Y_mean=Y_mean, const=const,
                                        n_features=20, interactions=data_interaction,
                                        weight_frac=weights, level=0.9, mode='weakhierarchy',
                                        solve_only=False, continued=False,
                                        parallel=False, p_val=True,
                                        return_pivot=True, intercept=True,
                                        target_ids=None,
                                        root_n_scaled=root_n_scaled)

            noselection_naive = coverages is None

            if not noselection_naive:
                # Naive
                oper_char["coverage rate"].append(np.mean(coverages))
                oper_char["avg length"].append(np.mean(lengths))
                oper_char["method"].append('Naive')
                oper_char["sd_y"].append(sd_y)
                oper_char["main signal"].append(main_sig)
                oper_char["prop"].append(prop)
                oper_char["rho"].append(rho)
                # oper_char["MSE"].append(MSE_naive)
                # oper_char["SNR"].append(SNR)
                oper_char["target"].append(targets[0])
                pval_dict[sd_y]['Naive'] += (p_values)
                oper_char["power"].append(calculate_power(p_values, targets, 0.1))
                update_targets(dict=target_dict,
                               true_inter_list=None,
                               targets=targets, parameter=sd_y,
                               method="Naive", idx=idx,
                               pvals=p_values,
                               pivots=pivots, sim_idx=i)

            # Continue if Naive yields a nonempty group lasso selection
            # (this is almost always the case)
            # Performing data splitting using 'all pairs'
            # DS solve only
            ds_rank_def = False
            (nonzero_ds, selected_groups_ds, subset_select_ds, beta_hat_ds) \
                = data_splitting_inter(X=design, Y=Y, groups=groups,
                                       Y_mean=Y_mean, const=const,
                                       n_features=20, interactions=data_interaction,
                                       proportion=prop,
                                       weight_frac=weights, level=0.9,
                                       mode='weakhierarchy',
                                       solve_only=True, continued=False,
                                       parallel=False, p_val=True,
                                       target_ids=None,
                                       root_n_scaled=root_n_scaled)
            if nonzero_ds.sum() + 1 >= n - subset_select_ds.sum():
                ds_rank_def = True
                ds_rank_def_count[sd_y] += 1

            if not ds_rank_def:
                (coverages_ds, lengths_ds, selected_inter_ds,
                 p_values_ds, pivots_ds, targets_ds, idx_ds, beta_hat_ds) \
                    = data_splitting_inter(X=design, Y=Y, groups=groups,
                                           Y_mean=Y_mean, const=const,
                                           n_features=20, interactions=data_interaction,
                                           proportion=prop,
                                           weight_frac=weights, level=0.9,
                                           mode='weakhierarchy',
                                           solve_only=False, continued=True,
                                           parallel=False, p_val=True,
                                           target_ids=None,
                                           root_n_scaled=root_n_scaled,
                                           subset_cont=subset_select_ds,
                                           nonzero_cont=nonzero_ds,
                                           selected_groups_cont=selected_groups_ds,
                                           soln_cont=beta_hat_ds)
                noselection_ds = coverages_ds is None

            if not ds_rank_def and not noselection_ds:
                # Data splitting
                oper_char["coverage rate"].append(np.mean(coverages_ds))
                oper_char["avg length"].append(np.mean(lengths_ds))
                oper_char["method"].append('Data Splitting')
                oper_char["sd_y"].append(sd_y)
                oper_char["main signal"].append(main_sig)
                oper_char["prop"].append(prop)
                oper_char["rho"].append(rho)
                # oper_char["MSE"].append(MSE_ds)
                # oper_char["SNR"].append(SNR)
                oper_char["target"].append(targets_ds[0])
                pval_dict[sd_y]['Data Splitting'] += (p_values_ds)
                oper_char["power"].append(calculate_power(p_values_ds, targets_ds, 0.1))
                update_targets(dict=target_dict,
                               true_inter_list=None,
                               targets=targets_ds, parameter=sd_y,
                               method="Data Splitting", idx=idx_ds,
                               pvals=p_values_ds,
                               pivots=pivots_ds, sim_idx=i)

            # Continue if data splitting yields a nonempty group lasso selection
            # (this is almost always the case)
            # Performing MLE using 'all pairs'
            (coverages_MLE, lengths_MLE, selected_inter_MLE, p_values_MLE,
             pivots_MLE, targets_MLE, idx_MLE, beta_hat_MLE) \
                = (MLE_inference_inter
                   (X=design, Y=Y, Y_mean=Y_mean, groups=groups,
                    n_features=p_nl, interactions=data_interaction,
                    intercept=True,
                    # randomizer_sd_const=tau,
                    proportion=prop,
                    weight_frac=weights,
                    level=0.9, mode='weakhierarchy', solve_only=False,
                    continued=False, parallel=False, p_val=True,
                    target_ids=None,
                    root_n_scaled=root_n_scaled))
            noselection_MLE = coverages_MLE is None

            # Collect results if all three methods yields
            # nonempty first-stage selection
            if not noselection_MLE:
                # MLE
                oper_char["coverage rate"].append(np.mean(coverages_MLE))
                oper_char["avg length"].append(np.mean(lengths_MLE))
                oper_char["method"].append('MLE')
                oper_char["sd_y"].append(sd_y)
                oper_char["main signal"].append(main_sig)
                oper_char["prop"].append(prop)
                oper_char["rho"].append(rho)
                # oper_char["MSE"].append(MSE_MLE)
                # oper_char["SNR"].append(SNR)
                oper_char["target"].append(targets_MLE[0])
                pval_dict[sd_y]['MLE'] += (p_values_MLE)
                oper_char["power"].append(calculate_power(p_values_MLE,
                                                          targets_MLE,
                                                          0.1))
                update_targets(dict=target_dict,
                               true_inter_list=None,
                               targets=targets_MLE, parameter=sd_y,
                               method="MLE", idx=idx_MLE,
                               pvals=p_values_MLE,
                               pivots=pivots_MLE, sim_idx=i)

            # Set MSE
            MSE_naive = np.mean((Y_test - predict(beta_hat_naive, design)) ** 2)
            MSE_dict["MSE"].append(MSE_naive)
            MSE_dict["method"].append("Naive")
            MSE_dict["sd_y"].append(sd_y)

            MSE_ds = np.mean((Y_test - predict(beta_hat_ds, design)) ** 2)
            MSE_dict["MSE"].append(MSE_ds)
            MSE_dict["method"].append("Data Splitting")
            MSE_dict["sd_y"].append(sd_y)

            MSE_MLE = np.mean((Y_test - predict(beta_hat_MLE, design)) ** 2)
            MSE_dict["MSE"].append(MSE_MLE)
            MSE_dict["method"].append("MLE")
            MSE_dict["sd_y"].append(sd_y)
            MSE_set = True

            joblib.dump([ds_rank_def_count, target_dict, pval_dict, MSE_dict, oper_char],
                        f'{dir}_{start}_{end}.pkl', compress=1)

if __name__ == '__main__':
    # Get the script's directory
    script_directory = os.path.dirname(os.path.abspath(__file__))
    # Change the working directory to the script's directory
    os.chdir(script_directory)
    current_directory = os.getcwd()
    print("Current Working Directory:", current_directory)

    argv = sys.argv
    ## sys.argv: [something, start, end, ncores]
    start, end = int(argv[1]), int(argv[2])
    dir = str(argv[3])

    vary_sparsity(start, end, dir)
import numpy as np
import pandas as pd
import time
# from multiprocess import Pool
import sys

# For greatlakes simulations
sys.path.append('/home/yilingh/SI-Interaction')

from selectinf.group_lasso_query import (group_lasso,
                                         split_group_lasso)

from selectinf.Simulation.test_group_lasso_simulation import (calculate_F1_score,
                                                              naive_inference,
                                                              randomization_inference,
                                                              data_splitting)

def generate_polynomial(n = 2000, p = 10, s = 3, order = 3, center=True, scale=True,
                        signal = 5., signal_fac=1., random_signs = True):
    data = np.zeros((n, p*order))
    for i in range(p):
        x_i = np.random.normal(size = (n,))
        data[:,3*i:3*i+3] = np.array([x_i,x_i**2,x_i**3]).T

    groups = np.arange(p).repeat(order)
    group_labels = np.unique(groups)

    # Assigning active groups
    # Assuming no intercept
    group_active = np.random.choice(np.arange(p), s, replace=False)

    beta = np.zeros(data.shape[1])
    if signal_fac is not None:
        signal = np.sqrt(signal_fac * 2 * np.log(data.shape[1]))
        print(signal)

    signal = np.atleast_1d(signal)

    active = np.isin(groups, group_active)

    if signal.shape == (1,):
        beta[active] = signal[0]
    else:
        beta[active] = np.linspace(signal[0], signal[1], active.sum())
    if random_signs:
        beta[active] *= (2 * np.random.binomial(1, 0.5, size=(active.sum(),)) - 1.)
    beta /= np.sqrt(n)

    if center:
        data -= data.mean(0)[None, :]

    if scale:
        # ----SCALE----
        # scales X by sqrt(n) and sd
        # if we need original X, uncomment the following line
        # X_raw = X
        # ----SCALE----
        scaling = data.std(0) * np.sqrt(n)
        data /= scaling[None, :]
        beta *= np.sqrt(n)

    Y = (data.dot(beta)) + np.random.normal(size = (n,))

    return data, Y, beta, groups

def comparison_cubic_spline_vary_s(n=2000,
                                   signal_fac=0.01,
                                   level=0.90,
                                   range=range(0,100)):
    """
    Compare to R randomized lasso
    """

    # Operating characteristics
    oper_char = {}
    oper_char["sparsity size"] = []
    oper_char["coverage rate"] = []
    oper_char["avg length"] = []
    oper_char["method"] = []
    oper_char["F1 score"] = []
    # oper_char["runtime"] = []

    confint_df = pd.DataFrame()

    for s_group in [5,8,10]:  # [0.01, 0.03, 0.06, 0.1]:
        for i in range:
            np.random.seed(i)

            const, const_split = group_lasso.gaussian, split_group_lasso.gaussian

            while True:  # run until we get some selection
                X, Y, beta, groups = generate_polynomial(p=30, s=s_group, signal_fac=signal_fac)
                # print(X)

                n, p = X.shape
                noselection = False  # flag for a certain method having an empty selected set

                if not noselection:
                    # MLE inference
                    coverage, length, beta_target, nonzero, conf_low, conf_up = \
                        randomization_inference(X=X, Y=Y, n=n, p=p, randomizer_scale=1.,
                                                beta=beta, groups=groups, weight_frac=1)
                    noselection = (coverage is None)

                if not noselection:
                    # data splitting
                    coverage_ds, lengths_ds, conf_low_ds, conf_up_ds, nonzero_ds, beta_target_ds = \
                        data_splitting(X=X, Y=Y, n=n, p=p, beta=beta, groups=groups,
                                       proportion=0.67, level=0.9, weight_frac=1)
                    noselection = (coverage_ds is None)

                if not noselection:
                    # naive inference
                    coverage_naive, lengths_naive, nonzero_naive, conf_low_naive, conf_up_naive, \
                    beta_target_naive = \
                        naive_inference(X=X, Y=Y, groups=groups,
                                        beta=beta, const=const,
                                        n=n, level=level, weight_frac=1)
                    noselection = (coverage_naive is None)

                if not noselection:
                    # F1 scores
                    # F1_s = calculate_F1_score(beta, selection=nonzero_s)
                    F1 = calculate_F1_score(beta, selection=nonzero)
                    F1_ds = calculate_F1_score(beta, selection=nonzero_ds)
                    F1_naive = calculate_F1_score(beta, selection=nonzero_naive)

                    # MLE coverage
                    oper_char["sparsity size"].append(s_group)
                    oper_char["coverage rate"].append(np.mean(coverage))
                    oper_char["avg length"].append(np.mean(length))
                    oper_char["F1 score"].append(F1)
                    oper_char["method"].append('MLE')
                    df_MLE = pd.concat([pd.DataFrame(np.ones(nonzero.sum()) * i),
                                        pd.DataFrame(beta_target),
                                        pd.DataFrame(conf_low),
                                        pd.DataFrame(conf_up),
                                        pd.DataFrame(beta[nonzero] != 0),
                                        pd.DataFrame(np.ones(nonzero.sum()) * s_group),
                                        pd.DataFrame(np.ones(nonzero.sum()) * F1),
                                        pd.DataFrame(["MLE"] * nonzero.sum())
                                        ], axis=1)
                    confint_df = pd.concat([confint_df, df_MLE], axis=0)

                    # Data splitting coverage
                    oper_char["sparsity size"].append(s_group)
                    oper_char["coverage rate"].append(np.mean(coverage_ds))
                    oper_char["avg length"].append(np.mean(lengths_ds))
                    oper_char["F1 score"].append(F1_ds)
                    oper_char["method"].append('Data splitting')
                    # oper_char["runtime"].append(0)
                    df_ds = pd.concat([pd.DataFrame(np.ones(nonzero_ds.sum()) * i),
                                       pd.DataFrame(beta_target_ds),
                                       pd.DataFrame(conf_low_ds),
                                       pd.DataFrame(conf_up_ds),
                                       pd.DataFrame(beta[nonzero_ds] != 0),
                                       pd.DataFrame(np.ones(nonzero_ds.sum()) * s_group),
                                       pd.DataFrame(np.ones(nonzero_ds.sum()) * F1_ds),
                                       pd.DataFrame(["Data splitting"] * nonzero_ds.sum())
                                       ], axis=1)
                    confint_df = pd.concat([confint_df, df_ds], axis=0)

                    # Naive coverage
                    oper_char["sparsity size"].append(s_group)
                    oper_char["coverage rate"].append(np.mean(coverage_naive))
                    oper_char["avg length"].append(np.mean(lengths_naive))
                    oper_char["F1 score"].append(F1_naive)
                    oper_char["method"].append('Naive')
                    df_naive = pd.concat([pd.DataFrame(np.ones(nonzero_naive.sum()) * i),
                                          pd.DataFrame(beta_target_naive),
                                          pd.DataFrame(conf_low_naive),
                                          pd.DataFrame(conf_up_naive),
                                          pd.DataFrame(beta[nonzero_naive] != 0),
                                          pd.DataFrame(np.ones(nonzero_naive.sum()) * s_group),
                                          pd.DataFrame(np.ones(nonzero_naive.sum()) * F1_naive),
                                          pd.DataFrame(["Naive"] * nonzero_naive.sum())
                                          ], axis=1)
                    confint_df = pd.concat([confint_df, df_naive], axis=0)

                    break  # Go to next iteration if we have some selection

    oper_char_df = pd.DataFrame.from_dict(oper_char)
    oper_char_df.to_csv('polynomial_vary_sparsity' + str(range.start) + '_' + str(range.stop) + '.csv', index=False)
    colnames = ['Index'] + ['target'] + ['LCB'] + ['UCB'] + ['TP'] + ['sparsity size'] + ['F1'] + ['Method']
    confint_df.columns = colnames
    confint_df.to_csv('polynomial_CI_vary_sparsity' + str(range.start) + '_' + str(range.stop) + '.csv', index=False)

if __name__ == '__main__':
    argv = sys.argv
    ## sys.argv: [something, start, end]
    start, end = int(argv[1]), int(argv[2])
    #p_l, s_l, order, nknots = 0, 0, 3, 3#int(argv[3]), int(argv[4]), int(argv[5]), int(argv[6])
    print("start:", start, ", end:", end)
    comparison_cubic_spline_vary_s(range=range(start, end))

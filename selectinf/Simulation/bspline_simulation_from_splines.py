import numpy as np
import pandas as pd
from selectinf.Simulation.spline import b_spline
from selectinf.Simulation.spline_instance import generate_gaussian_instance_nonlinear, generate_gaussian_instance_from_bspline
from selectinf.group_lasso_query import (group_lasso,
                                         split_group_lasso)
import regreg.api as rr
from selectinf.base import selected_targets
from selectinf.base import restricted_estimator
import scipy.stats
import sys

from selectinf.Simulation.test_group_lasso_simulation import (calculate_F1_score,
                                                              naive_inference,
                                                              randomization_inference,
                                                              randomization_inference_fast,
                                                              data_splitting)

def comparison_b_spline_vary_cplx(level=0.90, range=range(0,100)):
    """
    Compare to R randomized lasso
    """

    # Operating characteristics
    oper_char = {}
    oper_char["complexity"] = []
    oper_char["coverage rate"] = []
    oper_char["avg length"] = []
    oper_char["method"] = []
    oper_char["F1 score"] = []

    for complexity in [(8,1), (8,2), (8,3)]:  # [0.01, 0.03, 0.06, 0.1]:
        nknots = complexity[0]
        degree = complexity[1]
        for i in range:
            print(i)
            #np.random.seed(i)
            const = group_lasso.gaussian

            while True:  # run until we get some selection
                design, Y, beta, groups = \
                    generate_gaussian_instance_from_bspline(n=500, p_nl=50, p_l=0,
                                                            s_nl=2, s_l=0,
                                                            nknots=nknots, degree=degree,
                                                            signal_fac=1, SNR=0.1,
                                                            center=False, scale=True, random_signs=True,
                                                            intercept=False)
                print(design.shape)

                n, p = design.shape
                noselection = False  # flag for a certain method having an empty selected set

                if not noselection:
                    # MLE inference
                    coverage, length, beta_target, nonzero, conf_low, conf_up = \
                        randomization_inference_fast(X=design, Y=Y, n=n, p=p, proportion=0.5,
                                                     beta=beta, groups=groups, weight_frac=1)
                    # print(MLE_runtime)
                    noselection = (coverage is None)

                if not noselection:
                    # data splitting
                    coverage_ds, lengths_ds, conf_low_ds, conf_up_ds, nonzero_ds, beta_target_ds = \
                        data_splitting(X=design, Y=Y, n=n, p=p, beta=beta, groups=groups,
                                       proportion=0.5, level=0.9, weight_frac=1)
                    noselection = (coverage_ds is None)

                if not noselection:
                    # naive inference
                    coverage_naive, lengths_naive, nonzero_naive, conf_low_naive, conf_up_naive, \
                        beta_target_naive = \
                        naive_inference(X=design, Y=Y, groups=groups,
                                        beta=beta, const=const,
                                        n=n, level=0.9, weight_frac=1)
                    noselection = (coverage_naive is None)

                if not noselection:
                    # F1 scores
                    # F1_s = calculate_F1_score(beta, selection=nonzero_s)
                    F1 = calculate_F1_score(beta, selection=nonzero)
                    F1_ds = calculate_F1_score(beta, selection=nonzero_ds)
                    F1_naive = calculate_F1_score(beta, selection=nonzero_naive)

                    # MLE coverage
                    # oper_char["sparsity size"].append(s_group)
                    oper_char["complexity"].append("(" + str(nknots) + "," +
                                                   str(degree) + ")")
                    oper_char["coverage rate"].append(np.mean(coverage))
                    oper_char["avg length"].append(np.mean(length))
                    oper_char["F1 score"].append(F1)
                    oper_char["method"].append('MLE')

                    # Data splitting coverage
                    # oper_char["sparsity size"].append(s_group)
                    oper_char["complexity"].append("(" + str(nknots) + "," +
                                                   str(degree) + ")")
                    oper_char["coverage rate"].append(np.mean(coverage_ds))
                    oper_char["avg length"].append(np.mean(lengths_ds))
                    oper_char["F1 score"].append(F1_ds)
                    oper_char["method"].append('Data splitting')

                    # Naive coverage
                    # oper_char["sparsity size"].append(s_group)
                    oper_char["complexity"].append("(" + str(nknots) + "," +
                                                   str(degree) + ")")
                    oper_char["coverage rate"].append(np.mean(coverage_naive))
                    oper_char["avg length"].append(np.mean(lengths_naive))
                    oper_char["F1 score"].append(F1_naive)
                    oper_char["method"].append('Naive')

                    break  # Go to next iteration if we have some selection

    oper_char_df = pd.DataFrame.from_dict(oper_char)
    oper_char_df.to_csv('bspline_vary_complexity' + str(range.start) + '_' + str(range.stop) + '.csv',
                        index=False)

def comparison_b_spline_vary_SNR(level=0.90, range=range(0,100)):
    """
    Compare to R randomized lasso
    """

    # Operating characteristics
    oper_char = {}
    oper_char["complexity"] = []
    oper_char["SNR"] = []
    oper_char["coverage rate"] = []
    oper_char["avg length"] = []
    oper_char["method"] = []
    oper_char["F1 score"] = []

    complexity = (8,2)
    nknots = complexity[0]
    degree = complexity[1]
    for SNR in [0.1, 0.3, 0.5, 1, 1.5, 2]:
        for i in range:
            print(i)
            #np.random.seed(i)
            const = group_lasso.gaussian

            while True:  # run until we get some selection
                design, Y, beta, groups = \
                    generate_gaussian_instance_from_bspline(n=500, p_nl=50, p_l=0,
                                                            s_nl=2, s_l=0,
                                                            nknots=nknots, degree=degree,
                                                            signal_fac=1, SNR=SNR,
                                                            center=False, scale=True, random_signs=True,
                                                            intercept=False)
                print(design.shape)

                n, p = design.shape
                noselection = False  # flag for a certain method having an empty selected set

                if not noselection:
                    # MLE inference
                    coverage, length, beta_target, nonzero, conf_low, conf_up = \
                        randomization_inference_fast(X=design, Y=Y, n=n, p=p, proportion=0.5,
                                                     beta=beta, groups=groups, weight_frac=1)
                    # print(MLE_runtime)
                    noselection = (coverage is None)

                if not noselection:
                    # data splitting
                    coverage_ds, lengths_ds, conf_low_ds, conf_up_ds, nonzero_ds, beta_target_ds = \
                        data_splitting(X=design, Y=Y, n=n, p=p, beta=beta, groups=groups,
                                       proportion=0.5, level=0.9, weight_frac=1)
                    noselection = (coverage_ds is None)

                if not noselection:
                    # naive inference
                    coverage_naive, lengths_naive, nonzero_naive, conf_low_naive, conf_up_naive, \
                        beta_target_naive = \
                        naive_inference(X=design, Y=Y, groups=groups,
                                        beta=beta, const=const,
                                        n=n, level=0.9, weight_frac=1)
                    noselection = (coverage_naive is None)

                if not noselection:
                    # F1 scores
                    # F1_s = calculate_F1_score(beta, selection=nonzero_s)
                    F1 = calculate_F1_score(beta, selection=nonzero)
                    F1_ds = calculate_F1_score(beta, selection=nonzero_ds)
                    F1_naive = calculate_F1_score(beta, selection=nonzero_naive)

                    # MLE coverage
                    # oper_char["sparsity size"].append(s_group)
                    oper_char["complexity"].append("(" + str(nknots) + "," +
                                                   str(degree) + ")")
                    oper_char["SNR"].append(SNR)
                    oper_char["coverage rate"].append(np.mean(coverage))
                    oper_char["avg length"].append(np.mean(length))
                    oper_char["F1 score"].append(F1)
                    oper_char["method"].append('MLE')

                    # Data splitting coverage
                    # oper_char["sparsity size"].append(s_group)
                    oper_char["complexity"].append("(" + str(nknots) + "," +
                                                   str(degree) + ")")
                    oper_char["SNR"].append(SNR)
                    oper_char["coverage rate"].append(np.mean(coverage_ds))
                    oper_char["avg length"].append(np.mean(lengths_ds))
                    oper_char["F1 score"].append(F1_ds)
                    oper_char["method"].append('Data splitting')

                    # Naive coverage
                    # oper_char["sparsity size"].append(s_group)
                    oper_char["complexity"].append("(" + str(nknots) + "," +
                                                   str(degree) + ")")
                    oper_char["SNR"].append(SNR)
                    oper_char["coverage rate"].append(np.mean(coverage_naive))
                    oper_char["avg length"].append(np.mean(lengths_naive))
                    oper_char["F1 score"].append(F1_naive)
                    oper_char["method"].append('Naive')

                    break  # Go to next iteration if we have some selection

    oper_char_df = pd.DataFrame.from_dict(oper_char)
    oper_char_df.to_csv('bspline_vary_SNR_('+ str(nknots) + ','+ str(degree) + ')_'
                        + str(range.start) + '_' + str(range.stop) + '.csv',
                        index=False)

if __name__ == '__main__':
    argv = sys.argv
    ## sys.argv: [something, start, end, p_l, s_l, order, knots]
    start, end = 0, 30#int(argv[1]), int(argv[2])
    print("start:", start, ", end:", end)
    comparison_b_spline_vary_SNR(range=range(start, end))
import numpy as np
import pandas as pd
import time
# from multiprocess import Pool
import sys, scipy

# For greatlakes simulations
sys.path.append('/home/yilingh/SI-Interaction')

from selectinf.group_lasso_query import (group_lasso,
                                         split_group_lasso)

from selectinf.Simulation.spline import b_spline
from selectinf.Simulation.spline_instance import generate_gaussian_instance_nonlinear
import regreg.api as rr
from selectinf.base import selected_targets
from selectinf.base import restricted_estimator

from selectinf.Simulation.test_group_lasso_simulation import (calculate_F1_score)


def randomization_inference_spline(design, Y, n, p, Y_mean, groups,
                                   randomizer_scale=1.,
                                   weight_frac=1.25, level=0.9, ridge_term=1.):
    sigma_ = np.std(Y)
    if n > p:
        dispersion = np.linalg.norm(Y - design.dot(np.linalg.pinv(design).dot(Y))) ** 2 / (n - p)
    else:
        dispersion = sigma_ ** 2

    sigma_ = np.sqrt(dispersion)

    weights = dict([(i, weight_frac * sigma_ * np.sqrt(2 * np.log(p))) for i in np.unique(groups)])

    conv = group_lasso.gaussian(X=design,
                                Y=Y,
                                groups=groups,
                                weights=weights,
                                useJacobian=True,
                                ridge_term=ridge_term)

    signs, _ = conv.fit()
    nonzero = (signs != 0)

    def solve_target_restricted():
        loglike = rr.glm.gaussian(design, Y_mean)
        # For LASSO, this is the OLS solution on X_{E,U}
        _beta_unpenalized = restricted_estimator(loglike,
                                                 nonzero)
        return _beta_unpenalized

    if nonzero.sum() > 0:
        print("MLE |E|:", nonzero.sum())
        conv.setup_inference(dispersion=dispersion)

        target_spec = selected_targets(conv.loglike,
                                       conv.observed_soln,
                                       dispersion=dispersion)

        result, _ = conv.inference(target_spec,
                                   method='selective_MLE',
                                   level=level)

        pval = result['pvalue']
        intervals = np.asarray(result[['lower_confidence',
                                       'upper_confidence']])

        beta_target = solve_target_restricted()

        coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])

        selected_groups = conv.selection_variable['active_groups']
        active_flag = np.zeros(np.unique(groups).shape[0])
        active_flag[selected_groups] = 1.

        return coverage, (intervals[:, 1] - intervals[:, 0]), beta_target, \
            nonzero, intervals[:, 0], intervals[:, 1], active_flag
    return None, None, None, None, None, None, None

def randomization_inference_spline_hessian(design, Y, n, p, Y_mean, groups,
                                           proportion=0.5,
                                           weight_frac=1.25, level=0.9):
    hess = design.T @ design * (1 - proportion) / proportion

    sigma_ = np.std(Y)
    if n > p:
        dispersion = np.linalg.norm(Y - design.dot(np.linalg.pinv(design).dot(Y))) ** 2 / (n - p)
    else:
        dispersion = sigma_ ** 2

    sigma_ = np.sqrt(dispersion)

    weights = dict([(i, weight_frac * sigma_ * np.sqrt(2 * np.log(p))) for i in np.unique(groups)])

    conv = split_group_lasso.gaussian(X=design,
                                      Y=Y,
                                      groups=groups,
                                      weights=weights,
                                      useJacobian=True,
                                      proportion=proportion,
                                      cov_rand=hess)

    signs, _ = conv.fit()
    nonzero = (signs != 0)

    def solve_target_restricted():
        loglike = rr.glm.gaussian(design, Y_mean)
        # For LASSO, this is the OLS solution on X_{E,U}
        _beta_unpenalized = restricted_estimator(loglike,
                                                 nonzero)
        return _beta_unpenalized

    if nonzero.sum() > 0:
        print("MLE |E|:", nonzero.sum())
        conv.setup_inference(dispersion=dispersion)

        target_spec = selected_targets(conv.loglike,
                                       conv.observed_soln,
                                       dispersion=dispersion)

        result, _ = conv.inference(target_spec,
                                   method='selective_MLE',
                                   level=level)

        pval = result['pvalue']
        intervals = np.asarray(result[['lower_confidence',
                                       'upper_confidence']])

        beta_target = solve_target_restricted()

        coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])

        selected_groups = conv.selection_variable['active_groups']
        active_flag = np.zeros(np.unique(groups).shape[0])
        active_flag[selected_groups] = 1.

        return coverage, (intervals[:, 1] - intervals[:, 0]), beta_target, \
            nonzero, intervals[:, 0], intervals[:, 1], active_flag
    return None, None, None, None, None, None, None

def data_splitting_spline(X, Y, n, p, Y_mean, groups, weight_frac=1.25,
                   nonzero=None, subset_select=None,
                   proportion=0.5, level=0.9):
    if (nonzero is None) or (subset_select is None):
        # print("(Poisson Data Splitting) Selection done without carving")
        pi_s = proportion
        subset_select = np.zeros(n, np.bool_)
        subset_select[:int(pi_s * n)] = True
        n1 = subset_select.sum()
        n2 = n - n1
        np.random.shuffle(subset_select)
        X_S = X[subset_select, :]
        Y_S = Y[subset_select]

        # Selection on the first subset of data
        p = X.shape[1]
        sigma_ = np.std(Y_S)
        # weights = dict([(i, 0.5) for i in np.unique(groups)])
        weights = dict([(i, (n1/n)*weight_frac * sigma_ * np.sqrt(2 * np.log(p))) for i in np.unique(groups)])

        conv = group_lasso.gaussian(X=X_S,
                                    Y=Y_S,
                                    groups=groups,
                                    weights=weights,
                                    useJacobian=True,
                                    perturb=np.zeros(p),
                                    ridge_term=0.)

        signs, _ = conv.fit()
        # print("signs",  signs)
        nonzero = signs != 0

    n1 = subset_select.sum()
    n2 = n - n1

    if nonzero.sum() > 0:
        # Solving the inferential target
        def solve_target_restricted():
            loglike = rr.glm.gaussian(X, Y_mean)
            # For LASSO, this is the OLS solution on X_{E,U}
            _beta_unpenalized = restricted_estimator(loglike,
                                                     nonzero)
            return _beta_unpenalized

        target = solve_target_restricted()

        X_notS = X[~subset_select, :]
        Y_notS = Y[~subset_select]

        # E: nonzero flag

        X_notS_E = X_notS[:, nonzero]
        E_size = nonzero.sum()

        # Solve for the unpenalized MLE
        loglike = rr.glm.gaussian(X_notS, Y_notS)
        # For LASSO, this is the OLS solution on X_{E,U}
        beta_MLE_notS = restricted_estimator(loglike, nonzero)

        # Calculation the asymptotic covariance of the MLE
        dispersion_notS_E = np.linalg.norm(Y_notS - X_notS_E @ beta_MLE_notS) ** 2 / (n2 - E_size)
        f_info = X_notS_E.T @ X_notS_E
        cov = np.linalg.inv(f_info) * dispersion_notS_E

        # Standard errors
        sd = np.sqrt(np.diag(cov))

        # Normal quantiles
        qt_low = scipy.stats.t.ppf((1 - level) / 2, df=n2 - E_size)
        qt_up = scipy.stats.t.ppf(1 - (1 - level) / 2, df=n2 - E_size)
        assert np.abs(np.abs(qt_low) - np.abs(qt_up)) < 10e-6

        # Construct confidence intervals
        intervals_low = beta_MLE_notS + qt_low * sd
        intervals_up = beta_MLE_notS + qt_up * sd

        coverage = (target > intervals_low) * (target < intervals_up)

        selected_groups = conv.selection_variable['active_groups']
        active_flag = np.zeros(np.unique(groups).shape[0])
        active_flag[selected_groups] = 1.

        return (coverage, intervals_up - intervals_low,
                intervals_low, intervals_up, nonzero, target, active_flag)

    # If no variable selected, no inference
    return None, None, None, None, None, None, None

def naive_inference_spline(X, Y, groups, Y_mean, const,
                           n, weight_frac=1.25, level=0.9):
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

    conv = const(X=X,
                 Y=Y,
                 groups=groups,
                 weights=weights,
                 useJacobian=False,
                 perturb=np.zeros(p),
                 ridge_term=0.)

    signs, _ = conv.fit()
    nonzero = signs != 0

    if nonzero.sum() > 0:
        # E: nonzero flag
        X_E = X[:, nonzero]
        E_size = nonzero.sum()

        loglike = rr.glm.gaussian(X, Y)
        # For LASSO, this is the OLS solution on X_{E,U}
        beta_MLE = restricted_estimator(loglike, nonzero)

        def solve_target_restricted():
            loglike = rr.glm.gaussian(X, Y_mean)
            # For LASSO, this is the OLS solution on X_{E,U}
            _beta_unpenalized = restricted_estimator(loglike,
                                                     nonzero)
            return _beta_unpenalized

        target = solve_target_restricted()

        f_info = X_E.T @ X_E
        cov = np.linalg.inv(f_info) * dispersion

        # Standard errors
        sd = np.sqrt(np.diag(cov))

        # Normal quantiles
        qt_low = scipy.stats.t.ppf((1 - level) / 2, df=n-E_size)
        qt_up = scipy.stats.t.ppf(1 - (1 - level) / 2, df=n-E_size)
        assert np.abs(np.abs(qt_low) - np.abs(qt_up)) < 10e-6

        # Construct confidence intervals
        intervals_low = beta_MLE + qt_low * sd
        intervals_up = beta_MLE + qt_up * sd

        coverage = (target > intervals_low) * (target < intervals_up)

        selected_groups = conv.selection_variable['active_groups']
        active_flag = np.zeros(np.unique(groups).shape[0])
        active_flag[selected_groups] = 1.

        return (coverage, intervals_up - intervals_low, nonzero,
                intervals_low, intervals_up, target, active_flag)

    return None, None, None, None, None, None, None

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

    for complexity in [(8,1), (8,2), (8,3)]:#[(12,1), (12,2), (12,3), (12,4), (12,5)]:#[(10,1), (10,2), (10,3), (10,4)]: # # [0.01, 0.03, 0.06, 0.1]:
        nknots = complexity[0]
        degree = complexity[1]
        for i in range:
            print(i)
            #np.random.seed(i)
            const = group_lasso.gaussian

            while True:  # run until we get some selection
                design, Y, Y_mean, groups, active_flag = \
                    generate_gaussian_instance_nonlinear(n=1000, p_nl=100, p_l=0,
                                                         nknots=nknots, degree=degree,
                                                         center=False, scale=True, signal_fac=0.01,
                                                         noise_scale=np.sqrt(10))
                # print(X)

                n, p = design.shape
                print(n, p)
                noselection = False  # flag for a certain method having an empty selected set

                if not noselection:
                    # MLE inference
                    coverage, length, beta_target, nonzero, conf_low, conf_up, selected_groups = \
                        randomization_inference_spline_hessian(design=design, Y=Y, n=n, p=p, Y_mean=Y_mean,
                                                               groups=groups, weight_frac=1., level=0.9,
                                                               proportion=0.5)
                    noselection = (coverage is None)

                    #print("MLE beta:", beta_target)

                if not noselection:
                    # data splitting
                    (coverage_ds, lengths_ds, conf_low_ds, conf_up_ds,
                     nonzero_ds, beta_target_ds, selected_groups_ds) = \
                        data_splitting_spline(X=design, Y=Y, n=n, p=p, Y_mean=Y_mean, groups=groups,
                                              proportion=0.5, level=0.9, weight_frac=1.)
                    noselection = (coverage_ds is None)

                    #print("DS beta:", beta_target_ds)

                if not noselection:
                    # naive inference
                    (coverage_naive, lengths_naive, nonzero_naive,
                     conf_low_naive, conf_up_naive, beta_target_naive,
                     selected_groups_naive) = \
                        naive_inference_spline(X=design, Y=Y, groups=groups,
                                               Y_mean=Y_mean, const=const,
                                               n=n, level=level, weight_frac=1.)
                    noselection = (coverage_naive is None)
                    #print("naive beta:", beta_target_naive)

                if not noselection:
                    # F1 scores
                    # F1_s = calculate_F1_score(beta, selection=nonzero_s)
                    F1 = calculate_F1_score(active_flag, selection=selected_groups)
                    F1_ds = calculate_F1_score(active_flag, selection=selected_groups_ds)
                    F1_naive = calculate_F1_score(active_flag, selection=selected_groups_naive)

                    # MLE coverage
                    #oper_char["sparsity size"].append(s_group)
                    oper_char["complexity"].append("(" + str(nknots) + "," +
                                                   str(degree) + ")")
                    oper_char["coverage rate"].append(np.mean(coverage))
                    oper_char["avg length"].append(np.mean(length))
                    oper_char["F1 score"].append(F1)
                    oper_char["method"].append('MLE')

                    # Data splitting coverage
                    #oper_char["sparsity size"].append(s_group)
                    oper_char["complexity"].append("(" + str(nknots) + "," +
                                                   str(degree) + ")")
                    oper_char["coverage rate"].append(np.mean(coverage_ds))
                    oper_char["avg length"].append(np.mean(lengths_ds))
                    oper_char["F1 score"].append(F1_ds)
                    oper_char["method"].append('Data splitting')

                    # Naive coverage
                    #oper_char["sparsity size"].append(s_group)
                    oper_char["complexity"].append("(" + str(nknots) + "," +
                                                   str(degree) + ")")
                    oper_char["coverage rate"].append(np.mean(coverage_naive))
                    oper_char["avg length"].append(np.mean(lengths_naive))
                    oper_char["F1 score"].append(F1_naive)
                    oper_char["method"].append('Naive')

                    break  # Go to next iteration if we have some selection

    oper_char_df = pd.DataFrame.from_dict(oper_char)
    oper_char_df.to_csv('bspline_vary_complexity_nonlinear' + str(range.start) + '_' + str(range.stop) + '.csv',
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
                design, Y, Y_mean, groups, active_flag = \
                    generate_gaussian_instance_nonlinear(n=1000, p_nl=100, p_l=0,
                                                         nknots=nknots, degree=degree,
                                                         center=False, scale=True, SNR=SNR)
                # print(X)

                n, p = design.shape
                print(n, p)
                noselection = False  # flag for a certain method having an empty selected set

                if not noselection:
                    # MLE inference
                    coverage, length, beta_target, nonzero, conf_low, conf_up, selected_groups = \
                        randomization_inference_spline_hessian(design=design, Y=Y, n=n, p=p, Y_mean=Y_mean,
                                                               groups=groups, weight_frac=1., level=0.9,
                                                               proportion=0.5)
                    noselection = (coverage is None)

                    #print("MLE beta:", beta_target)

                if not noselection:
                    # data splitting
                    (coverage_ds, lengths_ds, conf_low_ds, conf_up_ds,
                     nonzero_ds, beta_target_ds, selected_groups_ds) = \
                        data_splitting_spline(X=design, Y=Y, n=n, p=p, Y_mean=Y_mean, groups=groups,
                                              proportion=0.5, level=0.9, weight_frac=1.)
                    noselection = (coverage_ds is None)

                    #print("DS beta:", beta_target_ds)

                if not noselection:
                    # naive inference
                    (coverage_naive, lengths_naive, nonzero_naive,
                     conf_low_naive, conf_up_naive, beta_target_naive,
                     selected_groups_naive) = \
                        naive_inference_spline(X=design, Y=Y, groups=groups,
                                               Y_mean=Y_mean, const=const,
                                               n=n, level=level, weight_frac=1.)
                    noselection = (coverage_naive is None)
                    #print("naive beta:", beta_target_naive)

                if not noselection:
                    # F1 scores
                    # F1_s = calculate_F1_score(beta, selection=nonzero_s)
                    F1 = calculate_F1_score(active_flag, selection=selected_groups)
                    F1_ds = calculate_F1_score(active_flag, selection=selected_groups_ds)
                    F1_naive = calculate_F1_score(active_flag, selection=selected_groups_naive)

                    # MLE coverage
                    #oper_char["sparsity size"].append(s_group)
                    oper_char["complexity"].append("(" + str(nknots) + "," +
                                                   str(degree) + ")")
                    oper_char["SNR"].append(SNR)
                    oper_char["coverage rate"].append(np.mean(coverage))
                    oper_char["avg length"].append(np.mean(length))
                    oper_char["F1 score"].append(F1)
                    oper_char["method"].append('MLE')

                    # Data splitting coverage
                    #oper_char["sparsity size"].append(s_group)
                    oper_char["complexity"].append("(" + str(nknots) + "," +
                                                   str(degree) + ")")
                    oper_char["SNR"].append(SNR)
                    oper_char["coverage rate"].append(np.mean(coverage_ds))
                    oper_char["avg length"].append(np.mean(lengths_ds))
                    oper_char["F1 score"].append(F1_ds)
                    oper_char["method"].append('Data splitting')

                    # Naive coverage
                    #oper_char["sparsity size"].append(s_group)
                    oper_char["complexity"].append("(" + str(nknots) + "," +
                                                   str(degree) + ")")
                    oper_char["SNR"].append(SNR)
                    oper_char["coverage rate"].append(np.mean(coverage_naive))
                    oper_char["avg length"].append(np.mean(lengths_naive))
                    oper_char["F1 score"].append(F1_naive)
                    oper_char["method"].append('Naive')

                    break  # Go to next iteration if we have some selection

    oper_char_df = pd.DataFrame.from_dict(oper_char)
    oper_char_df.to_csv('bspline_vary_SNR_nonlinear('+ str(nknots) + ','+ str(degree) + ')_' + str(range.start) + '_' + str(range.stop) + '.csv',
                        index=False)

if __name__ == '__main__':
    argv = sys.argv
    ## sys.argv: [something, start, end, p_l, s_l, order, knots]
    start, end = 0, 30#int(argv[1]), int(argv[2])
    print("start:", start, ", end:", end)
    comparison_b_spline_vary_SNR(range=range(start, end))

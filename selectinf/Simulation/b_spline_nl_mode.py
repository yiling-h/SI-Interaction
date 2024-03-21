import sys
sys.path.append('/home/yilingh/SI-Interaction')
import time
import itertools

import pandas as pd
from simulation_helpers import (one_sim_mode, one_sim_mode_serial,
                                generate_gaussian_instance_nonlinear_interaction)
from multiprocessing import Pool

from functools import partial


def interaction_filter_vary_mode(start, end, use_MLE=True, parallel=True,
                                 ncores=8):
    """
    Compare to R randomized lasso
    """

    # Operating characteristics
    oper_char = {}
    oper_char["coverage rate"] = []
    oper_char["avg length"] = []
    oper_char["method"] = []
    oper_char["F1 score interaction"] = []
    oper_char["|G|"] = []
    oper_char["mode"] = []
    oper_char["SNR"] = []
    p = 30
    SNR = 0.5
    intercept_flag = True

    """
    GOOD RESULT:
    1. 
    p = 30
    SNR = 1
    intercept_flag = True
    ### Partially correlated
    (design, data_interaction, Y, Y_mean, data_combined,
                 groups, active, active_inter_adj, active_inter_list) \
                    = inst(n=500, p_nl=p, p_l=0, s_l=0,
                           nknots=6, degree=2, SNR=SNR, rho=0.5, rho_noise=0.5,
                           center=False, scale=False, random_signs=True,
                           intercept=intercept_flag, structure='weakhierarchy', 
                           s_interaction=10, interaction_signal=2)
    weight_frac = 2

    2. 
    p = 30
    SNR = 0.5
    intercept_flag = True
    ### Fully correlated
    (design, data_interaction, Y, Y_mean, data_combined,
                 groups, active, active_inter_adj, active_inter_list) \
                    = inst(n=500, p_nl=p, p_l=0, s_l=0,
                           nknots=6, degree=2, SNR=SNR, rho=0.5, rho_noise=0.5,
                           center=False, scale=False, random_signs=True,
                           intercept=intercept_flag, structure='weakhierarchy', 
                           s_interaction=10, interaction_signal=2)
    weight_frac = 2
    """
    if parallel:
        oper_char_list = []

    for mode in ["stronghierarchy", "weakhierarchy", "allpairs"]:
        if parallel:
            with Pool(ncores) as pool:
                results = pool.map(partial(one_sim_mode, SNR=SNR,
                                           intercept_flag=intercept_flag,
                                           p=p, use_MLE=use_MLE, mode=mode,
                                           weight_frac=1.5,
                                           inst=generate_gaussian_instance_nonlinear_interaction,
                                           rho=0.5, rho_noise=0.5, full_corr=False),
                                   list(range(start, end)))
            oper_char_list = oper_char_list + results
        else:
            for i in range(start, end):
                print(i, "th simulation for mode:", mode)

                one_sim_mode_serial(SNR=SNR, intercept_flag=intercept_flag,
                                    p=p, oper_char=oper_char, use_MLE=use_MLE,
                                    mode=mode)
    if parallel:
        oper_char_list = list(itertools.chain(*oper_char_list))
        oper_char = pd.DataFrame(oper_char_list)
        oper_char.columns = ["coverage rate", "avg length", "F1 score interaction",
                             "method", "|G|", "mode", "SNR"]
        return oper_char
    else:
        return pd.DataFrame(oper_char)

if __name__ == '__main__':
    argv = sys.argv
    # argv = [..., start, end, logic_tf, ncores]
    start, end = int(argv[1]), int(argv[2])
    ncores = int(argv[3])

    oper_char = interaction_filter_vary_mode(start=start, end=end, ncores=ncores,
                                             use_MLE=True, parallel=True)
    oper_char.to_csv('bspline_nl_mode_' + str(start) + '_'
                     + str(end) + '.csv', index=False)
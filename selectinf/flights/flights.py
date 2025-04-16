import pandas as pd
import numpy as np
import sys, os
sys.path.append('/home/yilingh/SI-Interaction/selectinf/flights/')
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from selectinf.Simulation.spline import cubic_spline, b_spline
from selectinf.Simulation.H1.nonlinear_H1_helpers import *
from selectinf.RealDataHelpers.rdhelpers import *
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import joblib

def get_splines(x_nl, x_l, nknots, degree, intercept):
    bs = b_spline(data_nl=np.array(x_nl), data_l=np.array(x_l),
                  nknots=nknots, degree=degree, intercept=intercept)
    bs.construct_splines(use_quantiles=True, equally_spaced=False, center=False)
    design_train = bs.get_spline_data()
    design_train *= np.sqrt(design_train.shape[0])
    design_train[:, 0] = 1
    # Returning group labels with 0 meaning the intercept (if applicable)
    groups = bs.get_groups()

    return design_train, groups

def validate(x_test, design_test, y_test, nonzero, selected_groups,
             groups = None, n_features=None, intercept=True, mode="allpairs", level=0.9):
    X_E = design_test[:, nonzero]
    active_flag = np.zeros(np.unique(groups).shape[0])
    active_flag[selected_groups] = 1.
    raw_data=np.array(x_test)

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

    result_dict = interaction_t_tests_all(X_E, y_test, n_features,
                                          active_vars_flag, data_interaction,
                                          level=level, mode=mode)

    return result_dict

def predict(beta_hat, X_test):
    return X_test.dot(beta_hat)
def subsampling_inference(Y, X, test_size, n_rep, alpha=0.1):
    f1_dict = {"method": [], "F1": [], "Precision": [], "Recall": []}
    MSE_dict = {"method": [], "MSE_test": [], "MSE_train": []}
    main_effects_freq_Naive = {name: 0 for name in X.columns}
    interaction_freq_Naive = {}
    main_effects_freq_MLE = {name: 0 for name in X.columns}
    interaction_freq_MLE = {}

    for i in range(n_rep):
        x_train, x_test, y_train, y_test = train_test_split(X, Y,
                                                            test_size=test_size,
                                                            random_state=i)

        linear = list(x_train.columns[x_train.nunique() < 40])
        x_train_nl = x_train.drop(linear, axis=1)
        x_train_l_temp = x_train[linear]
        x_test_nl = x_test.drop(linear, axis=1)
        x_test_l = x_test[linear]
        n_train = x_train_l_temp.shape[0]
        n_test = x_test_l.shape[0]

        x_train_l = (x_train_l_temp - x_train_l_temp.mean()) / (x_train_l_temp.std() * np.sqrt(n_train))
        x_test_l = (x_test_l - x_train_l_temp.mean()) / (x_train_l_temp.std() * np.sqrt(n_test))
        design_train, groups_train = (
            get_splines(x_train_nl, x_train_l, nknots=6, degree=2, intercept=True))
        design_test, groups_test = (
            get_splines(x_test_nl, x_test_l, nknots=6, degree=2, intercept=True))

        names_map = {i: None for i in range(X.shape[1])}
        for i in range(len(x_train_nl.columns)):
            names_map[i] = x_train_nl.columns[i]
        for j in range(len(x_train_nl.columns),
                       len(x_train_nl.columns) + len(x_train_l.columns)):
            names_map[j] = x_train_l.columns[j - len(x_train_nl.columns)]
        print(names_map)

        const = group_lasso.gaussian

        result_naive, nonzero_naive, selected_groups_naive, soln_naive \
            = naive_inference_real_data(X=design_train, Y=np.array(y_train),
                                        raw_data=np.array(x_train),
                                        groups=groups_train, const=const,
                                        n_features=x_train.shape[1],
                                        intercept=True, weight_frac=0.5, level=0.9,
                                        mode="weakhierarchy", root_n_scaled=False)

        result_MLE, nonzero_MLE, selected_groups_MLE, soln_MLE \
            = MLE_inference_real_data(X=design_train, Y=np.array(y_train),
                                      raw_data=np.array(x_train), groups=groups_train,
                                      n_features=x_train.shape[1],
                                      intercept=True, weight_frac=0.5, level=0.9,
                                      mode="weakhierarchy",
                                      root_n_scaled=False, proportion=0.9)

        if result_naive is not None:
            for g in selected_groups_naive:
                if g != 0:
                    main_effects_freq_Naive[names_map[g - 1]] += 1

            result_naive_validate = validate(x_test, design_test, y_test, nonzero_naive,
                                             selected_groups_naive, groups=groups_train,
                                             n_features=x_train.shape[1],
                                             intercept=True, mode="weakhierarchy", level=0.9)
            naive_df = pd.DataFrame(result_naive)
            naive_test_df = pd.DataFrame(result_naive_validate)
            naive_train = [(naive_df["pval"][k] < alpha) for k in range(naive_df.shape[0])]
            naive_test = [(naive_test_df["pval"][k] < alpha) for k in range(naive_test_df.shape[0])]
            f1_dict["method"].append("Naive")
            f1_dict["F1"].append(f1_score(naive_test, naive_train))
            f1_dict["Precision"].append(precision_score(naive_test, naive_train))
            f1_dict["Recall"].append(recall_score(naive_test, naive_train))

            for index, row in naive_df.iterrows():
                if row["pval"] < alpha:
                    i = row["i"]
                    j = row["j"]
                    if (names_map[i], names_map[j]) in interaction_freq_Naive.keys():
                        interaction_freq_Naive[(names_map[i], names_map[j])] += 1
                    elif (names_map[j], names_map[i]) in interaction_freq_Naive.keys():
                        interaction_freq_Naive[(names_map[j], names_map[i])] += 1
                    else:
                        interaction_freq_Naive[(names_map[i], names_map[j])] = 1

        else:
            f1_dict["method"].append("Naive")
            f1_dict["F1"].append(0)
            f1_dict["Precision"].append(0)
            f1_dict["Recall"].append(0)

        if result_MLE is not None:
            for g in selected_groups_MLE:
                if g != 0:
                    main_effects_freq_MLE[names_map[g - 1]] += 1
            result_MLE_validate = validate(x_test, design_test, y_test, nonzero_MLE,
                                           selected_groups_MLE, groups=groups_train,
                                           n_features=x_train.shape[1],
                                           intercept=True, mode="weakhierarchy", level=0.9)

            MLE_df = pd.DataFrame(result_MLE)
            MLE_test_df = pd.DataFrame(result_MLE_validate)
            MLE_train = [(MLE_df["pval"][k] < alpha) for k in range(MLE_df.shape[0])]
            MLE_test = [(MLE_test_df["pval"][k] < alpha) for k in range(MLE_test_df.shape[0])]
            f1_dict["method"].append("MLE")
            f1_dict["F1"].append(f1_score(MLE_test, MLE_train))
            f1_dict["Precision"].append(precision_score(MLE_test, MLE_train))
            f1_dict["Recall"].append(recall_score(MLE_test, MLE_train))

            for index, row in MLE_df.iterrows():
                if row["pval"] < alpha:
                    i = row["i"]
                    j = row["j"]
                    if (names_map[i], names_map[j]) in interaction_freq_MLE.keys():
                        interaction_freq_MLE[(names_map[i], names_map[j])] += 1
                    elif (names_map[j], names_map[i]) in interaction_freq_MLE.keys():
                        interaction_freq_MLE[(names_map[j], names_map[i])] += 1
                    else:
                        interaction_freq_MLE[(names_map[i], names_map[j])] = 1
        else:
            f1_dict["method"].append("MLE")
            f1_dict["F1"].append(0)
            f1_dict["Precision"].append(0)
            f1_dict["Recall"].append(0)

        MSE_naive = np.mean((y_test - predict(soln_naive, design_test)) ** 2)
        MSE_naive_train = np.mean((y_train - predict(soln_naive, design_train)) ** 2)
        MSE_MLE = np.mean((y_test - predict(soln_MLE, design_test)) ** 2)
        MSE_MLE_train = np.mean((y_train - predict(soln_MLE, design_train)) ** 2)
        MSE_dict["method"].append("Naive")
        MSE_dict["MSE_test"].append(MSE_naive)
        MSE_dict["MSE_train"].append(MSE_naive_train)
        MSE_dict["method"].append("MLE")
        MSE_dict["MSE_test"].append(MSE_MLE)
        MSE_dict["MSE_train"].append(MSE_MLE_train)

    return (f1_dict, MSE_dict, interaction_freq_Naive, interaction_freq_MLE,
            main_effects_freq_Naive, main_effects_freq_MLE)

if __name__ == '__main__':
    # Get the script's directory
    script_directory = os.path.dirname(os.path.abspath(__file__))
    # Change the working directory to the script's directory
    os.chdir(script_directory)
    current_directory = os.getcwd()
    print("Current Working Directory:", current_directory)

    fpw = pd.read_csv("fpw.csv", index_col=0)
    # %%
    Y = fpw["dep_delay"]
    X = fpw.drop(["dep_delay", 'day', "wind_gust"], axis=1)

    results = subsampling_inference(Y, X, test_size=0.9, n_rep=500, alpha=0.1)
    joblib.dump(results, "flights_results.pkl")


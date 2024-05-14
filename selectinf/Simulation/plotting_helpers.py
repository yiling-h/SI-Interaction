# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def plot_ecdf(data):
    # Step 2: Sort the data
    data_sorted = np.sort(data)

    # Step 3: Calculate the empirical CDF
    # For each point, the CDF value is the proportion of data points less than or equal to that point
    cdf_values = np.arange(1, len(data_sorted) + 1) / len(data_sorted)

    # Step 4: Plot the empirical CDF
    plt.figure(figsize=(6, 6))
    plt.step(data_sorted, cdf_values, where='post', label='Empirical CDF')
    plt.title('Empirical Cumulative Distribution Function')
    # Add a y=x line
    plt.plot(data_sorted, data_sorted, label='Uniform CDF', linestyle='--')
    plt.xlabel('Data Points')
    plt.ylabel('CDF')
    plt.grid(True)
    plt.legend()
    plt.show()


# %%
def plot_ecdfs(data_dict, xaxis=None):
    """
    Plot an ECDF for each rho value with different methods overlayed.

    Parameters:
    - data_dict: dict, a nested dictionary with structure {rho: {method: [values]}}
    """

    my_palette = {"MLE": "#48c072",
                  "Naive": "#fc5a50",
                  "Data Splitting": "#03719c"}

    def ecdf(data):
        """Compute ECDF for a one-dimensional array of measurements."""
        # Number of data points
        n = len(data)
        # x-data for the ECDF
        x = np.sort(data)
        # y-data for the ECDF
        y = np.arange(1, n + 1) / n
        return x, y

    # Number of rho values
    num_rho = len(data_dict)

    # Create a figure
    plt.figure(figsize=(4 * num_rho, 4))

    # Loop through each rho
    for i, (signal, methods) in enumerate(data_dict.items(), 1):
        # Create a subplot for each rho
        ax = plt.subplot(1, num_rho, i)
        # Loop through each method within this rho
        for method, values in methods.items():
            # Calculate ECDF
            x, y = ecdf(values)
            # Plot ECDF
            ax.plot(x, y, label=f'Method: {method}', marker='.', linestyle='none',
                    color=my_palette[method])

        # Plot y=x line
        max_value = max(max(values) for values in methods.values())  # Get maximum value across all methods
        ax.plot([0, max_value], [0, 1], 'k--', label='y=x')

        # Setting plot titles and labels
        ax.set_title('ECDF for ' + xaxis + f' = {signal}')
        ax.set_xlabel('Value')
        ax.set_ylabel('ECDF')
        # Add legend
        ax.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    # Show plot
    plt.show()


# %%
def point_plot_multimetrics(oper_char_df, x_axis='p', hue='method', plot_size=False,
                            metric_list=None):
    oper_char_df = oper_char_df.copy()
    sns.set_style("white", {'axes.facecolor': 'white',
                            'axes.grid': True,
                            'axes.linewidth': 2.0,
                            'grid.linestyle': u'--',
                            'grid.linewidth': 4.0,
                            'xtick.major.size': 5.0,
                            })
    # sns.histplot(oper_char_df["sparsity size"])
    # plt.show()
    n_subplots = len(metric_list)
    # cols = int(np.ceil(n_subplots / 2))
    cols = n_subplots

    fig = plt.figure(figsize=(cols * 5, 6))

    my_palette = {"MLE": "#48c072",
                  "Naive": "#fc5a50",
                  "Data Splitting": "#03719c"}

    # Create each subplot
    for i in range(1, n_subplots + 1):
        # ax = fig.add_subplot(2, cols, i) #two rows
        ax = fig.add_subplot(1, cols, i)  # one row
        if hue is not None:
            sns.pointplot(x=oper_char_df[x_axis],
                          y=oper_char_df[metric_list[i - 1]],
                          hue=oper_char_df[hue],
                          markers='o',
                          palette=my_palette,
                          ax=ax)
        else:
            sns.pointplot(x=oper_char_df[x_axis],
                          y=oper_char_df[metric_list[i - 1]],
                          markers='o',
                          palette=my_palette,
                          ax=ax)
        if metric_list[i - 1] == 'coverage rate':
            ax.set_ylim([0, 1])
            ax.axhline(y=0.9, color='k', linestyle='--', linewidth=1)
        if metric_list[i - 1] == 'coverage rate':
            ax.set_ylabel("Coverage Rate", fontsize=15)  # remove y label, but keep ticks
        elif metric_list[i - 1] == 'avg length':
            # ax.set_ylim([0, 2.5])
            ax.set_ylabel("Average Length", fontsize=15)  # remove y label, but keep ticks
        else:
            ax.set_ylabel(metric_list[i - 1], fontsize=15)  # remove y label, but keep ticks
        ax.legend().set_visible(False)
        # ax.set_title(f'Category: {metric_list[i-1]}')

        # ax.set_xlabel('Signal Strength', fontsize=15)
        if x_axis == 'signal':
            ax.set_xlabel('Signal Strength', fontsize=15)
        elif x_axis == 'm':
            ax.set_xlabel('Sparsity', fontsize=15)

    handles, labels = ax.get_legend_handles_labels()
    # fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.2)

    fig.subplots_adjust(bottom=0.3)
    fig.legend(handles, labels, loc='lower center', ncol=n_subplots,
               prop={'size': 15})

    # cov_plot.legend_.remove()
    # len_plot.legend_.remove()

    # plt.suptitle("Changing n,p")
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    plt.show()


# %%
def plot_pvals_targets(gammas_list, targets_list, xaxis):
    plt.figure(figsize=(2.5 * len(np.unique(gammas_list)), 6))
    sns.boxplot(x=gammas_list, y=targets_list)
    # Get current x-axis tick labels
    locs, labels = plt.xticks()
    # Set the labels with 2 decimal places
    formatted_labels = [f'{float(label.get_text()):.3f}' for label in labels]
    plt.xticks(locs, formatted_labels)
    plt.title("Projected targets vs " + xaxis)

def plot_multi_targets(target_dict, xaxis):
    parameters = target_dict['parameter']
    plt.figure(figsize=(2.5 * len(np.unique(parameters)), 6))
    sns.boxplot(y=target_dict['target'],
                x=target_dict['target id'],
                hue=target_dict['parameter'],
                palette="Blues", orient="v",
                linewidth=1)
    # Get current x-axis tick labels
    # locs, labels = plt.xticks()
    # Set the labels with 2 decimal places
    #formatted_labels = [f'{float(label.get_text()):.3f}' for label in labels]
    # plt.xticks(locs, formatted_labels)
    plt.legend(title=xaxis, loc='lower center', ncol=4)
    plt.title("Projected targets vs " + xaxis)
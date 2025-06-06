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


def plot_ecdfs(data_dict, xaxis=None, title=None):
    """
    Plot an ECDF for each rho value with different methods overlayed, using a single shared legend.

    Parameters:
    - data_dict: dict, a nested dictionary with structure {rho: {method: [values]}}
    """
    # Define colors for each method
    my_palette = {"MLE": "#48c072",
                  "Naive": "#fc5a50",
                  "Data Splitting": "#03719c"}

    def ecdf(data):
        """Compute ECDF for a one-dimensional array of measurements."""
        n = len(data)
        x = np.sort(data)
        y = np.arange(1, n + 1) / n
        return x, y

    num_rho = len(data_dict)
    # Create subplots
    fig, axes = plt.subplots(1, num_rho, figsize=(4 * num_rho, 4), facecolor="w")
    # In case there's only one subplot, wrap it in a list for consistency.
    if num_rho == 1:
        axes = [axes]

    # Plot ECDFs in each subplot
    for ax, (signal, methods) in zip(axes, data_dict.items()):
        for method, values in methods.items():
            # Map "MLE" to "Proposed"
            label = "Proposed" if method == "MLE" else method
            x, y = ecdf(values)
            ax.plot(x, y, marker='.', linestyle='none',
                    color=my_palette[method], label=label)

        # Plot reference line y = x
        max_value = max(max(vals) for vals in methods.values())
        ax.plot([0, max_value], [0, 1], 'k--', label='y=x')

        # Set subplot title and axes labels
        ax.set_title(f'{xaxis} = {signal}', fontsize=16)
        ax.set_xlabel('x', fontsize=16)
        ax.set_ylabel(r'$\widehat{F}(x)$', fontsize=16)
        # Make the current subplot square: one unit on x is equal to one unit on y.
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid(True)

    # Extract legend handles and labels from one axis (e.g., the first one)
    handles, labels = axes[0].get_legend_handles_labels()

    # Remove duplicates while preserving order.
    seen = set()
    unique_handles_labels = []
    for handle, label in zip(handles, labels):
        if label not in seen:
            unique_handles_labels.append((handle, label))
            seen.add(label)
    unique_handles, unique_labels = zip(*unique_handles_labels)

    # Create a single, global legend
    fig.legend(unique_handles, unique_labels, loc='upper center', ncol=len(unique_labels),
               fontsize=16, title_fontsize=16,
               bbox_to_anchor=(0.5, 0.0))

    if title is not None:
        fig.suptitle(title, fontsize=16)

    # Adjust layout so that the legend and title don't overlap with subplots
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.style.use('default')
    #
    plt.show()


def F1_and_len(oper_char_full_df, F1_dict_full_df, xlabel, xaxis):
    # Suppose you have two dataframes (or two subsets of your data)
    # Here, I'm using oper_char_full_df for the first plot and oper_char_subset_df for the second.
    # If you use the same dataframe for both, adjust accordingly.

    # Define the palette
    my_palette = {"Proposed": "#48c072",
                  "Naive": "#fc5a50",
                  "Data Splitting": "#03719c"}

    # Create a figure with two subplots (side by side)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # First pointplot on the left subplot:
    sns.pointplot(x=oper_char_full_df[xaxis],
                  y=oper_char_full_df["avg length"],
                  hue=oper_char_full_df["method"],
                  markers='o',
                  palette=my_palette,
                  ax=axes[0],
                  legend=True)  # Disable automatic legend

    # Second pointplot on the right subplot:
    sns.pointplot(x=F1_dict_full_df[xaxis],
                  y=F1_dict_full_df["F1"],
              hue=F1_dict_full_df["method"], markers='o',
              palette=my_palette,
                  ax=axes[1],
                  legend=False)

    axes[0].set_ylabel("Average Length")
    axes[0].set_xlabel(f"{xlabel}")
    axes[1].set_ylabel("F1 score")
    axes[1].set_xlabel(f"{xlabel}")

    # Now, create a single global legend.
    # We'll extract the handles and labels from one of the axes (since both share the same palette & labels).
    handles, labels = axes[0].get_legend_handles_labels()
    # Remove the legend from the first subplot (so it doesn't appear there)
    axes[0].legend_.remove()

    # Place the legend above the subplots, centered horizontally.
    fig.legend(handles, labels, loc='upper center', ncol=len(labels),
               title="Method", bbox_to_anchor=(0.5, 0), fontsize=12, title_fontsize=12)

    # Adjust layout so that subplots and the global legend don’t overlap.
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()

# %%
def point_plot_multimetrics(oper_char_df, x_axis='p', hue='method', plot_size=False,
                            metric_list=None, ylim_low=None, ylim_high=None):
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

def plot_multi_targets(target_dict, xaxis, ylim_low=None, ylim_high=None):
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
    if ylim_low is not None and ylim_high is not None:
        plt.ylim(ylim_low, ylim_high)
    plt.legend(title=xaxis, loc='lower center', ncol=4)
    plt.title("Projected targets vs " + xaxis)
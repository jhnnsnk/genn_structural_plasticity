import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from matplotlib.patches import Rectangle
from pandas import NamedAgg

from data_utils import load_training_data_frame
from itertools import product
from math import prod

import plot_settings

def plot_cross_validation_figure(df, panel_group_params, title, num_folds):
    # Find unique values and lookup indices for each panel group parameter
    # **NOTE** using numpy rather than pandas sorts
    unique_params = [np.unique(df[p]) for p in panel_group_params]
    num_unique_params = [len(u) for u in unique_params]

    # Determine number of rows and columns
    assert len(unique_params) == 3
    num_col = num_unique_params[-1]
    num_row = prod(num_unique_params[:-1])

    # Group by panel group parameters
    panel_df = df.groupby(panel_group_params, as_index=False, dropna=False)

    fig, axes = plt.subplots(num_row, num_col, sharex="col", sharey="row")
    fig.suptitle(title)

    # Loop through groups i.e. different sparsities
    for name, group_df in panel_df:
        # Group by batch
        config_df = group_df.groupby("Epoch", as_index=False, dropna=False)

        # Aggregate to get mean and std accuracy across repeats
        config_df = config_df.agg(mean_accuracy=NamedAgg(column="Accuracy", aggfunc="mean"),
                                  std_accuracy=NamedAgg(column="Accuracy", aggfunc="std"))

        # Convert name tuple into indices into each unique parameter
        name_inds = [np.searchsorted(u, n) for n, u in zip(name, unique_params)]

        # Convert all but last of these into flat axis row index
        axis_row = np.ravel_multi_index(name_inds[:-1], num_unique_params[:-1], order="F")
        axis = axes[axis_row, name_inds[2]]
        
        actor = axis.plot(config_df["Epoch"], config_df["mean_accuracy"])[0]
        axis.fill_between(config_df["Epoch"], config_df["mean_accuracy"] - config_df["std_accuracy"], 
                          config_df["mean_accuracy"] + config_df["std_accuracy"],
                          alpha=0.1, color=actor.get_color())

        axis.set_ylim((0.0, 100.0))

        sns.despine(ax=axis)

    # Label columns with first unique parameter at top and x label at the bottom
    for i, u in enumerate(unique_params[-1]):
        axes[0,i].set_title(f"{panel_group_params[-1]} {u}")
        axes[-1,i].set_xlabel("Batch")
    
    # Loop through rows
    for i in range(num_row):
        # Convert row index back to name index
        name_ind = np.unravel_index(i, num_unique_params[:-1], order="F")
        
        name = [f"{p} = {u[n]}" for n, u, p in zip(name_ind, unique_params, panel_group_params)]
        axes[i, 0].set_ylabel("\n".join(name), rotation="horizontal",
                              horizontalalignment="right")

    fig.tight_layout(pad=0)
    return fig
    
def plot_cross_validation(panel_group_params, fig_group_params, dataset, num_folds):
    keys = ["record_rewiring", "fold", "input_ei_split", "hidden_ei_split", "l1_strength"] + panel_group_params + fig_group_params
    df = load_training_data_frame(keys, lambda params: (params["dataset"] == dataset 
                                                        and "record_rewiring" in params 
                                                        and params["record_rewiring"] == True),
                                  path=os.path.join("results", "cross_validation_uniform_pad_5_ie"))

    # If we should group into seperate figures
    if len(fig_group_params) > 0:
        fig_df = df.groupby(fig_group_params, as_index=False, dropna=False)
        for name, fig_df in fig_df:
            plot_cross_validation_figure(fig_df, panel_group_params, 
                                         f"Dataset: {dataset}, Name: {name}",
                                         num_folds)
    # Otherwise, create figure from whole dataframe
    else:
        plot_cross_validation_figure(df, panel_group_params,
                                     f"Dataset {dataset}",
                                     num_folds)

#plot_cross_validation(["hidden_input_sparsity", "hidden_recurrent_sparsity"], ["hidden_size"],
#                      "dvs_gesture", 8)
plot_cross_validation(["input_ei_split", "hidden_ei_split", "l1_strength"], ["hidden_input_sparsity", "hidden_recurrent_sparsity", "hidden_size"],
                      "shd", 10)


plt.show()

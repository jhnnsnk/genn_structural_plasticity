import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from matplotlib.patches import Rectangle
from pandas import NamedAgg

from data_utils import load_rewiring_data_frame
from itertools import product
from math import prod

import plot_settings

def plot_cross_validation_figure(df, panel_group_params, title, num_folds):
    assert len(panel_group_params) == 1
    # Find unique values and lookup indices for each panel group parameter
    # **NOTE** using numpy rather than pandas sorts
    unique_params = np.unique(df[panel_group_params[0]])

    # Group by panel group parameters
    panel_df = df.groupby(panel_group_params, as_index=False, dropna=False)

    fig, axes = plt.subplots(1, len(unique_params), sharey=True)
    fig.suptitle(title)

    # Loop through groups i.e. different sparsities
    last_col = 0
    pal = sns.color_palette()
    conn_map = {}
    for name, group_df in panel_df:
        # Group by batch
        config_df = group_df.groupby("batch", as_index=False, dropna=False)
        
        # Find columns with names of connections
        conns = [c for c in group_df.columns
                 if c.startswith("Conn_") and not c.endswith("_fail")]

        # Aggregate to get mean and std rewiring counts and find maximum failures
        mean_aggs = {f"mean_{c}": NamedAgg(column=c, aggfunc="mean") for c in conns}
        std_aggs = {f"std_{c}": NamedAgg(column=c, aggfunc="std") for c in conns}
        max_fail_agg = {f"max_{c}_fail": NamedAgg(column=f"{c}_fail", aggfunc="max") for c in conns}
        config_df = config_df.agg(**mean_aggs, **std_aggs, **max_fail_agg)

        # Convert name tuple into indices into each unique parameter
        name_inds = np.searchsorted(unique_params, name)

        # Convert all but last of these into flat axis row index
        axis = axes[name_inds]

        # Loop through c
        for c in conns:
            # Give warning if rewiring failures occured
            if (config_df[f"max_{c}_fail"] > 0).any():
                print("WARNING: rewiring failures")

            # Allocate colour for connection
            if c not in conn_map:
                conn_map[c] = pal[last_col]
                last_col += 1
        
            axis.plot(config_df["batch"], config_df[f"mean_{c}"], color=conn_map[c])
            axis.fill_between(config_df["batch"], config_df[f"mean_{c}"] - config_df[f"std_{c}"], 
                              config_df[f"mean_{c}"] + config_df[f"std_{c}"],
                              alpha=0.1, color=conn_map[c])
        
        
        sns.despine(ax=axis)

    # Label columns with first unique parameter at top and x label at the bottom
    for i, u in enumerate(unique_params):
        axes[i].set_title(f"{panel_group_params[-1]} {u}")
        axes[i].set_xlabel("Batch")

    fig.legend([Rectangle(color=c, width=10, height=10, xy=(0,0)) for c in conn_map.values()],
               conn_map.keys(), loc="lower center", ncol=len(conn_map))
    fig.tight_layout(pad=0)
    return fig
    
def plot_cross_validation(panel_group_params, fig_group_params, dataset, num_folds):
    keys = ["record_rewiring", "fold", "l1_strength"] + panel_group_params + fig_group_params
    df = load_rewiring_data_frame(keys, lambda params: (params["dataset"] == dataset 
                                                        and "rewire" in params 
                                                        and params["rewire"] == True),
                                  path=os.path.join("results", "cross_validation_no_dalian"))

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

plot_cross_validation(["l1_strength"], ["hidden_input_sparsity", "hidden_recurrent_sparsity"],
                      "dvs_gesture", 8)
plot_cross_validation(["l1_strength"], ["hidden_input_sparsity", "hidden_recurrent_sparsity"],
                      "shd", 10)


plt.show()

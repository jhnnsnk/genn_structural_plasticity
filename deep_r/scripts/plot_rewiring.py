import os
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import NamedAgg

from collections import defaultdict
from glob import glob
from itertools import chain
from json import load
from pandas import concat, read_csv

from data_utils import get_param_val, load_rewiring_data

import plot_settings

name_keys = ["hidden_size", "hidden_recurrent", "hidden_model",
             "hidden_input_sparsity", "hidden_recurrent_sparsity"]

globs = ["params_0_False_?_512_50_shd_1_1234_0.05_512_True_alif_0.05_0.01_0.5_0.8_5.0_valid.json",
         "params_0_False_?_512_50_shd_1_1234_0.1_512_True_alif_0.05_0.05_0.5_0.8_5.0_valid.json",
         "params_0_False_?_512_50_shd_1_1234_0.01_512_True_alif_0.05_0.1_0.4_0.8_5.0_valid.json"]

# Loop through parameter files
data = defaultdict(list)
path = os.path.join("results", "cross_validation_uniform_pad_5_ie")
for name in chain.from_iterable(glob(os.path.join(path, g)) 
                                for g in globs):
    # Load parameters
    with open(name) as fp:
        params = load(fp)
    
    if ("record_rewiring" in params and params["record_rewiring"] == True
        and params["dataset"] == "shd"):
        # Get title from file
        title = os.path.splitext(os.path.basename(name))[0]

        rewire_filename = os.path.join(path, f"rewiring_{title[7:]}.csv")
        if not os.path.exists(rewire_filename):
            print(f"ERROR: missing '{rewire_filename}'")
            continue
        else:
            rewire_data = load_rewiring_data(rewire_filename, params)
            
            # Build model_name string
            title = f"{get_param_val(params['hidden_input_sparsity'])} input\n{get_param_val(params['hidden_recurrent_sparsity'])} recurrent"
            data[title].append(rewire_data)


pal = sns.color_palette()

last_col = 0
conn_map = {}

rewire_fig, rewire_axes = plt.subplots(2, len(data), sharex="col", sharey="row")


for a, (t, d) in zip(rewire_axes[0,:], data.items()):
    # Get list of rewiring column names
    connections = list(d[0].columns.values[0::2])

    # Aggregate across repeats to get mean and std
    mean_aggs = {f"mean_{c}": NamedAgg(column=c, aggfunc="mean")
                 for c in connections}
    std_aggs = {f"std_{c}": NamedAgg(column=c, aggfunc="std")
                for c in connections}
    d = concat(d).groupby(level=0).agg(**mean_aggs, **std_aggs)

    # Loop through connections
    x = range(len(d))
    for c in connections:
        # Allocate colour for connection
        if c not in conn_map:
            conn_map[c] = pal[last_col]
            last_col += 1
    
        a.plot(x, d[f"mean_{c}"], color=conn_map[c])[0]
        a.fill_between(x, d[f"mean_{c}"] - d[f"std_{c}"], d[f"mean_{c}"] + d[f"std_{c}"],
                       alpha=0.1, color=conn_map[c])

    a.set_title(t)
    a.set_xlabel("Batch")

    # Remove axis junk
    sns.despine(ax=a)
    a.xaxis.grid(False)


for a, (t, d) in zip(rewire_axes[1,:], data.items()):
    # Get list of failure column names
    connections = list(d[0].columns.values[1::2])

    # Aggregate across repeats to get maximum
    max_aggs = {f"max_{c}": NamedAgg(column=c, aggfunc=np.amax)
                for c in connections}
    d = concat(d).groupby(level=0).agg(**max_aggs)

    # Loop through connections
    x = range(len(d))
    for c in connections:
        actor = a.plot(x, d[f"max_{c}"], color=conn_map[c[:-5]])[0]

    a.set_xlabel("Batch")

    # Remove axis junk
    sns.despine(ax=a)
    a.xaxis.grid(False)

rewire_axes[0,0].set_ylabel("Proportion synapses rewired")
rewire_axes[1,0].set_ylabel("Max rewiring failures")
rewire_fig.legend([mpatches.Rectangle(color=c, width=10, height=10, xy=(0,0)) for c in conn_map.values()],
                  conn_map.keys(), loc="lower center", ncol=len(conn_map))


plt.show()

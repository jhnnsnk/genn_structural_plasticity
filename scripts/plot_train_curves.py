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

from data_utils import get_param_val

import plot_settings

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

        train_filename = os.path.join(path, f"train_output_{title[7:]}.csv")
        if not os.path.exists(train_filename):
            print(f"ERROR: missing '{train_filename}'")
            continue
        else:
            with open(train_filename) as fp:
                train_data = read_csv(fp, delimiter=",")
            
            # Calculate accuracy
            train_data["Accuracy"] = 100.0 * train_data["Number correct"] / train_data["Num trials"]
            
            # Build model_name string
            title = f"{get_param_val(params['hidden_input_sparsity'])} input\n{get_param_val(params['hidden_recurrent_sparsity'])} recurrent"
            data[title].append(train_data)

train_curve_fig, train_curve_axes = plt.subplots(1, len(data), sharey=True)

for a, (t, d) in zip(train_curve_axes, data.items()):

    # Aggregate across repeats to get mean and std
    d = concat(d).groupby(level=0).agg(mean_accuracy=NamedAgg(column="Accuracy", aggfunc="mean"),
                                       std_accuracy=NamedAgg(column="Accuracy", aggfunc="std"))

    actor = a.plot(d.index, d["mean_accuracy"])[0]
    a.fill_between(d.index, d["mean_accuracy"] - d["std_accuracy"], d["mean_accuracy"] + d["std_accuracy"],
                   alpha=0.1, color=actor.get_color())

    a.set_title(t)
    a.set_xlabel("Epoch")

    # Remove axis junk
    sns.despine(ax=a)
    a.xaxis.grid(False)


train_curve_axes[0].set_ylabel("Train accuracy [%]")


plt.show()

import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from data_utils import load_rewiring_data
from glob import glob
from json import load
from pandas import concat, read_csv

from pandas import NamedAgg

from plot_settings import column_width, double_column_width, poster

def load_training(filename_glob):
    # Loop through parameter files
    data = []
    path = os.path.join("results", "best_test_no_dalian")
    for name in glob(os.path.join(path, filename_glob)):
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
            data.append(train_data)
    return data

def plot_training(axis, filename_globs):
    pal = sns.color_palette()
    for l, g, p in filename_globs:
        # Get dense and sparse data
        data = load_training(g)
        
        # Aggregate across repeats to get mean and std
        data = concat(data).groupby(level=0).agg(
            mean_accuracy=NamedAgg(column="Accuracy", aggfunc="mean"),
            std_accuracy=NamedAgg(column="Accuracy", aggfunc="std"))
         
        # Plot sparse and dense
        actor = axis.plot(data.index, data["mean_accuracy"], label=l, color=pal[p])[0]
        axis.fill_between(data.index, data["mean_accuracy"] - data["std_accuracy"],
                          data["mean_accuracy"] + data["std_accuracy"],
                          alpha=0.1, color=actor.get_color())
       
    axis.legend(loc="lower right")

    # Label axes
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Training accuracy (%)")



def plot_rewiring(axis, filename_glob, start_pal_index):
    # Loop through parameter files
    data = []
    path = os.path.join("results", "best_test_no_dalian")
    for name in glob(os.path.join(path, filename_glob)):
        # Load parameters
        with open(name) as fp:
            params = load(fp)
        if "record_rewiring" in params and params["record_rewiring"] == True:
            # Get title from file
            title = os.path.splitext(os.path.basename(name))[0]

            rewire_filename = os.path.join(path, f"rewiring_{title[7:]}.csv")
            if not os.path.exists(rewire_filename):
                print(f"ERROR: missing '{rewire_filename}'")
                continue
            else:
                rewire_data = load_rewiring_data(rewire_filename, params)
                
                # Build model_name string
                data.append(rewire_data)
    
    # Get list of rewiring column names
    connections = list(data[0].columns.values[0::2])

    # Aggregate across repeats to get mean and std
    mean_aggs = {f"mean_{c}": NamedAgg(column=c, aggfunc="mean")
                 for c in connections}
    std_aggs = {f"std_{c}": NamedAgg(column=c, aggfunc="std")
                for c in connections}
    data = concat(data).groupby(level=0).agg(**mean_aggs, **std_aggs)

    # Loop through connections
    x = range(len(data))
    name_map = {"Conn_Pop2_Pop2": "Recurrent",
                "Conn_Pop0_Pop2": "Input"}
    pal = sns.color_palette()
    for i, c in enumerate(connections):
        conn_actor = axis.plot(x, data[f"mean_{c}"], label=name_map[c], color=pal[i + start_pal_index])[0]
        axis.fill_between(x, data[f"mean_{c}"] - data[f"std_{c}"], data[f"mean_{c}"] + data[f"std_{c}"],
                          alpha=0.1, color=conn_actor.get_color())
    
    # Label axes
    axis.set_xlabel("Batch")
    axis.set_ylabel("Proportion of synapses rewired")
    axis.legend(loc="upper right")

if __name__ == "__main__": 
    fig, axes = plt.subplots(1, 2, figsize=(double_column_width, 0.8 * column_width))

    plot_training(axes[0], [("Static sparse", "params_0_False_0_True_512_100_dvs_gesture_1_????_True_0.5_512_True_alif_0.05_0.01_5.0_full.json", 0),
                            ("DEEP R", "params_0_False_0_False_512_100_dvs_gesture_1_????_False_0.01_512_True_alif_0.05_0.01_0.0_full.json", 1),
                            ("Dense", "params_0_False_0_512_100_dvs_gesture_1_????_512_True_alif_1.0_1.0_full.json", 4)])
    plot_rewiring(axes[1], "params_0_False_0_True_512_100_dvs_gesture_1_????_True_0.5_512_True_alif_0.05_0.01_5.0_full.json", 5)
    
    axes[0].set_title("A", loc="left")
    axes[1].set_title("B", loc="left")
    # Remove axis junk
    for a in axes:
        sns.despine(ax=a)
        a.xaxis.grid(False)
    fig.tight_layout(pad=0)

    fig.savefig(f"../figures/deep_r_analysis.pdf")

    plt.show()

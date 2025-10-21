import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data_utils import load_data_frame

from pandas import NamedAgg

from plot_settings import column_width, double_column_width, poster

def plot_performance_axis(df, axis):
    # Aggregate timing detail
    column_aggregations = [
        ("deep_r_time", ["train_custom_update_deep_r_1_time",
                         "train_custom_update_deep_r_2_time",
                         "train_custom_update_deep_r_l1_time"]),
        ("softmax_time", ["train_custom_update_softmax_1_time",
                          "train_custom_update_softmax_2_time"]),
        ("gradient_update_time", ["train_custom_update_gradient_batch_reduce_time",
                                  "train_custom_update_gradient_learn_time"])]

    for trg, src in column_aggregations:
        df[trg] = sum(df[c] for c in src)
        df = df.drop(columns=src)

    assert df["num_epochs"].nunique() == 1
    num_epochs = df["num_epochs"].iloc[0]

    # Count number of entries for each tick value
    xticks = df["hidden_input_sparsity"].value_counts(sort=False)

    # **YUCK** build bar x
    bar_start = 0.0
    static_bar_x = []
    deep_r_bar_x = []
    tick_x = []
    for i, c in xticks.items():
        group_start = bar_start
        static_bar_x.append(bar_start)
        bar_start += 0.4

        if c == 2:
            deep_r_bar_x.append(bar_start)
            bar_start += 0.4

        tick_x.append(group_start + ((bar_start - 0.4 - group_start) / 2))
        bar_start += 0.2

    # Add speedup vs dense
    dense_df = df[df["hidden_input_sparsity"] == "1.0"]
    if len(dense_df) == 1:
        dense_train_time = dense_df["train_time"].iloc[0]
        df["speedup"] = dense_train_time / df["train_time"]

    # Add synapse dynamics proportion
    df["synapse_dynamics_prop"] = df["train_synapse_dynamics_time"] / df["train_time"]

    # Add Deep-R overhead
    df["deep_r_overhead"] = df["deep_r_time"] / df["train_time"] * 100
    print(df)

    static_df = df[df["rewire"] != True]
    deep_r_df = df[df["rewire"] == True]

    static_actor = axis.bar(static_bar_x, static_df["train_time"] / num_epochs, width=0.4)
    deep_r_actor = axis.bar(deep_r_bar_x, deep_r_df["train_time"] / num_epochs, width=0.4)

    sns.despine(ax=axis)
    axis.xaxis.grid(False)

    axis.set_ylabel("Epoch train time (s)", y=0.45)
    axis.set_xticks(tick_x)
    axis.set_xticklabels(["Dense" if s == "1.0" else f"Sparse {(float(s) * 100):.0f}%" for s in xticks.index])

    return [static_actor, deep_r_actor]

if __name__ == "__main__":
    # Load data with same input and recurrent sparsity and with less insane levels of row padding
    df = load_data_frame(["hidden_input_sparsity", "hidden_recurrent_sparsity", "rewire", "num_epochs", "row_padding_prop"],
                         lambda params: (params["hidden_input_sparsity"] == params["hidden_recurrent_sparsity"] 
                                         and params["row_padding_prop"] < 5.0
                                         and params["dataset"] == "dvs_gesture"),
                         path=os.path.join("results", "performance_a100"),
                         load_train=True, load_train_perf=True)

    fig, axis = plt.subplots(figsize=(column_width, 0.8 * column_width))

    actors = plot_performance_axis(df, axis)

    fig.legend(actors, ["Static", "Deep-R"],
               loc="lower center", ncol=2, frameon=False)

    fig.tight_layout(pad=0, rect=[0.0, 0.2, 1.0, 1.0])

    #fig.savefig("../figures/deep_r_performance.pdf")
    plt.show()

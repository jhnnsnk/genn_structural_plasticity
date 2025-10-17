import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from data_utils import load_data_frame

from pandas import NamedAgg

from plot_settings import column_width, double_column_width, poster

def plot_accuracy_axis(df, axis, bar_group_params, xtick_format, ref_xtick):
    # Group by deep_r ness and sparsity; and aggregate across repeats
    group_df = df.groupby(["rewire"] + bar_group_params, as_index=False, dropna=False)
    group_df = group_df.agg(mean_test_accuracy=NamedAgg(column="test_accuracy", aggfunc="mean"),
                            std_test_accuracy=NamedAgg(column="test_accuracy", aggfunc="std"),
                            mean_train_accuracy=NamedAgg(column="train_accuracy", aggfunc="mean"),
                            std_train_accuracy=NamedAgg(column="train_accuracy", aggfunc="std"))
    # Use format string to produce tick label
    group_df["xtick"] = group_df.apply(
       lambda r: eval(f'f{xtick_format!r}',
                      {n: r[n].split("_")[0] 
                       for n in bar_group_params}),
       axis=1)

    # Sort 
    group_df = group_df.sort_values(bar_group_params)
    
    # Count number of entries for each tick value
    xticks = group_df["xtick"].value_counts(sort=False)

    # **YUCK** build bar x
    bar_start = 0.0
    static_train_bar_x = []
    deep_r_train_bar_x = []
    static_test_bar_x = []
    deep_r_test_bar_x = []
    tick_x = []
    for i, c in xticks.items():
        group_start = bar_start
        static_train_bar_x.append(bar_start)
        bar_start += 0.2

        if c == 2:
            deep_r_train_bar_x.append(bar_start)
            bar_start += 0.2

        static_test_bar_x.append(bar_start)
        bar_start += 0.2

        if c == 2:
            deep_r_test_bar_x.append(bar_start)
            bar_start += 0.2

        tick_x.append(group_start + ((bar_start - 0.2 - group_start) / 2))
        bar_start += 0.2

    # Split data frame into static and DEEP R
    static_df = group_df[group_df["rewire"] != True]
    deep_r_df = group_df[group_df["rewire"] == True]

    static_train_actor = axis.bar(static_train_bar_x, static_df["mean_train_accuracy"],
                                  yerr=static_df["std_train_accuracy"], width=0.2)
    deep_r_train_actor = axis.bar(deep_r_train_bar_x, deep_r_df["mean_train_accuracy"],
                                  yerr=deep_r_df["std_train_accuracy"], width=0.2)
    static_test_actor = axis.bar(static_test_bar_x, static_df["mean_test_accuracy"],
                                 yerr=static_df["std_test_accuracy"], width=0.2)
    deep_r_test_actor = axis.bar(deep_r_test_bar_x, deep_r_df["mean_test_accuracy"],
                                 yerr=deep_r_df["std_test_accuracy"], width=0.2)
    
    if ref_xtick is not None:
        reference_accuracy = static_df[static_df["xtick"] == ref_xtick]["mean_test_accuracy"]
        if len(reference_accuracy) == 1:
            axis.axhline(reference_accuracy.iloc[0], linestyle="--", color="grey")
        elif len(reference_accuracy) == 0:
            print(f"ERROR: missing referendce '{ref_xtick}'")
        else:
            assert False
    
    sns.despine(ax=axis)
    axis.xaxis.grid(False)
    axis.set_xticks(tick_x)
    axis.set_ylabel("Accuracy (%)")
    axis.set_ylim((60, 100))
    axis.set_xticklabels(["Dense" if s == ref_xtick else f"Sparse\n{s}" for s in xticks.index])

    return [static_train_actor, deep_r_train_actor, static_test_actor, deep_r_test_actor]

if __name__ == "__main__": 
    bar_group_params = ["hidden_input_sparsity", "hidden_recurrent_sparsity"]

    keys = ["rewire"] + bar_group_params
    df = load_data_frame(keys, lambda params: params["dataset"] == "dvs_gesture",
                         path=os.path.join("results", "best_test_no_dalian"),
                         load_train=True, load_test=True)

    fig, axis = plt.subplots(1, figsize=(column_width, 0.8 * column_width))
        
    actors = plot_accuracy_axis(df, axis, bar_group_params,
                                "{hidden_recurrent_sparsity}% Rec.\n{hidden_input_sparsity}% In.", 
                                "100 Rec.\n100 In.")

    fig.legend(actors, ["Static train", "Deep-R train", "Static test", "Deep-R test"],
               loc="lower center", ncol=2, frameon=False,
               columnspacing=0.8, handletextpad=0.4)
    fig.tight_layout(pad=0, rect=[0.0, 0.2, 1.0, 1.0])

    #fig.savefig(f"../figures/deep_r_accuracy.pdf")

    plt.show()

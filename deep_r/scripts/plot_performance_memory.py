import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data_utils import load_data_frame

from pandas import NamedAgg
from scipy.stats import binom

from plot_settings import column_width, double_column_width, poster
from plot_accuracy import plot_accuracy_axis
from plot_performance import plot_performance_axis

fig, axes = plt.subplots(1, 2, figsize=(double_column_width, 0.8 * column_width))

# Plot performance
# Load data with same input and recurrent sparsity and with less insane levels of row padding
performance_df = load_data_frame(["hidden_input_sparsity", "hidden_recurrent_sparsity", "rewire", "num_epochs", "row_padding_prop"],
                                 lambda params: (params["hidden_input_sparsity"] == params["hidden_recurrent_sparsity"] 
                                                 and params["row_padding_prop"] < 5.0
                                                 and params["dataset"] == "dvs_gesture"),
                                 path=os.path.join("results", "performance_a100"),
                                 load_train=True, load_train_perf=True)

performance_actors = plot_performance_axis(performance_df, axes[0])

# Plot memory
num_input = 32 * 32 * 2
num_hidden = 512
num_output = 11
sparsity = [0.01, 0.05]

dense_num_weights = ((num_input * num_hidden)
                     + (num_hidden * num_hidden)
                     + (num_hidden * num_output)) / 1e6
in_quantile = 0.9999 ** (1.0 / num_input)
hid_quantile = 0.9999 ** (1.0 / num_hidden)

standard_num_weights = []
padded_num_weights = []
for s in sparsity:
    in_hid_max_row_length = int(binom.ppf(in_quantile, num_hidden, s))
    hid_hid_max_row_length = int(binom.ppf(hid_quantile, num_hidden, s))
    standard_num_weights.append(((num_input * in_hid_max_row_length)
                                + (num_hidden * hid_hid_max_row_length)
                                + (num_hidden * num_output)) / 1e6)
                                
    padded_num_weights.append(((num_input * in_hid_max_row_length * 2)
                               + (num_hidden * hid_hid_max_row_length * 2)
                               + (num_hidden * num_output * 2)) / 1e6)
# Plot bars
standard_actor = axes[1].bar([0.0, 1.0, 2.0], standard_num_weights + [dense_num_weights], width=0.4)
padded_actor = axes[1].bar([0.4, 1.4], padded_num_weights, width=0.4)

print([dense_num_weights / s for s in standard_num_weights])
print([dense_num_weights / s for s in padded_num_weights])
axes[1].set_xticks([0.2, 1.2, 2.0], ["Sparse 1%", "Sparse 5%", "Dense"])
axes[1].set_ylabel("Number of weights (million)")


sns.despine(ax=axes[1])
axes[1].xaxis.grid(False)

# Set titles
axes[0].set_title("A", loc="left", weight="bold")
axes[1].set_title("B", loc="left", weight="bold")
    
fig.legend(performance_actors, ["Static train", "DEEP R train", "Static test", "DEEP R test"],
           loc="lower center", ncol=4, frameon=False)   
fig.tight_layout(pad=0, w_pad=1.0, rect=[0.0, 0.125, 1.0, 1.0])

fig.savefig("../figures/deep_r_perf_memory.pdf")
plt.show()

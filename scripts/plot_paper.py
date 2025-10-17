import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data_utils import load_data_frame

from pandas import NamedAgg

from plot_settings import column_width, double_column_width, poster
from plot_accuracy import plot_accuracy_axis
from plot_performance import plot_performance_axis

fig, axes = plt.subplots(1, 2, figsize=(double_column_width, 0.8 * column_width))


# Plot accuracy
accuracy_bar_group_params = ["hidden_input_sparsity", "hidden_recurrent_sparsity"]

accuracy_keys = ["rewire"] + accuracy_bar_group_params
accuracy_df = load_data_frame(accuracy_keys, lambda params: params["dataset"] == "dvs_gesture",
                              path=os.path.join("results", "best_test_no_dalian"),
                              load_train=True, load_test=True)

accuracy_actors = plot_accuracy_axis(accuracy_df, axes[0], accuracy_bar_group_params,
                                     "{(float(hidden_recurrent_sparsity) * 100):.0f}% $W^{{rec}}$\n{(float(hidden_input_sparsity) * 100):.0f}% $W^{{in}}$", 
                                     "100% $W^{rec}$\n100% $W^{in}$")

# Plot performance
# Load data with same input and recurrent sparsity and with less insane levels of row padding
performance_df = load_data_frame(["hidden_input_sparsity", "hidden_recurrent_sparsity", "rewire", "num_epochs", "row_padding_prop"],
                                 lambda params: (params["hidden_input_sparsity"] == params["hidden_recurrent_sparsity"] 
                                                 and params["row_padding_prop"] < 5.0
                                                 and params["dataset"] == "dvs_gesture"),
                                 path=os.path.join("results", "performance_a100"),
                                 load_train=True, load_train_perf=True)

performance_actors = plot_performance_axis(performance_df, axes[1])

# Set titles
axes[0].set_title("A", loc="left")
axes[1].set_title("B", loc="left")

# Check colours match
assert accuracy_actors[0][0].get_facecolor() == performance_actors[0][0].get_facecolor()
assert accuracy_actors[1][0].get_facecolor() == performance_actors[1][0].get_facecolor()

fig.legend(accuracy_actors, ["Static train", "DEEP R train", "Static test", "DEEP R test"],
           loc="lower center", ncol=4, frameon=False)
               
fig.tight_layout(pad=0, w_pad=1.0, rect=[0.0, 0.125, 1.0, 1.0])

fig.savefig("../figures/deep_r_results.pdf")
plt.show()

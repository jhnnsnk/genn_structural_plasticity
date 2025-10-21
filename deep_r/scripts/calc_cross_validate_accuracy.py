import os
import pandas as pd

from data_utils import load_data_frame

cfg_keys = ["hidden_input_sparsity", "hidden_recurrent_sparsity", "dataset"]
df = load_data_frame(["fold", "l1_strength"] + cfg_keys, lambda params: True,
                     path=os.path.join("results", "cross_validation_no_dalian"), load_test=True)

# Group again by configuration and aggregate over validation folds to get mean and std test accuracy
config_df = df.groupby(["l1_strength"] + cfg_keys, as_index=False, dropna=False)
config_df = config_df.agg(mean_deep_r_test_accuracy=pd.NamedAgg(column="test_accuracy", aggfunc="mean"),
                          std_deep_r_test_accuracy=pd.NamedAgg(column="test_accuracy", aggfunc="std"))
#print(config_df.sort_values(["hidden_input_sparsity", "hidden_recurrent_sparsity"], ascending=False))
for cfg, df in config_df.groupby(cfg_keys):
    best_idx = df["mean_deep_r_test_accuracy"].idxmax()
    print(cfg)
    print(df.loc[best_idx])



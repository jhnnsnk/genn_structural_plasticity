import os
import pandas as pd

from scipy.stats import wilcoxon
from data_utils import load_data_frame

def calc_significance(group_params, filter_fn, ref_cond):
    # Load best test data
    keys = ["rewire"] + group_params
    df = load_data_frame(keys, filter_fn,
                         path=os.path.join("results", "best_test_no_dalian"),
                         load_test=True)
    
    # If no reference conditions are provided, no rows are reference
    if ref_cond is None:
        ref_mask = pd.Series(False, index=df.index)
    # Otherwise
    else:
        # Start with mask with True for each row
        ref_mask = pd.Series(True, index=df.index)

        # AND together with reference conditions
        for col, val in ref_cond.items():
            ref_mask &= (df[col] == val)

    # Split main dataframe into reference rows and non-reference rows
    ref_df = df[ref_mask]
    comp_df = df[~ref_mask]

    # Group by chosen parameters
    comp_df = comp_df.groupby(group_params, as_index=False, dropna=False)
    for name, group_df in comp_df:
        # Split into deep_r and static
        static_df = group_df[group_df["rewire"] != True]
        deep_r_df = group_df[group_df["rewire"] == True]
        
        name_friendly = ", ".join(f"{p}={n}" for p, n in zip(group_params, name))
        print(f"\t{name_friendly}")
        if len(static_df) == len(deep_r_df):
            w_p = wilcoxon(deep_r_df["test_accuracy"], static_df["test_accuracy"], alternative="greater", mode="exact").pvalue
            print(f"\t\tWilcoxon Deep-R vs Static: p = {w_p}, N = {len(static_df)}")
        else:
            print("\t\tError: unmatched deep-r and static data")

print("DVS gesture, 512 hidden neurons")
dvs_512_filter = lambda params: (params["dataset"] == "dvs_gesture" and params["hidden_size"] == [512])
calc_significance(["hidden_input_sparsity", "hidden_recurrent_sparsity"], dvs_512_filter,
                  {"hidden_input_sparsity": "1.0", "hidden_recurrent_sparsity": "1.0"})

"""
build_response_longitudinal.py — PADA-1 early-response features in longitudinal form (one value per timepoint), so
that SHERLOC can process them in the feature-specific panel arm (Linear(T, 2) per feature) instead of the clinical arm.
Change features take 0 at BL (reference) and the BL->C2 change at C2; detection is observed at both timepoints.
Output: data/response_longitudinal.csv with columns {BL,C2}_resp_<feature> (joined via paths.extra_features_file).
"""
import pandas as pd
d = pd.read_csv("data/PADA1_SHERLOC_dataset.csv", index_col=0)
out = pd.DataFrame(index=d.index)
change = {"resp_logratio_mean_VAF": "dyn_log_ratio_mean_VAF", "resp_logratio_max_VAF": "dyn_log_ratio_max_VAF",
          "resp_clearance": "dyn_clearance", "resp_frac_mut_cleared": "dyn_frac_mut_cleared",
          "resp_nb_new_mut": "dyn_nb_new_mut_C2", "resp_nb_driver_lost": "dyn_nb_driver_lost"}
for new, old in change.items():
    out[f"BL_{new}"] = 0.0
    out[f"C2_{new}"] = d[old]
out["BL_resp_detected"] = (d["BL_nb_mut"] > 0).astype(float)
out["C2_resp_detected"] = d["dyn_C2_detected"]
out.to_csv("data/response_longitudinal.csv")
print(out.describe().round(3).T.to_string())

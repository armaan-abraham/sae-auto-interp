# %%
import pandas as pd
import numpy as np
from utils import results_png_dir, results_svg_dir, rc_params

results_df = pd.DataFrame(columns=["k", "normalized_mse", "arch"])
results_df.loc[0] = [64, 0.192, "0-0"]
results_df.loc[1] = [128, 0.143, "0-0"]
results_df.loc[2] = [256, 0.104, "0-0"]
results_df.loc[3] = [64, 0.145, "2-2"]
results_df.loc[4] = [128, 0.117, "2-2"]
results_df.loc[5] = [256, 0.0896, "2-2"]

results_df["log_k"] = np.log2(results_df["k"])

results_df["arch"] = results_df["arch"].map({"0-0": "Shallow", "2-2": "Deep (1 non-sparse)"})


import seaborn as sns
import matplotlib.pyplot as plt
from armaan.palette import palette

plt.rcParams.update(rc_params)
plt.figure(figsize=(4, 3))
ax = sns.lineplot(
    data=results_df,
    x="log_k",
    y="normalized_mse",
    hue="arch",
    marker="o",
    palette=[palette[3], palette[4]],
)
ax.set_xticks([np.log2(64), np.log2(128), np.log2(256)])
ax.set_xticklabels(["64", "128", "256"])
ax.set_ylabel("Normalized MSE")
ax.set_xlabel("k")
ax.set_title("MSE sparsity frontier")
ax.legend(title=None)

plt.savefig(results_png_dir / "mse_sparsity_frontier.png", dpi=300)
plt.savefig(results_svg_dir / "mse_sparsity_frontier.svg")

# %%

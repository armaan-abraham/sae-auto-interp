# %%
import pandas as pd
import numpy as np
from utils import results_png_dir, results_svg_dir

results_df = pd.DataFrame(columns=["dead", "arch"])
results_df.loc[0] = [9, "0-0"]
results_df.loc[1] = [2396, "2-2"]
results_df.loc[2] = [18568, "2-4-4-2"]
dim = 24576
results_df["dead_ratio"] = results_df["dead"] / dim

import seaborn as sns
import matplotlib.pyplot as plt
from armaan.palette import palette

order = ["0-0", "2-2", "2-4-4-2"]
palette=[palette[3], palette[4], palette[2]]
plt.figure(figsize=(5, 4))
ax = sns.barplot(
    data=results_df,
    x="arch",
    y="dead_ratio", 
    palette=palette,
    order=order,
)
ax.set_yscale("log")
ax.set_yticks([1e-3, 1e-2, 1e-1, 1])
ax.set_yticklabels(["0.1%", "1%", "10%", "100%"])
ax.set_ylabel("Percent dead neurons", labelpad=-2)
ax.set_xticklabels(["Shallow", "Deep (1 dense)", "Deep (2 dense)"])
ax.set_xlabel(None)

plt.savefig(results_png_dir / "dead_neurons.png", dpi=300)
plt.savefig(results_svg_dir / "dead_neurons.svg")

# %%

# %%
import pandas as pd
import numpy as np
from utils import results_png_dir, results_svg_dir, rc_params

results_df = pd.DataFrame(columns=["dead", "arch"])
results_df.loc[0] = [1140, "0-0"]
results_df.loc[1] = [2396, "2-2"]
results_df.loc[2] = [18568, "2-4-4-2"]
dim = 24576
results_df["dead_perc"] = results_df["dead"] / dim * 100

import seaborn as sns
import matplotlib.pyplot as plt
from armaan.palette import palette

order = ["0-0", "2-2", "2-4-4-2"]
palette=[palette[3], palette[4], palette[2]]
plt.rcParams.update(rc_params)

plt.figure(figsize=(4, 3))
ax = sns.barplot(
    data=results_df,
    x="arch",
    y="dead_perc", 
    palette=palette,
    order=order,
)
ax.set_ylabel("Dead neurons (%)")
ax.set_xticklabels(["Shallow", "Deep (1 non-sparse)", "Deep (2 non-sparse)"], rotation=12)
ax.set_xlabel(None)

plt.savefig(results_png_dir / "dead_neurons.png", dpi=300)
plt.savefig(results_svg_dir / "dead_neurons.svg")

# %%

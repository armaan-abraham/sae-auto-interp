# %%
from pathlib import Path
import os
import orjson
import numpy as np
import seaborn as sns
import pandas as pd
from generate_scores import score_dir
from utils import results_dir

exp_type = "detection"


def load_scores(arch_name):
    this_score_dir = score_dir / arch_name

    txt_files = sorted(
        list(this_score_dir.glob("*.txt")),
        key=lambda x: int(x.stem.split("feature")[-1]),
    )

    all_correctness = []
    num_invalid = 0

    print(f"Arch: {arch_name}")
    for txt_file in txt_files:
        with open(txt_file, "r") as f:
            scores = orjson.loads(f.read())

        invalid = len([score for score in scores if score["prediction"] == -1])
        num_invalid += invalid
        print(f"Number invalid: {invalid}")
        scores = [score for score in scores if score["prediction"] != -1]
        if not scores:
            continue

        all_correctness.extend([score["correct"] for score in scores])

    all_correctness = np.array(all_correctness, dtype=float)

    print(f"Percentage invalid: {num_invalid / (num_invalid + len(all_correctness))}")
    print(f"Total: {num_invalid + len(all_correctness)}")

    return all_correctness


# %%

arch_names = ["2-2", "0-0"]
all_correctness = [load_scores(arch_name) for arch_name in arch_names]
df = pd.DataFrame(columns=["arch", "correctness"])
for i, correctnesss in enumerate(all_correctness):
    for correctness in correctnesss:
        df.loc[len(df)] = [arch_names[i], correctness]


# %%

df_sub = df[df["arch"] == "2-2"]
print(df_sub)
print(df_sub["correctness"].sum() / len(df_sub))
print(len(df_sub))

# %%
from armaan.palette import palette
from matplotlib import pyplot as plt
from statannotations.Annotator import Annotator

plt.figure(figsize=(6, 5))
ax = sns.barplot(
    data=df,
    x="arch",
    y="correctness",
    palette=[palette[0], palette[3]],
    order=["0-0", "2-2"],
)
ax.set_title(f"{exp_type.capitalize()} accuracy")
ax.set_xticklabels(["Shallow SAE", "Deep SAE (1 hidden layer)"])
ax.set_ylabel("Accuracy")
ax.set_xlabel(None)
ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.8, linewidth=1)

# Add value labels on top of each bar
for i in ax.containers:
    ax.bar_label(i, fmt="%.3f", padding=3)

pairs = [("2-2", "0-0")]
annotator = Annotator(
    ax,
    pairs,
    data=df,
    x="arch",
    y="correctness",
)

annotator.configure(
    test="Mann-Whitney",
    text_format="star",
    hide_non_significant=True,
)
annotator.apply_and_annotate()
plt.savefig(results_dir / f"{exp_type}_accuracy.png", dpi=300)

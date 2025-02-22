# %%
from pathlib import Path
import os
import orjson
import numpy as np
import seaborn as sns
import pandas as pd
from generate_scores import score_dir
from utils import results_dir, results_png_dir, results_svg_dir

scorer_name = "fuzz"


def load_scores(arch_name):
    this_score_dir = score_dir / scorer_name / arch_name

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
        scores = [score for score in scores if score["prediction"] != -1]
        if not scores:
            continue

        all_correctness.extend([score["correct"] for score in scores])

    all_correctness = np.array(all_correctness, dtype=float)

    print(f"Percentage invalid: {num_invalid / (num_invalid + len(all_correctness))}")
    print(f"Total: {num_invalid + len(all_correctness)}")

    return all_correctness


# %%

arch_names = ["2-2", "0-0_act_decay", "2-4-4-2"]
all_correctness = [load_scores(arch_name) for arch_name in arch_names]
df = pd.DataFrame(columns=["arch", "correctness"])
for i, correctnesss in enumerate(all_correctness):
    for correctness in correctnesss:
        df.loc[len(df)] = [arch_names[i], correctness]


# %%
from armaan.palette import palette
from matplotlib import pyplot as plt
from statannotations.Annotator import Annotator

plt.figure(figsize=(4, 3.5))
for arch_name in arch_names:
    acc = df[df["arch"] == arch_name]["correctness"].mean()
    print(f"{arch_name}: {acc:.3f}")

order = ["0-0_act_decay", "2-2", "2-4-4-2"]
ax = sns.barplot(
    data=df,
    x="arch",
    y="correctness",
    palette=[palette[3], palette[4], palette[2]],
    order=order,
)
ax.set_title(f"Automated intepretability score ({scorer_name})", fontsize=10)
ax.set_xticklabels(["Shallow", "Deep (1 dense)", "Deep (2 dense)"], fontsize=8)
ax.set_ylabel("Accuracy", fontsize=10)
ax.set_xlabel(None)
ax.tick_params(axis='both', which='major', labelsize=8)
ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.8, linewidth=1)

# Add value labels on top of each bar
for i in ax.containers:
    ax.bar_label(i, fmt="%.3f", padding=3, fontsize=8)

pairs = [("0-0_act_decay", "2-2"), ("0-0_act_decay", "2-4-4-2")]
annotator = Annotator(
    ax,
    pairs,
    data=df,
    x="arch",
    y="correctness",
    order=order,
)

annotator.configure(
    test="Mann-Whitney",
    text_format="star",
    hide_non_significant=True,
    line_offset=5,
    use_fixed_offset=True,
    
    
)
annotator.apply_and_annotate()
results_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(results_png_dir / f"{scorer_name}_accuracy.png", dpi=300)
plt.savefig(results_svg_dir / f"{scorer_name}_accuracy.svg")


# %%

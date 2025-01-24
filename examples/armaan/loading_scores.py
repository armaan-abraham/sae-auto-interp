# %%
from pathlib import Path
import os
path_here = Path(os.getcwd())
arch_name = "10"
scores_dir = path_here / f"scores_{arch_name}"

# %%
import orjson
import numpy as np

txt_files = sorted(list(scores_dir.glob("*.txt")), key=lambda x: int(x.stem.split("feature")[-1]))

sensitivities = []
specificities = []
accuracies = []

print(f"Arch: {arch_name}")
for txt_file in txt_files:
    with open(txt_file, "r") as f:
        scores = orjson.loads(f.read())
    
    
    scores = [score for score in scores if score["prediction"] != -1]
    if not scores:
        continue

    def get_sensitivity(scores):
        return sum([score["ground_truth"] and score["correct"] for score in scores]) / sum([score["ground_truth"] for score in scores])

    def get_specificity(scores):
        return sum([not score["ground_truth"] and score["correct"] for score in scores]) / sum([not score["ground_truth"] for score in scores])
    
    def get_accuracy(scores):
        return sum([score["correct"] for score in scores]) / len(scores)

    print(txt_file.stem)
    sensitivity = get_sensitivity(scores)
    specificity = get_specificity(scores)
    accuracy = get_accuracy(scores)
    sensitivities.append(sensitivity)
    specificities.append(specificity)
    accuracies.append(accuracy)
    print(f"Sensitivity: {sensitivity:.2f}")
    print(f"Specificity: {specificity:.2f}")
    print(f"Accuracy: {accuracy:.2f}")
    print()

print(f"Mean Sensitivity: {np.mean(sensitivities):.2f}")
print(f"Mean Specificity: {np.mean(specificities):.2f}")
print(f"Mean Accuracy: {np.mean(accuracies):.2f}")

# %%
scores[39]
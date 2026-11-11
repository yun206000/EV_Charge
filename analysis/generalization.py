import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

"""
Generalisation of the Tainan source model to Taipei (scripts/run_generalization.sh baselines):
zero-shot (episode seeds 1-5) vs random policy (seeds 1-3).
Writes Results/Transfer/generalization_summary.txt and Images/Result_Plots/Transfer/generalization.png.

    python -m analysis.generalization
"""

CSV = "Results/Transfer/generalization.csv"
SUMMARY = "Results/Transfer/generalization_summary.txt"
OUT_FILE = "Images/Result_Plots/Transfer/generalization.png"
ZS, REF = "#1baf7a", "#52514e"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
NAMES = {"zeroshot": "Zero-shot\n(Tainan model)", "random": "Random\npolicy"}


def main():
    df = pd.read_csv(CSV)
    groups = {label: df[df.label == label].improvement for label in NAMES if (df.label == label).any()}
    zero, rand = groups["zeroshot"], groups["random"]
    welch = stats.ttest_ind(zero, rand, equal_var=False, alternative="greater")
    lines = ["Generalisation: Tainan source model on Taipei (improvement = score - existing score 0.2333)", ""]
    for label, values in groups.items():
        lines.append("{:10s} n = {}  mean {:.4f}  sd {:.4f}  min {:.4f}  max {:.4f}".format(
            label, len(values), values.mean(), values.std() if len(values) > 1 else 0, values.min(), values.max()))
    lines += ["",
              "zero-shot vs random: {:+.1f}%, Welch t = {:.2f}, one-sided p = {:.4f}".format(
                  (zero.mean() / rand.mean() - 1) * 100, welch.statistic, welch.pvalue)]
    text = "\n".join(lines) + "\n"
    with open(SUMMARY, "w") as file:
        file.write(text)
    print(text)

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    order = sorted(groups, key=lambda l: -groups[l].mean())
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for x, label in enumerate(order):
        values = groups[label]
        color = ZS if label == "zeroshot" else REF
        ax.bar(x, values.mean(), width=0.6, color=color, alpha=1 if label == "zeroshot" else 0.55)
        if len(values) > 1:
            ax.scatter(np.full(len(values), x), values, color=INK, s=12, zorder=3)
        ax.text(x, values.max() + 0.06, "{:.2f}".format(values.mean()), ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([NAMES[label] for label in order])
    ax.set_ylabel("Score improvement over existing plan", color=MUTED, fontsize=9)
    ax.set_title("Taipei: Tainan model without any Taipei training", loc="left", fontsize=12, color=INK, pad=22)
    ax.text(0, 1.02, "bars = mean, dots = episodes (zero-shot: episode seeds 1-5, random: seeds 1-3)",
            transform=ax.transAxes, fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=200, facecolor=SURFACE)
    plt.close(fig)


if __name__ == '__main__':
    main()

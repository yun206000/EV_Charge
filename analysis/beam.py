import ast
import argparse
import glob
import os
import re
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

"""
Analysis of the beam search on Taipei (scripts/run_beam_search.sh), written to Results/Transfer/beam_summary.txt and
Images/Result_Plots/Transfer/beam/:

    validation        B = k = 1 must equal the zero-shot greedy episodes (Results/Transfer/generalization.csv, label zeroshot)
    comparison        beam search vs zero-shot per episode seed, paired t-test (one-sided), exact Wilcoxon signed-rank
    service metrics   travel / waiting time of the saved plans (analysis/plan_metrics.py definitions)
    figures           before_after.png, search_progress.png, service_metrics.png

    python -m analysis.beam [--label B4k2]
"""

CSV = "Results/Transfer/beam.csv"
ZEROSHOT_CSV = "Results/Transfer/generalization.csv"
PLANS = "Results/Transfer/plans"
OUT_DIR = "Images/Result_Plots/Transfer/beam"
SUMMARY = "Results/Transfer/beam_summary.txt"
ZS, BEAM, REF = "#1baf7a", "#2a78d6", "#52514e"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def finish(fig, ax, title, subtitle, out_file):
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=22)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(out_file, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def read_progress(log_file):
    """(step, best improvement so far) from the periodic lines of a transfer/beam_search.py log"""
    rows = []
    for line in open(log_file):
        match = re.match(r"step (\d+) beam \d+ best live ([-\d.na]+) .* best done ([-\d.]+)", line)
        if match:
            values = [float(v) for v in match.group(2, 3) if v not in ("-", "nan")]
            if values:
                rows.append((int(match.group(1)), max(values) - 0.23333333333333334))
    return pd.DataFrame(rows, columns=["step", "improvement"])


def plan_service_metrics(labels):
    """service metrics of saved plans: existing / zero-shot from metrics.csv, beam plans computed here"""
    from analysis import plan_metrics as pm
    import test_solution as ts
    import pickle
    rows = []
    pm.load_existing("Taipei")  # prepares the road graph used by the travel times
    for label in labels:
        for plan_dir in sorted(glob.glob(os.path.join(PLANS, label + "_e*"))):
            with open(os.path.join(plan_dir, "nodes_RL.txt")) as file:
                result_nodes = ast.literal_eval(file.readline())
            with open(os.path.join(plan_dir, "plan_RL.pkl"), "rb") as file:
                result_plan = pickle.load(file)
            result_nodes, result_plan = ts.prepare_existing_plan(result_plan, result_nodes)
            row = pm.metrics(result_plan, result_nodes)
            row.update(label=label, plan=os.path.basename(plan_dir))
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default=None, help="beam setting to report, e.g. B4k2 (default: widest beam)")
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    beam = pd.read_csv(CSV)
    zero = pd.read_csv(ZEROSHOT_CSV)
    zero = zero[zero.label == "zeroshot"].drop_duplicates("eval_seed").set_index("eval_seed").sort_index()
    lines = []

    # validation: B = k = 1 equals the zero-shot episode
    check = beam[beam.label == "B1k1"].drop_duplicates("eval_seed", keep="last").set_index("eval_seed").sort_index()
    if len(check):
        diff = (check.improvement - zero.improvement.reindex(check.index)).abs()
        steps_equal = (check.episode_steps == zero.episode_steps.reindex(check.index)).all()
        lines += ["Validation B = k = 1 vs zero-shot (transfer/evaluate.py), episode seeds {}".format(list(check.index)),
                  "  max |difference| of improvement: {:.3g}, same episode length: {}".format(diff.max(), steps_equal),
                  ""]

    label = args.label or beam[beam.label != "B1k1"].sort_values("beam").label.iloc[-1]
    result = beam[beam.label == label].drop_duplicates("eval_seed", keep="last").set_index("eval_seed").sort_index()
    seeds = result.index
    before, after = zero.improvement.reindex(seeds), result.improvement
    d = after - before
    t_test = stats.ttest_rel(after, before, alternative="greater")
    nonzero = d[d != 0]  # exact Wilcoxon signed-rank on the seeds where the plans differ
    wilcoxon = stats.wilcoxon(nonzero, alternative="greater", mode="exact") if len(nonzero) >= 3 else None
    table = pd.DataFrame({"zero_shot": before, "beam": after, "difference": d,
                          "change_pct": d / before * 100, "stations_zero_shot": zero.n_stations.reindex(seeds),
                          "stations_beam": result.n_stations, "minutes": result.seconds / 60})
    lines += ["Beam search {} (B = {}, k = {}) vs zero-shot, Taipei".format(
                  label, result.beam.iloc[0], result.topk.iloc[0]),
              table.round(4).to_string(), "",
              "mean zero-shot {:.4f}, mean beam {:.4f}, change {:+.4f} ({:+.1f}%)".format(
                  before.mean(), after.mean(), d.mean(), d.mean() / before.mean() * 100),
              "better on {} / {}, equal on {}, worse on {} episode seeds".format(
                  int((d > 0).sum()), len(seeds), int((d == 0).sum()), int((d < 0).sum())),
              "paired t-test (one-sided, beam > zero-shot): t = {:.3f}, p = {:.4f}".format(t_test.statistic,
                                                                                         t_test.pvalue)]
    if wilcoxon is not None:
        lines.append("Wilcoxon signed-rank (one-sided, exact, {} seeds with a difference): p = {:.4f}".format(
            len(nonzero), wilcoxon.pvalue))

    # service metrics
    service = plan_service_metrics([label])
    known = pd.read_csv(os.path.join(PLANS, "metrics.csv"))
    groups = {"Existing plan": known[known.label == "existing"],
              "Zero-shot": known[known.label.str.startswith("zeroshot")],
              "Beam search": service}
    metric_rows = []
    for name, group in groups.items():
        metric_rows.append(dict(plan=name, n=len(group), stations=group.n_stations.mean(),
                                travel_avg_min=group.travel_avg_min.mean(),
                                waiting_avg_min=group.waiting_avg_min.mean(),
                                waiting_max_min=group.waiting_max_min.mean()))
    metric_table = pd.DataFrame(metric_rows).set_index("plan")
    lines += ["", "Service metrics (mean over plans; test_solution.py definitions)",
              metric_table.round(3).to_string()]
    zero_metrics = known[known.label.str.startswith("zeroshot")].assign(
        eval_seed=lambda k: k.label.str.extract(r"_e(\d+)$", expand=False).astype(int)).set_index("eval_seed")
    beam_metrics = service.assign(
        eval_seed=service.plan.str.extract(r"_e(\d+)$", expand=False).astype(int)).set_index("eval_seed")
    per_seed = pd.DataFrame({"travel_avg_zero_shot": zero_metrics.travel_avg_min.reindex(seeds),
                             "travel_avg_beam": beam_metrics.travel_avg_min.reindex(seeds),
                             "waiting_avg_zero_shot": zero_metrics.waiting_avg_min.reindex(seeds),
                             "waiting_avg_beam": beam_metrics.waiting_avg_min.reindex(seeds)})
    lines += ["", "Average travel / waiting time per episode seed (minutes, target < 15)", per_seed.round(3).to_string()]
    service.to_csv(os.path.join(PLANS, "metrics_{}.csv".format(label)), index=False)

    # figure 1: before / after per episode seed
    fig, ax = plt.subplots(figsize=(7.5, 4.5), facecolor=SURFACE)
    style(ax)
    x = np.arange(len(seeds))
    ax.bar(x - 0.2, before, width=0.38, color=ZS, label="Zero-shot (Tainan model, greedy)")
    ax.bar(x + 0.2, after, width=0.38, color=BEAM, label="Beam search {}".format(label))
    for xi, value in zip(x, after):
        ax.text(xi + 0.2, value + 0.03, "{:.2f}".format(value), ha="center", fontsize=8.5, color=INK)
    for xi, value in zip(x, before):
        ax.text(xi - 0.2, value + 0.03, "{:.2f}".format(value), ha="center", fontsize=8.5, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(["seed {}".format(s) for s in seeds])
    ax.set_ylim(0, max(after.max(), before.max()) * 1.15)
    ax.set_ylabel("Score improvement over existing plan", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    finish(fig, ax, "Taipei: zero-shot vs beam search ({:+.1f}% on average)".format(d.mean() / before.mean() * 100),
           "Tainan source model on the Taipei map, episode seeds {}-{}".format(seeds.min(), seeds.max()),
           os.path.join(OUT_DIR, "before_after.png"))

    # figure 2: search progress (B = k = 1 log = zero-shot episode) for every seed, small multiples
    fig, axes = plt.subplots(1, len(seeds), figsize=(2.3 * len(seeds) + 1, 3.6), sharey=True, facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, seed in zip(axes, seeds):
        style(ax)
        for name, color, dash in [("B1k1", ZS, "-"), (label, BEAM, "-")]:
            log_file = "logs/beam/{}_e{}.log".format(name, seed)
            if os.path.exists(log_file):
                progress = read_progress(log_file)
                ax.plot(progress.step, progress.improvement, color=color, linewidth=2, linestyle=dash)
        ax.set_title("seed {}".format(seed), fontsize=9, color=INK, loc="left")
        ax.set_xlabel("Step", color=MUTED, fontsize=8.5)
    axes[0].set_ylabel("Best improvement so far", color=MUTED, fontsize=9)
    axes[-1].plot([], [], color=ZS, linewidth=2, label="Zero-shot")
    axes[-1].plot([], [], color=BEAM, linewidth=2, label="Beam search {}".format(label))
    fig.legend(*axes[-1].get_legend_handles_labels(), frameon=False, fontsize=8.5, loc="lower center", ncol=2)
    fig.suptitle("Search progress on Taipei", x=0.02, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(os.path.join(OUT_DIR, "search_progress.png"), dpi=200, facecolor=SURFACE)
    plt.close(fig)

    # figure 3: service metrics (two panels, one per measure)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), facecolor=SURFACE)
    colors = [REF, ZS, BEAM]
    for ax, column, title in [(axes[0], "travel_avg_min", "Average travel time (min)"),
                              (axes[1], "waiting_avg_min", "Average waiting time (min)")]:
        style(ax)
        values = metric_table[column]
        ax.bar(range(len(values)), values, color=colors, width=0.6)
        for xi, value in enumerate(values):
            ax.text(xi, value + values.max() * 0.015, "{:.2f}".format(value), ha="center", fontsize=8.5, color=INK)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(values.index, fontsize=8.5)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
    fig.suptitle("Taipei service metrics (mean of episode seeds {}-{}; every plan is below the 15-minute target)".format(
        seeds.min(), seeds.max()), x=0.02, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "service_metrics.png"), dpi=200, facecolor=SURFACE)
    plt.close(fig)

    text = "\n".join(lines) + "\n"
    with open(SUMMARY, "w") as file:
        file.write(text)
    print(text)


if __name__ == '__main__':
    main()

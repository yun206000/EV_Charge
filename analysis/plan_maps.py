import argparse
import os
import pickle
import numpy as np
import pandas as pd
import osmnx as ox
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

"""
Maps of the Taipei charging plans before (zero-shot) and after fine-tuning (beam search), one figure per episode seed,
written to Images/Result_Plots/Transfer/beam/map_seed<seed>.png:

    grey   station in both plans (includes the 29 existing stations)
    green  station only in the plan before fine-tuning
    blue   station only in the plan after fine-tuning
    (every station of these plans ends with 15 chargers, so all markers have the same size)

    python -m analysis.plan_maps [--label B4k2] [--seeds 1 2 3 4 5]
"""

PLANS = "Results/Transfer/plans"
OUT_DIR = "Images/Result_Plots/Transfer/beam"
BOTH, BEFORE, AFTER = "#52514e", "#1baf7a", "#2a78d6"
INK, MUTED, ROAD, SURFACE = "#0b0b0b", "#52514e", "#d6d5d0", "#fcfcfb"


def load_plan(plan_dir):
    """{node id: number of chargers} of a saved plan"""
    with open(os.path.join(plan_dir, "plan_RL.pkl"), "rb") as file:
        plan = pickle.load(file)
    stations = {}
    for station in plan:
        node = station[0][0] if isinstance(station[0], (tuple, list)) else station[0]
        stations[node] = int(np.sum(station[1]))
    return stations


def draw(ax, graph, stations, other, own_color, title, subtitle):
    ox.plot_graph(graph, ax=ax, node_size=0, edge_color=ROAD, edge_linewidth=0.3, bgcolor=SURFACE,
                  show=False, close=False)
    ax.set_facecolor(SURFACE)
    for in_other, color, zorder in [(True, BOTH, 3), (False, own_color, 4)]:
        nodes = [node for node in stations if (node in other) == in_other]
        if nodes:
            ax.scatter([graph.nodes[n]["x"] for n in nodes], [graph.nodes[n]["y"] for n in nodes],
                       s=16, color=color, linewidths=0.4,
                       edgecolors=SURFACE, zorder=zorder)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=18)
    ax.text(0, 1.01, subtitle, transform=ax.transAxes, fontsize=8.5, color=MUTED)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="B4k2", help="beam search plans Results/Transfer/plans/<label>_e<seed>")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    graph = ox.load_graphml("Graph/Taipei/Taipei.graphml")
    metrics = pd.concat([pd.read_csv(os.path.join(PLANS, "metrics.csv")),
                         pd.read_csv(os.path.join(PLANS, "metrics_{}.csv".format(args.label)))])
    metrics["name"] = metrics.plan.str.rstrip("/").str.split("/").str[-1]
    metrics = metrics.set_index("name")

    for seed in args.seeds:
        before_name, after_name = "zeroshot_e{}".format(seed), "{}_e{}".format(args.label, seed)
        before = load_plan(os.path.join(PLANS, before_name))
        after = load_plan(os.path.join(PLANS, after_name))
        added, removed = len(set(after) - set(before)), len(set(before) - set(after))
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 7.2), facecolor=SURFACE)
        for ax, stations, other, color, name, title in [
                (axes[0], before, after, BEFORE, before_name, "Before fine-tuning (zero-shot)"),
                (axes[1], after, before, AFTER, after_name, "After fine-tuning (beam search)")]:
            row = metrics.loc[name]
            draw(ax, graph, stations, other, color, title,
                 "{} stations | avg travel {:.2f} min, avg waiting {:.3f} min".format(
                     len(stations), row.travel_avg_min, row.waiting_avg_min))
        legend = [Line2D([0], [0], marker="o", lw=0, color=BOTH, markersize=7, label="In both plans"),
                  Line2D([0], [0], marker="o", lw=0, color=BEFORE, markersize=7, label="Only before fine-tuning"),
                  Line2D([0], [0], marker="o", lw=0, color=AFTER, markersize=7, label="Only after fine-tuning")]
        fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, fontsize=9.5)
        fig.suptitle("Taipei charging stations, episode seed {}: {} stations added, {} removed by fine-tuning".format(
            seed, added, removed), x=0.02, ha="left", fontsize=13, color=INK)
        fig.tight_layout(rect=(0, 0.05, 1, 0.93))
        out_file = os.path.join(OUT_DIR, "map_seed{}.png".format(seed))
        fig.savefig(out_file, dpi=200, facecolor=SURFACE)
        plt.close(fig)
        print("{}: {} -> {} stations, {} added, {} removed".format(out_file, len(before), len(after), added, removed))


if __name__ == '__main__':
    main()

import os
import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

"""
Figures of the training phase (image records), written to Images/Result_Plots/Training/:

    original_Tainan_training.png  reinforcement.py on Tainan (env_plus): mean reward of the last 3 episodes,
                                  checked every 400 steps (tmp_Tainan/train.log), saved best models
    source_Tainan_training.png    transfer/train.py on Tainan (transfer/env.py): reward of every training episode
                                  (runs/source_Tainan/monitor.csv) and greedy score of every checkpoint
                                  (Results/Transfer/source_selection.csv)
    Tainan_service_metrics.png    average travel and waiting time of the Tainan plans vs the 15-minute target
                                  (Results/Tainan/metrics.csv, written by analysis/plan_metrics.py)

    python -m analysis.training_plots
"""

ORIGINAL_GREEDY = 7.9091  # greedy episode of best_model_Tainan_11200, logs/training/original_model.log
MAIN, SECOND, REF, SOURCE = "#2a78d6", "#eb6834", "#52514e", "#1baf7a"
TARGET_MIN = 15  # service target: average travel time and average waiting time below 15 minutes
TAINAN_PLANS = [("existing", "Existing\nstations", REF), ("original_model", "Original\nPCRL model", MAIN),
                ("source_model", "Source\nmodel", SOURCE)]
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


def read_train_log(path):
    """(steps, mean reward of the last 3 episodes, saved?) for every check of SaveOnBestTrainingRewardCallback"""
    rows, steps = [], None
    for line in open(path):
        match = re.match(r"Num timesteps: (\d+)", line)
        if match:
            steps = int(match.group(1))
        match = re.search(r"Last mean reward per episode: ([-\d.]+)", line)
        if match:
            rows.append(dict(steps=steps, reward=float(match.group(1)), saved=False))
        if line.startswith("Saving new best model") and rows:
            rows[-1]["saved"] = True
    return pd.DataFrame(rows)


def original_training(out_file):
    log = read_train_log("tmp_Tainan/train.log")
    fig, ax = plt.subplots(figsize=(7.5, 4.5), facecolor=SURFACE)
    style(ax)
    ax.plot(log.steps, log.reward, color=MAIN, linewidth=2, label="Mean training reward, last 3 episodes")
    saved = log[log.saved]
    ax.scatter(saved.steps, saved.reward, color=MAIN, s=22, zorder=3, label="Saved as new best model")
    chosen = log[log.steps == 11200].iloc[0]
    ax.annotate("best_model_Tainan_11200\n(used model)", (chosen.steps, chosen.reward), xytext=(0, 18),
                textcoords="offset points", ha="center", fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))
    ax.axhline(ORIGINAL_GREEDY, color=SECOND, linewidth=1.5, linestyle=":",
               label="Greedy episode of the used model ({:.2f})".format(ORIGINAL_GREEDY))
    ax.set_ylim(6.8, 8.8)
    ax.set_xlabel("Training steps on Tainan", color=MUTED, fontsize=9)
    ax.set_ylabel("Episode reward = score improvement", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    finish(fig, ax, "Original PCRL model: DQN training on Tainan",
           "reinforcement.py, observation 127,865 values, net [64, 64], checked every 400 steps",
           out_file)
    return log


def source_training(out_file):
    monitor = pd.read_csv("runs/source_Tainan/monitor.csv", skiprows=1)
    monitor["steps"] = monitor.l.cumsum()
    selection = pd.read_csv("Results/Transfer/source_selection.csv").sort_values("train_steps")
    fig, ax = plt.subplots(figsize=(7.5, 4.5), facecolor=SURFACE)
    style(ax)
    ax.scatter(monitor.steps, monitor.r, color=SECOND, s=14, alpha=0.8, zorder=3,
               label="Training episode (ε-greedy exploration)")
    ax.plot(selection.train_steps, selection.improvement, color=MAIN, linewidth=2, marker="o", markersize=4,
            label="Checkpoint, greedy episode")
    best = selection[selection.train_steps == 17000].iloc[0]
    ax.annotate("selected: 17,000 steps ({:.2f})".format(best.improvement), (best.train_steps, best.improvement),
                xytext=(-10, 16), textcoords="offset points", ha="right", fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))
    ax.axvline(1000, color=GRID, linewidth=1)
    ax.text(1100, 0.3, "learning starts", fontsize=8, color=MUTED)
    ax.set_ylim(0, 9.8)
    ax.set_xlabel("Training steps on Tainan", color=MUTED, fontsize=9)
    ax.set_ylabel("Score improvement over existing plan", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    finish(fig, ax, "Source model: DQN training on Tainan (city-independent observation)",
           "transfer/train.py, observation 324 values, net [256, 256]; 1,000-step checkpoint = untrained net, excluded",
           out_file)
    return monitor, selection


def tainan_service(out_file):
    metrics = pd.read_csv("Results/Tainan/metrics.csv").drop_duplicates("label", keep="last").set_index("label")
    plans = [(label, name, color) for label, name, color in TAINAN_PLANS if label in metrics.index]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2), facecolor=SURFACE, sharey=True)
    for ax, column, title in [(axes[0], "travel_avg_min", "Average travel time (min)"),
                              (axes[1], "waiting_avg_min", "Average waiting time (min)")]:
        style(ax)
        values = [metrics.loc[label, column] for label, _, _ in plans]
        ax.bar(range(len(plans)), values, color=[color for _, _, color in plans], width=0.6)
        for xi, value in enumerate(values):
            ax.text(xi, value + 0.25, "{:.2f}".format(value), ha="center", fontsize=8.5, color=INK)
        ax.axhline(TARGET_MIN, color=INK, linewidth=1, linestyle="--")
        ax.text(len(plans) - 0.6, TARGET_MIN + 0.3, "target < {} min".format(TARGET_MIN), ha="right", fontsize=8.5,
                color=INK)
        ax.set_xticks(range(len(plans)))
        ax.set_xticklabels([name for _, name, _ in plans], fontsize=8.5)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
    axes[0].set_ylim(0, TARGET_MIN * 1.25)
    fig.suptitle("Tainan service metrics (test_solution.py definitions)", x=0.02, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(out_file, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return metrics


def main():
    out = "Images/Result_Plots/Training"
    os.makedirs(out, exist_ok=True)
    log = original_training(os.path.join(out, "original_Tainan_training.png"))
    monitor, selection = source_training(os.path.join(out, "source_Tainan_training.png"))
    if os.path.exists("Results/Tainan/metrics.csv"):
        metrics = tainan_service(os.path.join(out, "Tainan_service_metrics.png"))
        print(metrics[["n_stations", "travel_avg_min", "waiting_avg_min"]].round(3).to_string())
    print("original: {} checks, saved at {}".format(len(log), log[log.saved].steps.tolist()))
    print("source: {} episodes, mean length {:.0f}, mean reward {:.3f}, last 10 episodes {:.3f}".format(
        len(monitor), monitor.l.mean(), monitor.r.mean(), monitor.r.tail(10).mean()))


if __name__ == '__main__':
    main()

import argparse
import ast
import csv
import json
import math
import os
import re
from datetime import datetime, timezone

"""
Training and fine-tuning logs as CSV files for trackio (Hugging Face): trackio.import_csv() reads a "step" column, an
optional "timestamp" column (ISO 8601) and numeric metric columns; a cell left empty means "not logged at this step".
trackio keeps only the first 10 letters, digits or "/" of a column name, so every name below fits that.

    Results/Trackio/training/original_Tainan.csv          original PCRL model (tmp_Tainan/train.log, evaluation episode)
    Results/Trackio/training/source_Tainan.csv            source model (runs/source_Tainan/monitor.csv, checkpoint scores)
    Results/Trackio/finetuning/Taipei_B4k2_e<seed>.csv    beam search B = 4, k = 2 on Taipei (logs/beam/B4k2_e<seed>.log)
    Results/Trackio/finetuning/Taipei_zeroshot_e<seed>.csv  B = k = 1, the greedy zero-shot episode (logs/beam/B1k1_*.log)

    python -m analysis.trackio_logs
    python -c "import trackio; trackio.import_csv('Results/Trackio/training/source_Tainan.csv', project='pcrl-source')"
"""

# exploration schedule of both trainings (README_for_doc.md 7.2, 7.4): epsilon 1.0 -> 0.05 over the first 10% of steps
EPS_INITIAL, EPS_FINAL, EPS_FRACTION = 1.0, 0.05, 0.1
ORIGINAL_STEPS = 20000

ORIGINAL_COLUMNS = ["step", "reward/avg", "reward/max", "saved", "epsilon", "eval/imp", "eval/stns"]
SOURCE_COLUMNS = ["step", "timestamp", "hours", "reward", "length", "epsilon", "eval/imp", "eval/stns"]
BEAM_COLUMNS = ["step", "score", "improve", "live/best", "done/best", "key", "beam", "stations", "budget", "seconds",
                "travel", "waiting"]

BEAM_LINE = re.compile(r"^step (\d+) beam (\d+) best live (\S+) \(key (\S+), (\d+) stations, budget (\d+)\) "
                       r"best done (\S+) \((\d+)s\)")
SERVICE_LINE = re.compile(r"step (\d+): \d+ stations, avg travel time ([\d.]+) min, avg waiting time ([\d.]+) min")


def epsilon(step, total):
    """exploration rate of stable-baselines3's linear schedule after `step` of `total` steps"""
    return EPS_INITIAL + (EPS_FINAL - EPS_INITIAL) * min(1.0, step / (EPS_FRACTION * total))


def number(text):
    """float of a logged value; None for '-' and nan"""
    try:
        value = float(text)
    except ValueError:
        return None
    return None if math.isnan(value) else value


def merge(rows):
    """one row per step (rows of the same step are combined), sorted by step"""
    by_step = {}
    for row in rows:
        by_step.setdefault(row["step"], {}).update(row)
    return [by_step[step] for step in sorted(by_step)]


def original_training(train_log, eval_monitor=None, eval_log=None, total=ORIGINAL_STEPS):
    """
    reinforcement.py checks the training every 400 steps: mean reward of the last 3 training episodes ("reward/avg"),
    best mean so far ("reward/max"), 1 if that model was saved; the greedy evaluation episode of the final model
    (model_eval.py: improvement from the monitor.csv of the evaluation, stations from its log line) is added at the
    step of the last saved model, the one model_eval.py uses.
    """
    rows, row, last_saved = [], None, None
    with open(train_log) as file:
        for line in file:
            if line.startswith("Num timesteps:"):
                row = dict(step=int(line.split(":")[1]), saved=0)
                row["epsilon"] = round(epsilon(row["step"], total), 4)
                rows.append(row)
            elif line.startswith("Best mean reward:") and row is not None:
                best, last = re.findall(r"(-?inf|-?[\d.]+)", line)[:2]
                row["reward/avg"] = float(last)
                row["reward/max"] = float(best) if best != "-inf" else float(last)
            elif line.startswith("New best mean reward:") and row is not None:
                row["reward/max"] = float(line.split(":")[1])
            elif line.startswith("Saving new best model") and row is not None:
                row["saved"], last_saved = 1, row["step"]
    if eval_monitor and last_saved is not None:
        episodes = read_monitor(eval_monitor)[1]
        if episodes:
            rows.append({"step": last_saved, "eval/imp": float(episodes[-1]["r"])})
    if eval_log and last_saved is not None:
        with open(eval_log) as file:
            stations = re.search(r"stations (\d+)", file.read())
        if stations:
            rows.append({"step": last_saved, "eval/stns": int(stations.group(1))})
    return merge(rows)


def read_monitor(path):
    """(t_start, episodes) of a stable-baselines3 monitor.csv"""
    with open(path) as file:
        header = json.loads(file.readline().lstrip("#"))
        return header["t_start"], list(csv.DictReader(file))


def source_training(monitor, selection=None, config=None):
    """
    one row per finished training episode (end step, wall-clock time, episode reward = improvement reached with
    exploration, length) and one row per checkpoint with the improvement of its greedy episode on Tainan
    (Results/Transfer/source_selection.csv)
    """
    total = 20000
    if config:
        with open(config) as file:
            total = json.load(file)["steps"]
    t_start, episodes = read_monitor(monitor)
    rows, step = [], 0
    for episode in episodes:
        step += int(episode["l"])
        seconds = float(episode["t"])
        rows.append(dict(step=step, timestamp=datetime.fromtimestamp(t_start + seconds, timezone.utc).isoformat(),
                         hours=round(seconds / 3600, 4), reward=float(episode["r"]), length=int(episode["l"]),
                         epsilon=round(epsilon(step, total), 4)))
    if selection:
        with open(selection) as file:
            for checkpoint in csv.DictReader(file):
                rows.append({"step": int(checkpoint["train_steps"]), "eval/imp": float(checkpoint["improvement"]),
                             "eval/stns": int(checkpoint["n_stations"])})
    return merge(rows)


def beam_progress(log_file):
    """
    progress of transfer/beam_search.py: every logged step the score of the best plan found so far (best of the live
    candidates and the finished ones, i.e. the plan the search would return), its improvement over the existing plan,
    the search key, number of live candidates, stations and budget of the leading candidate, elapsed seconds and, in
    logs that have them, the average travel / waiting time (minutes) of the plan of the best candidate
    """
    rows, existing = [], None
    with open(log_file) as file:
        lines = file.read().splitlines()
    for line in lines:
        if line.startswith("{"):
            existing = ast.literal_eval(line)["existing_score"]
    for line in lines:
        match = BEAM_LINE.match(line)
        service = SERVICE_LINE.search(line)
        if line.startswith("start: score"):
            rows.append(dict(step=0, score=float(line.split()[2].rstrip(","))))
        elif match:
            live, done = number(match.group(3)), number(match.group(7))
            row = dict(step=int(match.group(1)), beam=int(match.group(2)), seconds=int(match.group(8)))
            row["score"] = max(value for value in (live, done) if value is not None)
            if live is not None:
                row.update({"live/best": live, "key": number(match.group(4)), "stations": int(match.group(5)),
                            "budget": int(match.group(6))})
            if done is not None:
                row["done/best"] = done
            rows.append(row)
        elif service and ("existing plan" in line or "current plan" in line):
            rows.append(dict(step=int(service.group(1)), travel=float(service.group(2)),
                             waiting=float(service.group(3))))
    rows = merge(rows)
    if existing is not None:
        for row in rows:
            if "score" in row:
                row["improve"] = round(row["score"] - existing, 4) + 0.0  # + 0.0: no "-0.0"
    return rows


def write_csv(rows, path, columns):
    """columns without any value (e.g. travel / waiting time in logs written before they were printed) are left out"""
    columns = [column for column in columns if column == "step" or any(row.get(column) is not None for row in rows)]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Training and fine-tuning logs as trackio CSV files.")
    parser.add_argument("--root", default=".", help="repository root")
    parser.add_argument("--out", default="Results/Trackio")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = parser.parse_args(argv)
    root = args.root
    written = [
        write_csv(original_training(os.path.join(root, "tmp_Tainan", "train.log"),
                                    os.path.join(root, "tmp_Tainan", "monitor.csv"),
                                    os.path.join(root, "logs", "training", "original_model.log")),
                  os.path.join(args.out, "training", "original_Tainan.csv"), ORIGINAL_COLUMNS),
        write_csv(source_training(os.path.join(root, "runs", "source_Tainan", "monitor.csv"),
                                  os.path.join(root, "Results", "Transfer", "source_selection.csv"),
                                  os.path.join(root, "runs", "source_Tainan", "config.json")),
                  os.path.join(args.out, "training", "source_Tainan.csv"), SOURCE_COLUMNS)]
    for label, name in [("B4k2", "B4k2"), ("B1k1", "zeroshot")]:
        for seed in args.seeds:
            log_file = os.path.join(root, "logs", "beam", "{}_e{}.log".format(label, seed))
            if os.path.exists(log_file):
                written.append(write_csv(beam_progress(log_file), os.path.join(
                    args.out, "finetuning", "Taipei_{}_e{}.csv".format(name, seed)), BEAM_COLUMNS))
    for path in written:
        with open(path) as file:
            print("{}: {} rows".format(path, sum(1 for _ in file) - 1))
    return written


if __name__ == '__main__':
    main()

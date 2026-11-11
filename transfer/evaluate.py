import argparse
import csv
import fcntl
import glob
import os
import pickle
import random
import re
import time
import numpy as np
import torch
from stable_baselines3 import DQN
from transfer import env as et

"""
Score policies on a city with one greedy episode each and append the results to a CSV file.

Run from the repository root (scripts/run_generalization.sh):

Model (or every checkpoint in a folder):
    python -m transfer.evaluate --city Taipei --policy model --model runs/source_Tainan/best_source.zip \
        --label zeroshot --eval-seed 1 --save-plan Results/Transfer/plans/zeroshot_e1
Baseline:
    python -m transfer.evaluate --city Taipei --policy random --label random --seed 1

score       = normalised score of the best plan found in the episode (env.best_score)
improvement = score - score of the existing plan, i.e. the summed episode reward
"""

FIELDS = ["city", "label", "seed", "policy", "model", "train_steps", "existing_score", "score", "improvement",
          "relative_pct", "n_stations", "episode_steps", "seconds", "eval_seed"]


def train_steps_of(model_path):
    match = re.search(r"model_(\d+)_steps\.zip$", model_path)
    if match:
        return int(match.group(1))
    config = os.path.join(os.path.dirname(model_path), "config.json")
    if os.path.basename(model_path) == "final_model.zip" and os.path.exists(config):
        import json
        with open(config) as file:
            return json.load(file)["steps"]
    return 0


def run_episode(env, choose_action, seed):
    random.seed(seed)
    np.random.seed(seed)
    start = time.time()
    obs = env.reset()
    existing_score = env.best_score
    done = False
    steps = 0
    while not done:
        obs, _, done, _ = env.step(choose_action(obs))
        steps += 1
    best_node_list, best_plan = env.best_node_list, env.best_plan
    result = dict(existing_score=existing_score, score=env.best_score,
                  improvement=env.best_score - existing_score,
                  relative_pct=env.best_score / existing_score * 100 if existing_score else float("nan"),
                  n_stations=len(best_plan), episode_steps=steps,
                  seconds=round(time.time() - start, 1))
    return result, best_node_list, best_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    parser.add_argument("--policy", choices=["model", "random"], default="model")
    parser.add_argument("--model", help="model zip, or a folder whose *.zip files are all evaluated")
    parser.add_argument("--label", required=True, help="experiment group, e.g. G1_zeroshot")
    parser.add_argument("--seed", type=int, default=1, help="training seed of the model (also the episode seed)")
    parser.add_argument("--eval-seed", type=int, default=None, help="episode seed, if different from --seed")
    parser.add_argument("--csv", default="Results/Transfer/results.csv")
    parser.add_argument("--save-plan", default=None, help="folder to write plan_RL.pkl / nodes_RL.txt")
    args = parser.parse_args()

    eval_seed = args.seed if args.eval_seed is None else args.eval_seed
    torch.set_num_threads(1)  # evaluations run in parallel
    repository_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # transfer/..
    env = et.make_env(args.city, repository_root)

    if args.policy == "model":
        if os.path.isdir(args.model):
            model_paths = sorted(glob.glob(os.path.join(args.model, "*.zip")), key=train_steps_of)
        else:
            model_paths = [args.model]
    else:
        model_paths = [None]

    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
    for model_path in model_paths:
        if args.policy == "model":
            model = DQN.load(model_path, device='cpu')
            if not all(torch.isfinite(param).all() for param in model.policy.parameters()):
                raise ValueError("Model {} has non-finite weights".format(model_path))
            choose_action = lambda obs: int(model.predict(obs, deterministic=True)[0])
        else:
            rng = np.random.default_rng(eval_seed)
            choose_action = lambda obs: int(rng.integers(env.action_space.n))
        result, best_node_list, best_plan = run_episode(env, choose_action, eval_seed)
        row = dict(city=args.city, label=args.label, seed=args.seed,
                   policy=args.policy,
                   model=model_path or "", train_steps=train_steps_of(model_path) if model_path else 0,
                   eval_seed=eval_seed, **result)
        print(row)
        with open(args.csv, "a", newline="") as file:
            fcntl.flock(file, fcntl.LOCK_EX)  # evaluations run in parallel
            writer = csv.DictWriter(file, fieldnames=FIELDS, extrasaction="ignore")
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
        if args.save_plan:
            os.makedirs(args.save_plan, exist_ok=True)
            pickle.dump(best_plan, open(os.path.join(args.save_plan, "plan_RL.pkl"), "wb"))
            with open(os.path.join(args.save_plan, "nodes_RL.txt"), "w") as file:
                file.write(str(best_node_list))


if __name__ == '__main__':
    main()

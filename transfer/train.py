import argparse
import json
import os
import random
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from transfer import env as et

"""
Train a DQN agent on the city-independent environment (transfer/env.py). Run from the repository root:

    python -m transfer.train --city Tainan --out runs/source_Tainan --steps 20000 --seed 1 --threads 8
"""

NET_ARCH = [256, 256]



def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
    random.seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    parser.add_argument("--out", required=True, help="output folder for monitor.csv, checkpoints and final model")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--target-update", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor (0.99 = horizon of ~100 steps)")
    parser.add_argument("--eps-initial", type=float, default=1.0)
    parser.add_argument("--eps-final", type=float, default=0.05)
    parser.add_argument("--eps-fraction", type=float, default=0.1)
    parser.add_argument("--checkpoint-freq", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=4, help="torch CPU threads, keep low for parallel runs")
    args = parser.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(args.threads)
    repository_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # transfer/..
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as file:
        json.dump(vars(args), file, indent=2)

    env = Monitor(et.make_env(args.city, repository_root), args.out)
    model = DQN("MlpPolicy", env, verbose=1, batch_size=args.batch_size, buffer_size=args.buffer_size,
                learning_rate=args.lr, gamma=args.gamma, device='cpu', learning_starts=args.learning_starts,
                target_update_interval=args.target_update, exploration_initial_eps=args.eps_initial,
                exploration_final_eps=args.eps_final, exploration_fraction=args.eps_fraction,
                policy_kwargs=dict(net_arch=NET_ARCH), seed=args.seed)

    checkpoints = CheckpointCallback(save_freq=args.checkpoint_freq, save_path=os.path.join(args.out, "checkpoints"),
                                     name_prefix="model")
    model.learn(total_timesteps=args.steps, log_interval=10 ** 4, callback=checkpoints)
    model.save(os.path.join(args.out, "final_model"))
    print("Saved final model to {}".format(os.path.join(args.out, "final_model.zip")))


if __name__ == '__main__':
    main()

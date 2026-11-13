from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
import argparse
import os
import numpy as np
import torch
import random
import env_plus as ev
import test_solution as ts

"""
Train the model by reinforcement learning.

    python reinforcement.py --city Tainan                                  # original setting, 20,000 steps
    python reinforcement.py --city Taipei --steps 1000 --learning-starts 200
"""

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Code from Stable Baselines3, 
    https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/monitor_training.ipynb
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param my_log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, my_log_dir: str, my_modelname: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = my_log_dir
        self.modelname = my_modelname
        self.save_path = os.path.join(self.log_dir, self.modelname)
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                my_mean_reward = np.mean(y[-3:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(
                        self.best_mean_reward, my_mean_reward))

                if my_mean_reward > self.best_mean_reward:
                    self.best_mean_reward = my_mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("New best mean reward: {:.2f}".format(self.best_mean_reward))
                        # we want to make sure that the best models are not overwritten
                        new_name = self.modelname + str(self.num_timesteps)
                        if self.save_path is not None:
                            os.makedirs(self.save_path, exist_ok=True)
                        self.save_path = os.path.join(self.log_dir, new_name)
                        print("Saving new best model to {}".format(self.save_path))
                    self.model.save(self.save_path)
        return True


class ServiceMetricsCallback(BaseCallback):
    """prints the average travel and waiting time of the plan of the running episode every ``freq`` steps"""

    def __init__(self, freq: int):
        super(ServiceMetricsCallback, self).__init__()
        self.freq = freq

    def _on_step(self) -> bool:
        if self.freq > 0 and self.n_calls % self.freq == 0:
            env = self.training_env.envs[0].unwrapped
            print(ts.service_progress(self.num_timesteps, env.node_list, env.plan_instance.plan), flush=True)
        return True


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the original PCRL model (DQN on env_plus) on one city.")
    parser.add_argument("--city", default="Tainan", help="city folder in Graph/ (Tainan, Taipei, Toy_Example, ...)")
    parser.add_argument("--steps", type=int, default=20000, help="total training steps")
    parser.add_argument("--learning-starts", type=int, default=1000,
                        help="steps of random exploration before the first network update")
    parser.add_argument("--check-freq", type=int, default=400, help="check / save the best model every n steps")
    parser.add_argument("--metrics-freq", type=int, default=50,
                        help="print the average travel / waiting time of the current plan every n steps (0 = off)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-dir", default=None, help="model and monitor folder (default: <root>/tmp_<city>)")
    parser.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)),
                        help="repository root holding Graph/<city>/ (default: folder of this script)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    # set a seed for reproducibility
    os.environ['PYTHONASHSEED'] = '0'
    seed = args.seed
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
    random.seed(seed)
    # Instantiate the env
    location = args.city
    repository_root = args.root
    graph_dir = os.path.join(repository_root, "Graph", location)
    graph_file = os.path.join(graph_dir, location + ".graphml")
    node_file = os.path.join(graph_dir, "node_list_" + location + ".txt") # nodes_extended_
    plan_file = os.path.join(graph_dir, "existingplan_" + location + ".pkl")

    env = ev.StationPlacement(graph_file, node_file, plan_file)
    log_dir = args.log_dir or os.path.join(repository_root, "tmp_" + location)
    modelname = "best_model_" + location + "_"
    os.makedirs(log_dir, exist_ok=True)

    """
    Define and train the agent 
    """
    env = Monitor(env, log_dir)
    model = DQN("MlpPolicy", env, verbose=1, batch_size=128, buffer_size=10000, learning_rate=0.001, device='cpu',
                learning_starts=args.learning_starts, target_update_interval=1000, seed=seed)
    callback = SaveOnBestTrainingRewardCallback(check_freq=args.check_freq, my_log_dir=log_dir, my_modelname=modelname)
    model.learn(total_timesteps=args.steps, log_interval=10 ** 4,
                callback=[callback, ServiceMetricsCallback(args.metrics_freq)])
    return model


if __name__ == '__main__':
    main()

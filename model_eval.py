import argparse
import glob
import re
import gym_compat as gc
from stable_baselines3.common.monitor import Monitor
import pickle
import os
import env_plus as ev
import test_solution as ts

"""
Generate a charging plan based on the model.

    python model_eval.py --city Tainan          # model tmp_Tainan/best_model_Tainan_11200.zip -> Results/Tainan/
    python model_eval.py --city Taipei --model tmp_Taipei/best_model_Taipei_800.zip
"""


def latest_model(log_dir, location):
    """best_model_<city>_<steps>.zip with the most steps: the callback of reinforcement.py saves a model only when
    the mean reward improved, so the last one saved is the best one"""
    models = glob.glob(os.path.join(log_dir, "best_model_" + location + "_*.zip"))
    matches = [re.search(r"_(\d+)\.zip$", path) for path in models]
    steps = [int(match.group(1)) for match in matches if match]
    if not steps:
        raise FileNotFoundError("no best_model_{}_<steps>.zip in {}; train with reinforcement.py --city {} first"
                                .format(location, log_dir, location))
    return os.path.join(log_dir, "best_model_{}_{}.zip".format(location, max(steps)))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Charging plan of a trained original PCRL model (greedy episode).")
    parser.add_argument("--city", default="Tainan", help="city folder in Graph/ (Tainan, Taipei, Toy_Example, ...)")
    parser.add_argument("--model", default=None,
                        help="model .zip (default: the best_model_<city>_<steps>.zip with the most steps in tmp_<city>)")
    parser.add_argument("--output", default=None,
                        help="folder for plan_RL.pkl and nodes_RL.txt (default: <root>/Results/<city>)")
    parser.add_argument("--metrics-freq", type=int, default=50,
                        help="print the average travel / waiting time of the current plan every n steps (0 = off)")
    parser.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)),
                        help="repository root holding Graph/<city>/ and tmp_<city>/ (default: folder of this script)")
    args = parser.parse_args(argv)

    # Instantiate the env
    location = args.city
    repository_root = args.root
    graph_dir = os.path.join(repository_root, "Graph", location)
    graph_file = os.path.join(graph_dir, location + ".graphml")
    node_file = os.path.join(graph_dir, "node_list_" + location + ".txt") # nodes_extended_
    plan_file = os.path.join(graph_dir, "existingplan_" + location + ".pkl")

    env = ev.StationPlacement(graph_file, node_file, plan_file)
    log_dir = os.path.join(repository_root, "tmp_" + location)
    os.makedirs(log_dir, exist_ok=True)
    model_file = args.model or latest_model(log_dir, location)

    """
    Ab hier kommt die Evaluation.
    """
    print("Evaluation for best model {}".format(model_file))
    env = Monitor(env, log_dir)  # new environment for evaluation

    model = gc.load_dqn(model_file, env)

    obs = gc.reset(env)
    done = False
    best_plan, best_node_list = None, None
    step = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = gc.step(env, action)
        step += 1
        if args.metrics_freq > 0 and (step % args.metrics_freq == 0 or done):
            print(ts.service_progress(step, env.unwrapped.node_list, env.unwrapped.plan_instance.plan), flush=True)
        if done:
            best_node_list, best_plan = env.unwrapped.render()

    results_dir = args.output or os.path.join(repository_root, "Results", location)
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "plan_RL.pkl"), "wb") as file:
        pickle.dump(best_plan, file)
    with open(os.path.join(results_dir, "nodes_RL.txt"), 'w') as file:
        file.write(str(best_node_list))
    print("Plan saved to {}".format(results_dir))
    return best_plan, best_node_list


if __name__ == '__main__':
    main()

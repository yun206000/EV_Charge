from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import pickle
import os
import env_plus as ev

"""
Generate a charging plan based on the model.
"""
# Instantiate the env
location = "Toy_Example"

repository_root = os.path.dirname(os.path.abspath(__file__))
graph_dir = os.path.join(repository_root, "Graph", location)
graph_file = os.path.join(graph_dir, location + ".graphml")
node_file = os.path.join(graph_dir, "nodes_extended_" + location + ".txt")
plan_file = os.path.join(graph_dir, "existingplan_" + location + ".pkl")

env = ev.StationPlacement(graph_file, node_file, plan_file)
log_dir = os.path.join(repository_root, "tmp_Toy_Example")
os.makedirs(log_dir, exist_ok=True)

"""
Ab hier kommt die Evaluation.
"""
print("Evaluation for best model")
env = Monitor(env, log_dir)  # new environment for evaluation

model = DQN.load(
    os.path.join(log_dir, "best_model_" + location + "_2000.zip"),
    device="cpu",
)

obs = env.reset()
done = False
best_plan, best_node_list = None, None
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        best_node_list, best_plan = env.unwrapped.render()


results_dir = os.path.join(repository_root, "Results", location)
os.makedirs(results_dir, exist_ok=True)
pickle.dump(best_plan, open(os.path.join(results_dir, "plan_RL.pkl"), "wb"))
with open(os.path.join(results_dir, "nodes_RL.txt"), 'w') as file:
    file.write(str(best_node_list))

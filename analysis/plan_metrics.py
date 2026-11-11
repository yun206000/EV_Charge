import ast
import argparse
import csv
import fcntl
import os
import pickle
import evaluation_framework as ef
import test_solution as ts

"""
Service metrics of saved charging plans, with the definitions of test_solution.py:

    travel time  (minutes) demand-weighted average and maximum over all nodes
    waiting time (minutes) weighted by the EV flow D_s of the stations, average and maximum

    python -m analysis.plan_metrics --city Taipei --plan Results/Transfer/plans/zeroshot_e1 --label zeroshot_e1 \
        --csv Results/Transfer/plans/metrics.csv
    python -m analysis.plan_metrics --city Taipei --existing --csv Results/Transfer/plans/metrics.csv
"""

FIELDS = ["city", "label", "plan", "n_stations", "travel_avg_min", "travel_max_min", "waiting_avg_min",
          "waiting_max_min"]


def load_existing(city):
    graph_dir = os.path.join("Graph", city)
    node_file = os.path.join(graph_dir, "node_list_" + city + ".txt")
    ef.prepare_graph(os.path.join(graph_dir, city + ".graphml"), node_file)
    with open(node_file) as file:
        node_list = ast.literal_eval(file.readline())
    with open(os.path.join(graph_dir, "existingplan_" + city + ".pkl"), "rb") as file:
        plan = pickle.load(file)
    node_list, plan = ts.prepare_existing_plan(plan, node_list)
    return node_list, plan


def metrics(plan, node_list):
    travel_avg, travel_max = ts.travel_metric(node_list)
    wait_avg, wait_max = ts.waiting_metric(plan)
    return dict(n_stations=len(plan), travel_avg_min=travel_avg, travel_max_min=travel_max,
                waiting_avg_min=wait_avg, waiting_max_min=wait_max)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    parser.add_argument("--plan", help="folder with plan_RL.pkl and nodes_RL.txt")
    parser.add_argument("--label", default="")
    parser.add_argument("--existing", action="store_true", help="score the existing plan of the city")
    parser.add_argument("--csv", default="Results/Transfer/plans/metrics.csv")
    args = parser.parse_args()

    if args.existing:
        node_list, plan = load_existing(args.city)
        row = metrics(plan, node_list)
        label, plan_dir = args.label or "existing", ""
    else:
        with open(os.path.join(args.plan, "nodes_RL.txt")) as file:
            result_nodes = ast.literal_eval(file.readline())
        with open(os.path.join(args.plan, "plan_RL.pkl"), "rb") as file:
            result_plan = pickle.load(file)
        graph_dir = os.path.join("Graph", args.city)
        ef.prepare_graph(os.path.join(graph_dir, args.city + ".graphml"),
                         os.path.join(graph_dir, "node_list_" + args.city + ".txt"))
        result_nodes, result_plan = ts.prepare_existing_plan(result_plan, result_nodes)
        row = metrics(result_plan, result_nodes)
        label, plan_dir = args.label, args.plan
    row.update(city=args.city, label=label, plan=plan_dir)
    print(row)
    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
    with open(args.csv, "a", newline="") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow(row)


if __name__ == '__main__':
    main()

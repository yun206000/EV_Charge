import ast
import evaluation_framework as ef
import pickle
from math import ceil
import osmnx as ox

"""
Calculate evaluation metrics for the created charging plan.
"""

def travel_metric(my_node_list):
    """Return demand-weighted average and maximum travel time in minutes."""
    big_travel_list = []
    for my_node in my_node_list:
        travel = my_node[1].get('travel time', my_node[1]['distance'] / ef.VELOCITY) * 60
        times = ceil(10 * ef.weak_demand(my_node))
        for time in range(times):
            big_travel_list.append(travel)
    return sum(big_travel_list) / len(big_travel_list), max(big_travel_list)


def waiting_metric(my_plan):
    """Return demand-weighted average and maximum waiting time in minutes."""
    big_waiting_list = []
    for my_station in my_plan:
        times = ceil(my_station[2]["D_s"])
        for time in range(times):
            big_waiting_list.append(my_station[2]["W_s"] * 60)
    return sum(big_waiting_list) / len(big_waiting_list), max(big_waiting_list)


def eci_test(my_plan, my_node_list, my_norm_benefit, my_norm_charging, my_norm_waiting,
             my_norm_travel):
    score, benefit, cost, charg_time, wait_time, cost_travel = ef.norm_score(my_plan, my_node_list, my_norm_benefit,
                                                                             my_norm_charging, my_norm_waiting,
                                                                             my_norm_travel)
    return score


def test(my_plan, my_node_list, my_basic_cost, my_norm_benefit, my_norm_charging, my_norm_waiting,
         my_norm_travel, my_norm_score):
    """
    prints results of the evaulation metrics
    """
    travel_average, travel_max = travel_metric(my_node_list)
    wait_average, wait_max = waiting_metric(my_plan)
    score, benefit, cost, charg_time, wait_time, cost_travel = ef.norm_score(my_plan, my_node_list, my_norm_benefit,
                                                                             my_norm_charging, my_norm_waiting,
                                                                             my_norm_travel)
    # test if solution satisfies all constraints
    ef.constraint_check(my_plan, my_node_list, my_basic_cost)
    total_inst_cost = (sum([my_station[2]["fee"] for my_station in my_plan]) - my_basic_cost) / ef.BUDGET
    score = score / my_norm_score * 100
    travel_time_minutes = ef.travel_cost(my_node_list) * 60
    waiting_time_minutes = ef.waiting_time(my_plan) * 60
    charging_time_minutes = ef.charging_time(my_plan) * 60
    # print("The score is {}".format(score))
    # print("Benefit: {}".format(benefit * 100))
    # print("Waiting time: {}, Travel time: {}, Charging time: {}".format(wait_time * 100, cost_travel * 100,
    #                                                                     charg_time * 100))
    # print("Actual total time (minutes) - Waiting: {:.3f}, Travel: {:.3f}, Charging: {:.3f}".format(
    #     waiting_time_minutes, travel_time_minutes, charging_time_minutes))
    print("Travel time (minutes) - Average: {:.3f}, Maximum: {:.3f}".format(travel_average, travel_max))
    print("Waiting time (minutes) - Average: {:.3f}, Maximum: {:.3f}".format(wait_average, wait_max))
    print("Used budget: {} \n".format(total_inst_cost * 100))


def prepare_existing_plan(my_plan, my_node_list):
    my_cost_dict = {}
    my_node_dict = {}
    node_by_id = {my_node[0]: my_node for my_node in my_node_list}
    for my_station in my_plan:
        station_id = my_station[0][0]
        if station_id not in node_by_id:
            raise ValueError("Station node {} is missing from the node list".format(station_id))
        my_station[0] = node_by_id[station_id]

    for my_node in my_node_list:
        my_node_dict[my_node[0]] = {}  # prepare node_dict
        my_node[1]["charging station"] = None
        my_node[1]["distance"] = None

    for j in range(2):
        for index in range(len(my_plan)):
            my_plan[index] = ef.s_dictionnary(my_plan[index], my_node_list)
        my_node_list, _, _ = ef.station_seeking(my_plan, my_node_list, my_node_dict, my_cost_dict)
        j += 1
    for index in range(len(my_plan)):
        my_plan[index] = ef.s_dictionnary(my_plan[index], my_node_list)
    return my_node_list, my_plan


def perform_test(my_node_file, my_basic_cost, my_result_file, my_norm_benefit, my_norm_charging,
                 my_norm_waiting, my_norm_travel, my_norm_score):
    with open(my_node_file, "r") as file:
        my_node_list = ast.literal_eval(file.readline())
    with (open(my_result_file, "rb")) as f:
        my_plan = pickle.load(f)
    print("Number of charging stations: {}".format(len(my_plan)))
    my_node_list, my_plan = prepare_existing_plan(my_plan, my_node_list)
    test(my_plan, my_node_list, my_basic_cost, my_norm_benefit, my_norm_charging, my_norm_waiting,
         my_norm_travel, my_norm_score)


if __name__ == '__main__':
    location = "Tainan"

    graph_file = "Graph/" + location + "/" + location + ".graphml"
    node_file = "Graph/" + location + "/node_list_" + location + ".txt"
    graph, _ = ef.prepare_graph(graph_file, node_file)
    """
    Test existing charging stations.
    """
    with open(node_file, "r") as file:
        node_list = ast.literal_eval(file.readline())
    with (open("Graph/" + location + "/existingplan_" + location + ".pkl", "rb")) as f:
        plan = pickle.load(f)
    print("Number of already existing charging stations: {}".format(len(plan)))

    node_list, plan = prepare_existing_plan(plan, node_list)
    basic_cost = sum([station[2]["fee"] for station in plan])
    norm_benefit, norm_cost, norm_charging, norm_waiting, norm_travel = ef.existing_score(plan, node_list)
    norm_score = eci_test(plan, node_list, norm_benefit, norm_charging, norm_waiting, norm_travel)
    test(plan, node_list, basic_cost, norm_benefit, norm_charging, norm_waiting, norm_travel, norm_score)

    print("Reinforcement Learning")
    node_file = "Results/" + location + "/nodes_RL.txt"
    result_file = "Results/" + location + "/plan_RL.pkl"
    perform_test(node_file, basic_cost, result_file, norm_benefit, norm_charging, norm_waiting, norm_travel, norm_score)

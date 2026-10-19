import osmnx as ox
import networkx as nx
import numpy as np
from math import sin, cos, sqrt, atan2, radians
import re

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as csgraph_dijkstra
from scipy.spatial import cKDTree

"""
Utility model and help functions.

OSM ROUTING REWRITE
--------------------
The original implementation called nx.shortest_path_length() once per
(node, station) pair. That is wasteful for this problem's access pattern:
a handful of station locations are each queried against every node, over
and over, as candidate plans are evaluated. A single-source Dijkstra run
already has to explore roughly the same part of the graph whether it stops
at one target or returns distances to every node, so instead of paying for
a fresh shortest-path search per pair, we now pay for ONE Dijkstra run per
distinct station location (source) and cache the resulting "distance to
every node" row. Every later lookup against that station is then an O(1)
array read.

This is done with scipy's C-level `dijkstra` on a precomputed CSR sparse
matrix (built once, at graph-load time), rather than networkx's pure-Python
Dijkstra, which gives a further large constant-factor speedup on top of the
caching change. A one-time KD-tree replaces the previous
ox.distance.nearest_nodes() calls, which were silently rebuilding a spatial
index from scratch on every single lookup.

Note on scope: this optimizes the "few repeated sources, many targets"
pattern, which is what this codebase actually needs. It is not a full
Contraction Hierarchies implementation (which mainly pays off for
*arbitrary* point-to-point queries at huge scale, e.g. via OSRM/Valhalla or
NetworKit's pruned landmark labeling). If the access pattern ever changes
to "many distinct arbitrary pairs, queried once each," that's the point at
which a real CH/PLL structure would be worth the added complexity.
"""

_OSM_GRAPH = None
_OSM_NODE_CACHE = {}       # user node id (from my_node_list / station pos) -> matrix row index
_OSM_ROUTE_CACHE = {}      # matrix row index (station) -> [time_row, length_row, undirected_fallback_or_None]

_OSM_NODE_INDEX = {}       # OSM graph node id -> matrix index
_OSM_INDEX_NODE = []       # matrix index -> OSM graph node id
_OSM_KDTREE = None         # cKDTree over (x, y) of every OSM graph node, built once

_OSM_TIME_MATRIX = None                # CSR, directed, edge weight = travel time in seconds
_OSM_LENGTH_MATRIX = None              # CSR, directed, edge weight = length in meters
_OSM_TIME_MATRIX_UNDIRECTED = None     # fallback for when a directed route doesn't exist
_OSM_LENGTH_MATRIX_UNDIRECTED = None


def prepare_graph(my_graph_file, my_node_file):
    """
    loads graph and nodes prepared in load_graph.py
    """
    global _OSM_GRAPH, _OSM_NODE_CACHE, _OSM_ROUTE_CACHE
    my_graph = ox.load_graphml(my_graph_file)
    _OSM_GRAPH = my_graph
    _OSM_NODE_CACHE = {}
    _OSM_ROUTE_CACHE = {}
    _build_sparse_graph(my_graph)
    with open(my_node_file, "r") as file:
        my_node_list = eval(file.readline())
    return my_graph, my_node_list


def cost_single(my_node, my_station, my_node_dict, my_cost_dict):
    """
    calculate the social cost for one station
    """
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    # check if distance has to be calculated
    if s_pos[0] in my_node_dict[my_node[0]]:
        distance = my_node_dict[my_node[0]][s_pos[0]]
    else:
        distance, travel_time = _osm_route(s_pos, my_node)
        my_node_dict[my_node[0]][s_pos[0]] = distance
        my_node[1]["travel time"] = travel_time
    # check if cost has to be calculated
    try:
        _a = my_cost_dict[my_node[0]]
    except KeyError:
        my_cost_dict[my_node[0]] = {}
    if s_pos[0] in my_cost_dict[my_node[0]]:
        cost_node = my_cost_dict[my_node[0]][s_pos[0]]
    else:
        cost_travel = alpha * my_node[1].get("travel time", distance / VELOCITY)
        cost_boring = (1 - alpha) / distance * (s_dict["W_s"] + 1 / s_dict["service rate"])
        cost_node = weak_demand(my_node) * (cost_travel + cost_boring)
        my_cost_dict[my_node[0]][s_pos[0]] = cost_node
    return cost_node, my_node_dict, my_cost_dict


def station_seeking(my_plan, my_node_list, my_node_dict, my_cost_dict):
    """
    output station assignment: Each node gets assigned the charging station with minimal social cost
    """
    for the_node in my_node_list:
        cost_list = [cost_single(the_node, my_station, my_node_dict, my_cost_dict) for my_station in my_plan]
        costminindex = cost_list.index(min(cost_list))
        chosen_station = my_plan[costminindex]
        s_pos = chosen_station[0]
        the_node[1]["charging station"] = s_pos[0]
        the_node[1]["distance"] = my_node_dict[the_node[0]][s_pos[0]]
        the_node[1]["travel time"] = _osm_route(s_pos, the_node)[1]
    return my_node_list, my_node_dict, my_cost_dict


################################################################################################
def installment_fee(my_station):
    """
    returns cost to install the respective chargers at that position
    """
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    charger_cost = np.sum(INSTALL_FEE * s_x)
    fee = price_parkingplace * s_pos[1]['estate price'] + charger_cost
    s_dict["fee"] = fee  # [fee] = €
    return my_station


def charging_capability(my_station):
    """
    returns the summed up charging capability of the CS
    """
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    total_capacity = np.sum(CHARGING_POWER * s_x)
    s_dict["capability"] = total_capacity  # [capability] = kw
    return my_station


def weak_demand(my_node):
    return my_node[1]["demand"] * (1 - 0.1 * my_node[1]["private CS"])


def influence_radius(my_station):
    """
    gives the radius of the nodes whose charging demand the CS could satisfy
    """
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    total_capacity = s_dict["capability"]
    radius_s = RADIUS_MAX * 1 / (1 + np.exp(-total_capacity / (100 * capacity_unit)))
    s_dict["radius"] = radius_s  # [radius] = m
    return my_station


def haversine(s_pos, my_node):
    """
    yields the approximate distance of two GPS points, middle computational cost
    """
    lon1, lat1 = s_pos[1]['x'], s_pos[1]['y']
    R_earth = 6372800  # approximate radius of earth. [R_earth] = m
    lon2, lat2 = my_node[1]['x'], my_node[1]['y']
    dlon = radians(lon2 - lon1)
    dlat = radians(lat2 - lat1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R_earth * c  # [distance] = m
    if distance < 0.1:  # to avoid ZeroDivisionError
        distance = 0.1
    return distance


def osm_travel_time(s_pos, my_node):
    """Return estimated driving time in hours using the loaded OSM road graph."""
    return _osm_route(s_pos, my_node)[1]


def _osm_route(s_pos, my_node):
    if _OSM_TIME_MATRIX is None:
        raise RuntimeError("An OSM graph must be loaded with prepare_graph before calculating road travel time")

    source_idx = _nearest_osm_node(s_pos)
    target_idx = _nearest_osm_node(my_node)
    return _distance_and_time(source_idx, target_idx)


def _distance_and_time(source_idx, target_idx):
    """
    Look up (distance, travel_time) between two already-resolved matrix
    indices, computing and caching the station's full one-to-all row the
    first time it's needed.
    """
    time_row, length_row, undirected = _station_distance_rows(source_idx)

    distance_m = length_row[target_idx]
    travel_time_h = time_row[target_idx] / 3600

    if not np.isfinite(distance_m) or not np.isfinite(travel_time_h):
        # no route in the directed graph (e.g. one-way streets blocking the
        # direct path) -- fall back to the undirected graph, same as the
        # original code's `_OSM_GRAPH.to_undirected()` retry
        if undirected is None:
            und_time_row = csgraph_dijkstra(_OSM_TIME_MATRIX_UNDIRECTED, directed=True, indices=source_idx)
            und_length_row = csgraph_dijkstra(_OSM_LENGTH_MATRIX_UNDIRECTED, directed=True, indices=source_idx)
            undirected = (und_time_row, und_length_row)
            _OSM_ROUTE_CACHE[source_idx][2] = undirected

        distance_m = undirected[1][target_idx]
        travel_time_h = undirected[0][target_idx] / 3600

        if not np.isfinite(distance_m):
            raise ValueError("No drivable OSM route exists between the two points")

    return max(float(distance_m), 0.1), max(float(travel_time_h), 0.0)


def _station_distance_rows(source_idx):
    """
    One-to-all shortest paths (by time and by length) from a single OSM
    graph node index, computed once with scipy's C-level Dijkstra and
    cached for the lifetime of the loaded graph. Every later query against
    this same source is then just an array read (O(1)) instead of a fresh
    shortest-path search.
    """
    if source_idx in _OSM_ROUTE_CACHE:
        return _OSM_ROUTE_CACHE[source_idx]

    time_row = csgraph_dijkstra(_OSM_TIME_MATRIX, directed=True, indices=source_idx)
    length_row = csgraph_dijkstra(_OSM_LENGTH_MATRIX, directed=True, indices=source_idx)

    _OSM_ROUTE_CACHE[source_idx] = [time_row, length_row, None]
    return _OSM_ROUTE_CACHE[source_idx]


def _nearest_osm_node(position):
    """
    Resolve a (node_id, attrs) position to its nearest OSM graph node's
    matrix index, using a KD-tree built once at graph-load time instead of
    rebuilding a spatial index on every call.
    """
    node_id, attributes = position
    if node_id not in _OSM_NODE_CACHE:
        _, idx = _OSM_KDTREE.query([float(attributes["x"]), float(attributes["y"])])
        _OSM_NODE_CACHE[node_id] = int(idx)
    return _OSM_NODE_CACHE[node_id]


def _build_sparse_graph(graph):
    """
    One-time preprocessing: turn the osmnx MultiDiGraph into CSR sparse
    matrices (directed + undirected-fallback, both time- and
    length-weighted) plus a KD-tree over node coordinates. Everything that
    used to happen per-query in the old code (parsing maxspeed, resolving
    parallel edges, finding nearest nodes) now happens once here.
    """
    global _OSM_TIME_MATRIX, _OSM_LENGTH_MATRIX
    global _OSM_TIME_MATRIX_UNDIRECTED, _OSM_LENGTH_MATRIX_UNDIRECTED
    global _OSM_NODE_INDEX, _OSM_INDEX_NODE, _OSM_KDTREE

    _OSM_INDEX_NODE = list(graph.nodes())
    _OSM_NODE_INDEX = {node_id: idx for idx, node_id in enumerate(_OSM_INDEX_NODE)}
    n = len(_OSM_INDEX_NODE)

    coords = np.array(
        [[graph.nodes[node_id]["x"], graph.nodes[node_id]["y"]] for node_id in _OSM_INDEX_NODE]
    )
    _OSM_KDTREE = cKDTree(coords)

    # For parallel edges (a MultiDiGraph can have several roads between the
    # same two nodes) keep the minimum time and minimum length separately,
    # matching networkx's own behaviour of taking the min weight per
    # (u, v) pair on a multigraph.
    best_time = {}
    best_length = {}
    for u, v, edge_data in graph.edges(data=True):
        length = float(edge_data.get("length", 0.0))
        speed_kmh = _parse_speed_kmh(edge_data.get("maxspeed"))
        speed_m_per_hour = speed_kmh * 1000 if speed_kmh else VELOCITY
        travel_seconds = length / speed_m_per_hour * 3600 if speed_m_per_hour else float("inf")

        key = (_OSM_NODE_INDEX[u], _OSM_NODE_INDEX[v])
        if key not in best_time or travel_seconds < best_time[key]:
            best_time[key] = travel_seconds
        if key not in best_length or length < best_length[key]:
            best_length[key] = length

    _OSM_TIME_MATRIX = _matrix_from_dict(best_time, n)
    _OSM_LENGTH_MATRIX = _matrix_from_dict(best_length, n)

    _OSM_TIME_MATRIX_UNDIRECTED = _matrix_from_dict(_symmetrize(best_time), n)
    _OSM_LENGTH_MATRIX_UNDIRECTED = _matrix_from_dict(_symmetrize(best_length), n)


def _matrix_from_dict(edge_dict, n):
    if not edge_dict:
        return csr_matrix((n, n))
    rows, cols = zip(*edge_dict.keys())
    data = list(edge_dict.values())
    return csr_matrix((data, (rows, cols)), shape=(n, n))


def _symmetrize(edge_dict):
    """
    Add the reverse of every directed edge (keeping the smaller weight if
    both directions already exist), so the resulting matrix behaves like
    graph.to_undirected() for routing purposes.
    """
    sym = dict(edge_dict)
    for (u_idx, v_idx), w in edge_dict.items():
        rev_key = (v_idx, u_idx)
        if rev_key not in sym or w < sym[rev_key]:
            sym[rev_key] = w
    return sym


def _parse_speed_kmh(value):
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if value is None:
        return None
    match = re.search(r"[0-9]+(?:\.[0-9]+)?", str(value))
    if not match:
        return None
    speed = float(match.group())
    if "mph" in str(value).lower():
        speed *= 1.60934
    return speed if speed > 0 else None


def nodes_covered(my_station, my_node_list):
    """
    yields the number of nodes within the influence radius of the station
    """
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    radius_s = s_dict["radius"]
    I_1 = sum([1 if haversine(s_pos, my_node) <= radius_s else 0 for my_node in my_node_list])
    return I_1


def node_coverage(my_plan, my_node):
    """
    yields the number of nodes within the influence radius of the station
    """
    I_1, I_2 = 0, 0
    priv_CS = my_node[1]["private CS"]
    for my_station in my_plan:
        s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
        radius_s = s_dict["radius"]
        distance = haversine(s_pos, my_node)
        if distance <= radius_s:
            I_1 += 1
    for ith in range(I_1):
        I_2 += 1 / (ith + 1)
    single_benefit = I_2 * (1 - 0.1 * priv_CS)
    return single_benefit


def total_number_EVs(my_station, my_node_list):
    """
    yields total number of EVs coming to S in a unit time interval for charging
    """
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    D_s = sum([1 / my_node[1]["distance"] * weak_demand(my_node) if my_node[1]["charging station"] == s_pos[0]
               else 0 for my_node in my_node_list])
    s_dict["D_s"] = D_s  # dimensionless
    return my_station


def service_rate(my_station):
    """
    returns how many cars can be served within one hour
    """
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    s_dict["service rate"] = s_dict["capability"] / BATTERY  # [service rate] = 1/h
    return my_station


def W_s(my_station):
    """
    returns the expected value of waiting time
    """
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    tau_s = 1 / s_dict["service rate"]
    rho_s = s_dict["D_s"] * tau_s * time_unit  # dimensionless (shortened away)
    if rho_s >= 1:
        my_W_s = my_inf
        s_dict["W_s"] = my_W_s
    else:
        my_W_s = rho_s * tau_s / (2 * (1 - rho_s))  # W_s = expected waiting time at S, [W_s] = h
        s_dict["W_s"] = my_W_s
    return my_station


def s_dictionnary(my_station, my_node_list):
    """
    returns the dictionnary for the station
    """
    my_station = installment_fee(my_station)
    my_station = charging_capability(my_station)
    my_station = influence_radius(my_station)
    my_station = total_number_EVs(my_station, my_node_list)
    my_station = service_rate(my_station)
    my_station = W_s(my_station)
    return my_station


# SCORE #####################################################################
def social_benefit(my_plan, my_node_list):
    """
    returns the social benefit of the charging plan (our definition of benefit)
    """
    my_benefit = 0
    for my_node in my_node_list:
        I3 = node_coverage(my_plan, my_node)
        my_benefit += I3
    my_benefit /= len(my_node_list)
    return my_benefit


def travel_cost(my_node_list):
    """ yields the estimated travel time of all vehicles """
    my_cost_travel = sum([my_node[1].get("travel time", my_node[1]["distance"] / VELOCITY) * weak_demand(my_node)
                          for my_node in my_node_list])
    return my_cost_travel


def charging_time(my_plan):
    """
    yields the total charging time given the capability of the CS of the charging plan
    """
    my_charg_time = sum([my_station[2]["D_s"] / my_station[2]["service rate"] for my_station in my_plan])
    return my_charg_time / time_unit


def waiting_time(my_plan):
    """
    returns the average total waiting time of the charging plan
    """
    my_wait_time = sum([my_station[2]["D_s"] * my_station[2]["W_s"] for my_station in my_plan])
    return my_wait_time / time_unit


def social_cost(my_plan, my_node_list):
    """
    returns the social cost, i.e. the negative side of the charging plan
    """
    cost_travel = travel_cost(my_node_list)  # dimensionless
    charg_time = charging_time(my_plan)  # dimensionless
    wait_time = waiting_time(my_plan)  # dimensionless
    cost_boring = charg_time + wait_time  # dimensionless
    my_social_cost = alpha * cost_travel + (1 - alpha) * cost_boring
    return my_social_cost


def existing_score(my_existing_plan, my_node_list):
    """
    computes the score of the existing infrastructure
    """
    my_benefit = social_benefit(my_existing_plan, my_node_list)
    travel_time = travel_cost(my_node_list)  # dimensionless
    charg_time = charging_time(my_existing_plan)  # dimensionless
    wait_time = waiting_time(my_existing_plan)
    cost_boring = charg_time + wait_time  # dimensionless
    my_cost = alpha * travel_time + (1 - alpha) * cost_boring
    return my_benefit, my_cost, charg_time, wait_time, travel_time


def norm_score(my_plan, my_node_list, norm_benefit, norm_charg, norm_wait, norm_travel):
    """
    same as score, but normalised.
    """
    my_score = -my_inf
    if not my_plan:
        return my_score
    benefit = social_benefit(my_plan, my_node_list) / norm_benefit
    cost_travel = travel_cost(my_node_list) / norm_travel  # dimensionless
    charg_time = charging_time(my_plan) / norm_charg  # dimensionless
    wait_time = waiting_time(my_plan) / norm_wait  # dimensionless
    cost = (alpha * cost_travel + (1 - alpha) * (charg_time + wait_time)) / 3
    my_score = my_lambda * benefit - (1 - my_lambda) * cost
    return my_score, benefit, cost, charg_time, wait_time, cost_travel


def score(my_plan, my_node_list):
    """
    returns the final result, i.e., the social score
    """
    my_score = -my_inf
    benefit = 0
    cost = 0
    if not my_plan:
        return my_score, benefit, cost
    benefit = social_benefit(my_plan, my_node_list)  # dimensionless
    cost = social_cost(my_plan, my_node_list)
    my_score = my_lambda * benefit - (1 - my_lambda) * cost
    return my_score, benefit, cost


# Constraints checks ############################################################################
def station_capacity_check(my_plan):
    """
    check if number of stations exceed capacity
    """
    for my_station in my_plan:
        s_x = my_station[1]
        if sum(s_x) > K:
            print("Error: More chargers at the station than admitted: {} chargers".format(sum(s_x)))


def installment_cost_check(my_plan, my_basic_cost):
    """
    check if instalment costs exceed budget
    """
    total_inst_cost = sum([my_station[2]["fee"] for my_station in my_plan]) - my_basic_cost
    if total_inst_cost > BUDGET:
        print("Error: Maximal BUDGET for installation costs exceeded.")


def control_charg_decision(my_plan, my_node_list):
    for my_node in my_node_list:
        station_sum = sum([1 for my_station in my_plan if my_node[1]["charging station"] == my_station[0]])
        if station_sum > 1:
            print("Error: More than one station is assigned to a node.")


def waiting_time_check(my_plan):
    """
    check that wiating time is bounded
    """
    for my_station in my_plan:
        s_dict = my_station[2]
        if s_dict["W_s"] == my_inf:
            print("Error: Waiting time goes to infinity.")


def constraint_check(my_plan, my_node_list, basic_cost):
    """
    test if solution satisfies all constraints
    """
    installment_cost_check(my_plan, basic_cost)
    control_charg_decision(my_plan, my_node_list)
    station_capacity_check(my_plan)
    waiting_time_check(my_plan)


# Parameters ########################################################
alpha = 0.4
my_lambda = 0.5

K = 15  # maximal number of chargers at a station
RADIUS_MAX = 1000  # [radius_max] = m
INSTALL_FEE = np.array([300, 750, 28000])  # fee per installing a charger of type 1, 2 or 3. [fee] = $
CHARGING_POWER = np.array([7, 22, 50])  # [power] = kW, rounded
BATTERY = 85  # battery capacity, [BATTERY] = kWh

BUDGET = 5 * 10 ** 6  # [B] = €
price_parkingplace = 200 * 3.5 * 2  # in €

time_unit = 1  # [time_unit] = h, introduced for getting the units correctly
capacity_unit = 1  # [cap_unit] = kW, introduced for getting the units correctly
VELOCITY = 23 * 1000  # based on m per hour, but here dimensionless

my_inf = 10 ** 6
my_dis_inf = 10 ** 7
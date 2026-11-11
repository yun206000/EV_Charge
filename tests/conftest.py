import os
import pickle
import random
import networkx as nx
import numpy as np
import osmnx as ox
import pytest

"""
Shared fixture: a small synthetic city "Synth" in the same file layout as Graph/<city>/ of the real cities, so the tests
run in seconds and do not depend on the large Tainan / Taipei files.

    5 x 5 road grid near Tainan (about 410 m between neighbours, both directions, maxspeed 50 km/h, one street in mph)
    node 900  connected to the grid by a one-way street only (routing to it needs the undirected fallback)
    node 901  isolated, no road at all (no route exists)
    existing plan: 2 stations with 2 chargers of type 2 each
"""

CITY = "Synth"
GRID = 5
ONE_WAY_NODE, ISOLATED_NODE = 900, 901


def node_id(i, j):
    return 1000 + i * GRID + j


def build_graph():
    graph = nx.MultiDiGraph(crs="epsg:4326")
    for i in range(GRID):
        for j in range(GRID):
            graph.add_node(node_id(i, j), x=120.20 + 0.004 * i, y=23.00 + 0.004 * j)
    for i in range(GRID):
        for j in range(GRID):
            for di, dj in [(1, 0), (0, 1)]:
                if i + di < GRID and j + dj < GRID:
                    a, b = node_id(i, j), node_id(i + di, j + dj)
                    length = 409.0 if di else 444.0
                    speed = "30 mph" if (i, j) == (0, 0) and di else "50"
                    graph.add_edge(a, b, length=length, maxspeed=speed)
                    graph.add_edge(b, a, length=length, maxspeed=speed)
    graph.add_node(ONE_WAY_NODE, x=120.184, y=23.0)
    graph.add_edge(ONE_WAY_NODE, node_id(0, 0), length=1640.0, maxspeed="50")
    graph.add_node(ISOLATED_NODE, x=120.30, y=23.10)
    return graph


def build_node_list():
    nodes = []
    for i in range(GRID):
        for j in range(GRID):
            nodes.append((node_id(i, j), {"x": 120.20 + 0.004 * i, "y": 23.00 + 0.004 * j,
                                          "demand": round(0.1 + 0.05 * ((3 * i + j) % 7), 3),
                                          "estate price": 10.0, "private CS": 0.05}))
    nodes.append((ONE_WAY_NODE, {"x": 120.184, "y": 23.0, "demand": 0.2, "estate price": 10.0, "private CS": 0.05}))
    return nodes


def build_plan(node_list, station_ids, chargers):
    by_id = dict(node_list)
    return [[(sid, dict(by_id[sid])), np.array(config, dtype=float), {}] for sid, config in zip(station_ids, chargers)]


EXISTING_IDS = [node_id(1, 1), node_id(3, 3)]
EXISTING_CHARGERS = [[0, 2, 0], [0, 2, 0]]


@pytest.fixture(scope="session")
def city_root(tmp_path_factory):
    """folder holding Graph/Synth/ (graphml, node list, existing plan) and a result plan in plans/synth_e1/"""
    root = tmp_path_factory.mktemp("city")
    graph_dir = root / "Graph" / CITY
    graph_dir.mkdir(parents=True)
    ox.save_graphml(build_graph(), filepath=str(graph_dir / (CITY + ".graphml")))
    node_list = build_node_list()
    (graph_dir / ("node_list_" + CITY + ".txt")).write_text(repr(node_list))
    with open(graph_dir / ("existingplan_" + CITY + ".pkl"), "wb") as file:
        pickle.dump(build_plan(node_list, EXISTING_IDS, EXISTING_CHARGERS), file)
    plan_dir = root / "plans" / "synth_e1"
    plan_dir.mkdir(parents=True)
    (plan_dir / "nodes_RL.txt").write_text(repr(node_list))
    with open(plan_dir / "plan_RL.pkl", "wb") as file:
        pickle.dump(build_plan(node_list, EXISTING_IDS + [node_id(0, 4)], EXISTING_CHARGERS + [[0, 15, 0]]), file)
    return root


@pytest.fixture
def city_files(city_root):
    graph_dir = city_root / "Graph" / CITY
    return dict(graph=str(graph_dir / (CITY + ".graphml")), nodes=str(graph_dir / ("node_list_" + CITY + ".txt")),
                plan=str(graph_dir / ("existingplan_" + CITY + ".pkl")), result=str(city_root / "plans" / "synth_e1"))


@pytest.fixture
def workdir(tmp_path, city_root, monkeypatch):
    """empty working directory with Graph/ linked in, for the scripts that use paths relative to the repository root"""
    (tmp_path / "Graph").symlink_to(city_root / "Graph")
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def isolated_state():
    """seeded random modules and empty module-level caches, so the tests do not depend on their order"""
    import env_plus as ev
    random.seed(0)
    np.random.seed(0)
    ev.StationPlacement.node_dict.clear()
    ev.StationPlacement.cost_dict.clear()

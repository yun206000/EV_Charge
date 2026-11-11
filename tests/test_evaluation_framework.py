import ast
import numpy as np
import pytest
import evaluation_framework as ef
import test_solution as ts
from conftest import EXISTING_IDS, ISOLATED_NODE, ONE_WAY_NODE, build_node_list, build_plan, node_id


def position(node, x=None, y=None):
    return (node, {"x": x, "y": y})


@pytest.fixture
def prepared(city_files):
    graph, node_list = ef.prepare_graph(city_files["graph"], city_files["nodes"])
    return graph, node_list


def test_haversine_one_hundredth_degree_latitude():
    distance = ef.haversine(position(1, 120.2, 23.0), position(2, 120.2, 23.01))
    assert distance == pytest.approx(1112.2, abs=1)


def test_haversine_same_point_is_not_zero():
    assert ef.haversine(position(1, 120.2, 23.0), position(1, 120.2, 23.0)) == 0.1


@pytest.mark.parametrize("value, expected", [("50", 50.0), (["40", "60"], 40.0), ("30 mph", 30 * 1.60934),
                                             ("50.5 km/h", 50.5), (None, None), ([], None), ("signals", None),
                                             ("0", None)])
def test_parse_speed(value, expected):
    result = ef._parse_speed_kmh(value)
    assert result == (pytest.approx(expected) if expected else expected)


def test_symmetrize_keeps_smaller_weight():
    assert ef._symmetrize({(0, 1): 5.0, (1, 0): 3.0, (1, 2): 2.0}) == {(0, 1): 3.0, (1, 0): 3.0, (1, 2): 2.0,
                                                                         (2, 1): 2.0}


def test_matrix_from_empty_dict():
    assert ef._matrix_from_dict({}, 3).shape == (3, 3)
    assert ef._matrix_from_dict({}, 3).nnz == 0


def test_route_needs_loaded_graph(monkeypatch):
    monkeypatch.setattr(ef, "_OSM_TIME_MATRIX", None)
    with pytest.raises(RuntimeError, match="prepare_graph"):
        ef.osm_travel_time(position(1, 120.2, 23.0), position(2, 120.2, 23.01))


def test_prepare_graph_reads_node_list(prepared, city_files):
    graph, node_list = prepared
    with open(city_files["nodes"]) as file:
        assert node_list == ast.literal_eval(file.readline())
    assert graph.number_of_nodes() == 27


def test_route_between_neighbours(prepared):
    nodes = dict(build_node_list())
    a, b = node_id(1, 1), node_id(2, 1)
    distance, hours = ef._osm_route((a, nodes[a]), (b, nodes[b]))
    assert distance == pytest.approx(409.0)
    assert hours == pytest.approx(409.0 / 50000)
    assert ef.osm_travel_time((a, nodes[a]), (b, nodes[b])) == pytest.approx(hours)


def test_route_uses_lowest_speed_limit_in_mph(prepared):
    nodes = dict(build_node_list())
    a, b = node_id(0, 0), node_id(1, 0)
    _, hours = ef._osm_route((a, nodes[a]), (b, nodes[b]))
    assert hours == pytest.approx(409.0 / (30 * 1.60934 * 1000))


def test_route_against_one_way_street_uses_undirected_graph(prepared):
    nodes = dict(build_node_list())
    start = node_id(0, 0)
    distance, _ = ef._osm_route((start, nodes[start]), (ONE_WAY_NODE, nodes[ONE_WAY_NODE]))
    assert distance == pytest.approx(1640.0)
    # second lookup is served from the cached undirected row
    assert ef._osm_route((start, nodes[start]), (ONE_WAY_NODE, nodes[ONE_WAY_NODE]))[0] == distance


def test_route_to_isolated_node_fails(prepared):
    nodes = dict(build_node_list())
    start = node_id(0, 0)
    with pytest.raises(ValueError, match="No drivable OSM route"):
        ef._osm_route((start, nodes[start]), (ISOLATED_NODE, {"x": 120.30, "y": 23.10}))


def test_coverage_vector_equals_node_coverage(prepared):
    _, node_list = prepared
    node_list, plan = ts.prepare_existing_plan(build_plan(build_node_list(), EXISTING_IDS, [[0, 2, 0], [3, 0, 1]]),
                                               build_node_list())
    vector = ef.coverage_vector(plan, node_list)
    assert vector == pytest.approx([ef.node_coverage(plan, my_node) for my_node in node_list])
    assert ef.nodes_covered(plan[0], node_list) == int(np.sum(
        [ef.haversine(plan[0][0], my_node) <= plan[0][2]["radius"] for my_node in node_list]))
    assert not ef.coverage_vector([], node_list).any()


def test_station_dictionary_fields(prepared):
    node_list, plan = ts.prepare_existing_plan(build_plan(build_node_list(), EXISTING_IDS, [[0, 2, 0], [0, 2, 0]]),
                                               build_node_list())
    station = plan[0]
    assert station[2]["fee"] == pytest.approx(ef.price_parkingplace * 10.0 + 2 * ef.INSTALL_FEE[1])
    assert station[2]["capability"] == 2 * ef.CHARGING_POWER[1]
    assert 0 < station[2]["radius"] < ef.RADIUS_MAX
    assert station[2]["service rate"] == pytest.approx(44 / ef.BATTERY)
    assert all(my_node[1]["charging station"] in EXISTING_IDS for my_node in node_list)


def test_overloaded_station_has_infinite_waiting_time(capsys):
    station = [position(1, 120.2, 23.0), np.array([1.0, 0, 0]), {"service rate": 1.0, "D_s": 2.0}]
    ef.W_s(station)
    assert station[2]["W_s"] == ef.my_inf
    ef.waiting_time_check([station])
    assert "Waiting time goes to infinity" in capsys.readouterr().out


def test_constraint_check_reports_every_violation(capsys):
    node = (1, {"charging station": (1, {}), "distance": 1.0})
    station = [(1, {}), np.array([0, 16, 0]), {"fee": 2 * ef.BUDGET, "W_s": ef.my_inf}]
    ef.constraint_check([station, station], [node], 0)
    out = capsys.readouterr().out
    for message in ["BUDGET", "More than one station", "More chargers", "infinity"]:
        assert message in out


def test_empty_plan_scores():
    assert ef.score([], []) == (-ef.my_inf, 0, 0)
    assert ef.norm_score([], [], 1, 1, 1, 1) == -ef.my_inf


def test_existing_plan_has_normalised_score_one(prepared):
    node_list, plan = ts.prepare_existing_plan(build_plan(build_node_list(), EXISTING_IDS, [[0, 2, 0], [0, 2, 0]]),
                                               build_node_list())
    benefit, cost, charg, wait, travel = ef.existing_score(plan, node_list)
    score, n_benefit, n_cost, n_charg, n_wait, n_travel = ef.norm_score(plan, node_list, benefit, charg, wait, travel)
    assert (n_benefit, n_charg, n_wait, n_travel) == pytest.approx((1, 1, 1, 1))
    assert score == pytest.approx(ef.my_lambda - (1 - ef.my_lambda) * (ef.alpha + (1 - ef.alpha) * 2) / 3)
    raw_score, raw_benefit, raw_cost = ef.score(plan, node_list)
    assert raw_benefit == pytest.approx(benefit)
    assert raw_cost == pytest.approx(ef.social_cost(plan, node_list))
    assert raw_score == pytest.approx(ef.my_lambda * benefit - (1 - ef.my_lambda) * raw_cost)

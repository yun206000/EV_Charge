import csv
import sys
import numpy as np
import pytest
import evaluation_framework as ef
import test_solution as ts
from analysis import plan_metrics as pm
from conftest import CITY, EXISTING_IDS, build_node_list, build_plan


def test_travel_metric_is_demand_weighted():
    nodes = [(1, {"travel time": 0.1, "distance": 1.0, "demand": 0.1, "private CS": 0}),  # 6 min, ceil(10 * 0.1) = 1 time
             (2, {"travel time": 0.2, "distance": 1.0, "demand": 0.25, "private CS": 0})]  # 12 min, 3 times
    assert ts.travel_metric(nodes) == pytest.approx(((6 + 3 * 12) / 4, 12))


def test_travel_metric_without_road_time_uses_velocity():
    nodes = [(1, {"distance": ef.VELOCITY / 2, "demand": 0.1, "private CS": 0})]
    assert ts.travel_metric(nodes) == pytest.approx((30, 30))


def test_waiting_metric_is_weighted_by_station_flow():
    plan = [[None, None, {"D_s": 1.5, "W_s": 0.1}],   # 6 min, counted ceil(1.5) = 2 times
            [None, None, {"D_s": 0.5, "W_s": 0.4}]]   # 24 min, counted once
    assert ts.waiting_metric(plan) == pytest.approx((12, 24))


def test_service_progress_line():
    nodes = [(1, {"travel time": 0.1, "distance": 1.0, "demand": 0.1, "private CS": 0})]
    plan = [[None, None, {"D_s": 1.5, "W_s": 0.1}]]
    assert ts.service_progress(7, nodes, plan) == ("step 7: 1 stations, avg travel time 6.000 min, "
                                                   "avg waiting time 6.000 min")
    assert ts.average_service(nodes, []) == pytest.approx((6, 0))  # no station with EV flow yet
    plan[0][2]["W_s"] = ef.my_inf  # overloaded station
    assert "unbounded" in ts.service_progress(7, nodes, plan)
    plan[0][2]["W_s"] = float("nan")  # station without chargers
    assert "unbounded" in ts.service_progress(7, nodes, plan)


def test_prepare_plan_with_unknown_station_fails():
    plan = [[(12345, {}), np.array([0, 1, 0]), {}]]
    with pytest.raises(ValueError, match="12345"):
        ts.prepare_existing_plan(plan, build_node_list())


def test_prepare_plan_assigns_every_node(city_files):
    ef.prepare_graph(city_files["graph"], city_files["nodes"])
    node_list, plan = ts.prepare_existing_plan(build_plan(build_node_list(), EXISTING_IDS, [[0, 2, 0], [0, 2, 0]]),
                                               build_node_list())
    assert all(my_node[1]["charging station"] in EXISTING_IDS for my_node in node_list)
    assert all(my_node[1]["distance"] >= 0.1 for my_node in node_list)
    assert all(station[2]["D_s"] > 0 and 0 < station[2]["W_s"] < ef.my_inf for station in plan)
    # the station objects are replaced by the node objects of the node list
    assert plan[0][0] is next(n for n in node_list if n[0] == EXISTING_IDS[0])


def test_report_prints_service_metrics(city_files, capsys):
    ef.prepare_graph(city_files["graph"], city_files["nodes"])
    node_list, plan = ts.prepare_existing_plan(build_plan(build_node_list(), EXISTING_IDS, [[0, 2, 0], [0, 2, 0]]),
                                               build_node_list())
    basic_cost = sum(station[2]["fee"] for station in plan)
    norm_benefit, _, norm_charging, norm_waiting, norm_travel = ef.existing_score(plan, node_list)
    norm_score = ts.eci_test(plan, node_list, norm_benefit, norm_charging, norm_waiting, norm_travel)
    ts.perform_test(city_files["nodes"], basic_cost, city_files["result"] + "/plan_RL.pkl", norm_benefit,
                    norm_charging, norm_waiting, norm_travel, norm_score)
    out = capsys.readouterr().out
    assert "Number of charging stations: 3" in out
    assert "Travel time (minutes) - Average" in out
    assert "Waiting time (minutes) - Average" in out


def run_plan_metrics(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["plan_metrics", "--city", CITY] + list(args))
    pm.main()


def test_plan_metrics_writes_one_row_per_plan(workdir, city_root, monkeypatch):
    out_csv = str(workdir / "out" / "metrics.csv")
    run_plan_metrics(monkeypatch, "--existing", "--csv", out_csv)
    run_plan_metrics(monkeypatch, "--plan", str(city_root / "plans" / "synth_e1"), "--label", "synth_e1",
                     "--csv", out_csv)
    with open(out_csv) as file:
        rows = list(csv.DictReader(file))
    assert list(rows[0].keys()) == pm.FIELDS
    assert [row["label"] for row in rows] == ["existing", "synth_e1"]
    assert [row["n_stations"] for row in rows] == ["2", "3"]
    for row in rows:
        assert 0 < float(row["travel_avg_min"]) <= float(row["travel_max_min"])
        assert 0 < float(row["waiting_avg_min"]) <= float(row["waiting_max_min"])
    # a third station with 15 chargers shortens the trips
    assert float(rows[1]["travel_avg_min"]) < float(rows[0]["travel_avg_min"])


def test_load_existing_matches_metrics(workdir):
    node_list, plan = pm.load_existing(CITY)
    row = pm.metrics(plan, node_list)
    assert row["n_stations"] == 2
    assert (row["travel_avg_min"], row["travel_max_min"]) == pytest.approx(ts.travel_metric(node_list))
    assert (row["waiting_avg_min"], row["waiting_max_min"]) == pytest.approx(ts.waiting_metric(plan))

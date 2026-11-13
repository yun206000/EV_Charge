import os
import pickle
import sys
import numpy as np
import pandas as pd
import pytest
from analysis import beam, data_maps, generalization, plan_maps, plan_metrics, training_plots
from conftest import CITY, EXISTING_IDS, build_node_list, build_plan, node_id

"""
The analysis scripts read and write paths relative to the repository root; every test here runs in an empty working
directory (fixture workdir) with small made-up inputs, so the real results and figures are never touched.
"""


def write(path, text):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as file:
        file.write(text)


def save_plan(plan_dir, station_ids, chargers):
    os.makedirs(plan_dir, exist_ok=True)
    nodes = build_node_list()
    with open(os.path.join(plan_dir, "plan_RL.pkl"), "wb") as file:
        pickle.dump(build_plan(nodes, station_ids, chargers), file)
    write(os.path.join(plan_dir, "nodes_RL.txt"), repr(nodes))


# plan_maps ##########################################################################################################
def test_load_plan_counts_chargers(tmp_path):
    plan = [[(7, {}), np.array([0.0, 15.0, 0.0]), {}], [8, np.array([1.0, 2.0, 0.0]), {}]]
    with open(tmp_path / "plan_RL.pkl", "wb") as file:
        pickle.dump(plan, file)
    assert plan_maps.load_plan(str(tmp_path)) == {7: 15, 8: 3}


def test_plan_maps_figure(workdir, city_files, monkeypatch, capsys):
    plans = "Results/Transfer/plans"
    save_plan(os.path.join(plans, "zeroshot_e1"), EXISTING_IDS + [node_id(0, 4)], [[0, 15, 0]] * 3)
    save_plan(os.path.join(plans, "B4k2_e1"), EXISTING_IDS + [node_id(4, 0), node_id(4, 4)], [[0, 15, 0]] * 4)
    columns = "plan,travel_avg_min,waiting_avg_min\n"
    write(os.path.join(plans, "metrics.csv"), columns + "{}/zeroshot_e1,1.5,0.12\n".format(plans))
    write(os.path.join(plans, "metrics_B4k2.csv"), columns + "B4k2_e1,1.2,0.10\n")
    load_graphml = plan_maps.ox.load_graphml
    monkeypatch.setattr(plan_maps.ox, "load_graphml", lambda _: load_graphml(city_files["graph"]))
    monkeypatch.setattr(sys, "argv", ["plan_maps", "--seeds", "1"])
    plan_maps.main()
    assert os.path.getsize(os.path.join(plan_maps.OUT_DIR, "map_seed1.png")) > 0
    assert "3 -> 4 stations, 2 added, 1 removed" in capsys.readouterr().out


# data_maps ##########################################################################################################
def test_data_map_shows_mapping(workdir, tmp_path):
    mapping = tmp_path / "mapping.csv"  # one site next to its node, one 1.2 km away; both on existing stations
    write(str(mapping), "ocm_id,latitude,longitude,nearest_node,nearest_distance_m,n_7kw,n_22kw,n_50kw,numberofpoints\n"
          "1,23.0001,120.2001,{},15.0,0,2,0,2\n2,22.99,120.19,{},1200.0,0,2,0,2\n".format(*EXISTING_IDS))
    info = data_maps.draw_city(CITY, str(workdir / "map.png"), far_m=500, mapping_file=str(mapping))
    assert info["stations"] == 2 and info["ocm_sites"] == 2 and info["chargers"] == 4
    assert info["chargers_by_power"] == {22: 4}
    assert info["far_sites"] == 1 and info["max_distance_m"] == 1200.0
    assert os.path.getsize(workdir / "map.png") > 0
    assert [data_maps.marker_area(n) for n in [1, 3, 4, 7, 8, 30]] == [14, 14, 30, 30, 60, 60]


# training_plots #####################################################################################################
TRAIN_LOG = """Num timesteps: 10800
Best mean reward: 7.10 - Last mean reward per episode: 7.10
Saving new best model to tmp_Tainan/best_model_Tainan_10800
Num timesteps: 11200
Best mean reward: 7.10 - Last mean reward per episode: 7.85
Saving new best model to tmp_Tainan/best_model_Tainan_11200
Num timesteps: 11600
Best mean reward: 7.85 - Last mean reward per episode: 7.40
"""

TAINAN_METRICS = """city,label,plan,n_stations,travel_avg_min,travel_max_min,waiting_avg_min,waiting_max_min
Tainan,existing,,70,10.8,40.1,7.4,30.2
Tainan,original_model,Results/Tainan,500,1.8,9.0,0.4,2.0
Tainan,source_model,Results/Tainan/source_model,480,2.2,9.5,0.1,1.5
"""


def test_read_train_log(tmp_path):
    write(str(tmp_path / "train.log"), TRAIN_LOG)
    log = training_plots.read_train_log(str(tmp_path / "train.log"))
    assert log.steps.tolist() == [10800, 11200, 11600]
    assert log.reward.tolist() == [7.10, 7.85, 7.40]
    assert log.saved.tolist() == [True, True, False]


def test_training_plots(workdir, capsys):
    write("tmp_Tainan/train.log", TRAIN_LOG)
    write("runs/source_Tainan/monitor.csv", '#{"t_start": 0}\nr,l,t\n1.0,600,10\n5.0,650,20\n8.0,640,30\n')
    write("Results/Transfer/source_selection.csv", "train_steps,improvement\n1000,0.5\n17000,8.4\n25000,8.0\n")
    write("Results/Tainan/metrics.csv", TAINAN_METRICS)
    training_plots.main()
    out_dir = "Images/Result_Plots/Training"
    for name in ["original_Tainan_training.png", "source_Tainan_training.png", "Tainan_service_metrics.png"]:
        assert os.path.getsize(os.path.join(out_dir, name)) > 0
    out = capsys.readouterr().out
    assert "original: 3 checks, saved at [10800, 11200]" in out
    assert "source: 3 episodes, mean length 630" in out


def test_tainan_service_keeps_last_row_per_plan(workdir):
    write("Results/Tainan/metrics.csv", TAINAN_METRICS + "Tainan,existing,,70,10.5,40.1,7.0,30.2\n")
    metrics = training_plots.tainan_service(str(workdir / "service.png"))
    assert metrics.loc["existing", "travel_avg_min"] == 10.5
    assert (metrics[["travel_avg_min", "waiting_avg_min"]] < training_plots.TARGET_MIN).all().all()


# generalization #####################################################################################################
def test_generalization(workdir):
    rows = ["label,eval_seed,improvement"] + ["zeroshot,{},{}".format(s, v) for s, v in
                                               zip(range(1, 6), [2.47, 2.31, 2.22, 2.40, 2.35])]
    rows += ["random,{},{}".format(s, v) for s, v in zip(range(1, 4), [1.40, 1.55, 1.48])]
    write(generalization.CSV, "\n".join(rows) + "\n")
    generalization.main()
    summary = open(generalization.SUMMARY).read()
    assert "zeroshot   n = 5  mean 2.3500" in summary
    assert "zero-shot vs random: +59.1%" in summary
    assert os.path.getsize(generalization.OUT_FILE) > 0


# beam ###############################################################################################################
def test_read_progress(tmp_path):
    write(str(tmp_path / "beam.log"), "\n".join([
        "step 25 beam 4 best live 1.2333333333333334 expansions 50 best done -",
        "step 50 beam 4 best live nan expansions 100 best done 2.2333333333333334",
        "step 75 beam 4 best live 2.5333333333333334 expansions 150 best done 2.2333333333333334",
        "finished"]))
    progress = beam.read_progress(str(tmp_path / "beam.log"))
    assert progress.step.tolist() == [25, 50, 75]
    assert progress.improvement.tolist() == pytest.approx([1.0, 2.0, 2.3])


def beam_inputs(seeds):
    plans = beam.PLANS
    zero = [2.4, 2.3, 2.2, 2.4][:len(seeds)]
    after = [2.6, 2.3, 2.5, 2.5][:len(seeds)]
    rows = ["label,eval_seed,improvement,episode_steps,n_stations"]
    rows += ["zeroshot,{},{},600,{}".format(s, z, 300 + s) for s, z in zip(seeds, zero)]
    write(beam.ZEROSHOT_CSV, "\n".join(rows) + "\n")
    rows = ["label,beam,topk,eval_seed,improvement,episode_steps,n_stations,seconds"]
    rows += ["B1k1,1,1,{},{},600,{},60".format(s, z, 300 + s) for s, z in zip(seeds, zero)]
    rows += ["B4k2,4,2,{},{},640,{},600".format(s, a, 310 + s) for s, a in zip(seeds, after)]
    write(beam.CSV, "\n".join(rows) + "\n")
    rows = ["city,label,plan,n_stations,travel_avg_min,travel_max_min,waiting_avg_min,waiting_max_min",
            "Taipei,existing,,29,9.5,30,6.0,20"]
    rows += ["Taipei,zeroshot_e{0},{1}/zeroshot_e{0},300,1.3,8,0.12,1.0".format(s, plans) for s in seeds]
    write(os.path.join(plans, "metrics.csv"), "\n".join(rows) + "\n")
    for s in seeds:
        save_plan(os.path.join(plans, "B4k2_e{}".format(s)), EXISTING_IDS + [node_id(0, s)], [[0, 2, 0]] * 2 +
                  [[0, 15, 0]])
        for name, value in [("B1k1", 2.2), ("B4k2", 2.4)]:
            write("logs/beam/{}_e{}.log".format(name, s), "step 25 beam 4 best live {} expansions 9 best done -\n"
                  .format(value + 0.2333))


@pytest.fixture
def synth_city(monkeypatch):
    """the beam analysis works on Taipei; here it is pointed at the synthetic city"""
    load_existing = plan_metrics.load_existing
    monkeypatch.setattr(plan_metrics, "load_existing", lambda city: load_existing(CITY))


def test_plan_service_metrics(workdir, synth_city):
    beam_inputs([1, 2])
    service = beam.plan_service_metrics(["B4k2"])
    assert service.plan.tolist() == ["B4k2_e1", "B4k2_e2"]
    assert (service.n_stations == 3).all()
    assert ((service.travel_avg_min > 0) & (service.waiting_avg_min > 0)).all()


def test_beam_summary(workdir, synth_city, monkeypatch):
    seeds = [1, 2, 3, 4]
    beam_inputs(seeds)
    monkeypatch.setattr(sys, "argv", ["beam"])
    beam.main()
    summary = open(beam.SUMMARY).read()
    assert "max |difference| of improvement: 0, same episode length: True" in summary
    assert "mean zero-shot 2.3250, mean beam 2.4750, change +0.1500 (+6.5%)" in summary
    assert "better on 3 / 4, equal on 1, worse on 0 episode seeds" in summary
    assert "Wilcoxon signed-rank (one-sided, exact, 3 seeds with a difference)" in summary
    assert "Average travel / waiting time per episode seed (minutes, target < 15)" in summary
    written = pd.read_csv(os.path.join(beam.PLANS, "metrics_B4k2.csv"))
    assert written.plan.tolist() == ["B4k2_e{}".format(s) for s in seeds]
    for name in ["before_after.png", "search_progress.png", "service_metrics.png"]:
        assert os.path.getsize(os.path.join(beam.OUT_DIR, name)) > 0

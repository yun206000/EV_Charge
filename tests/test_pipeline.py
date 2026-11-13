import importlib.util
import os
import pickle
import pytest
import torch
import model_eval
import reinforcement
import test_solution as ts
from conftest import CITY

"""
The original PCRL pipeline on the synthetic city, chosen with --city as on the command line:
reinforcement.py (training) -> model_eval.py (plan) -> test_solution.py (metrics) -> Visualisierung/visualise.py (map).
Everything is written below the test's working directory.
"""

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("visualise", os.path.join(REPO, "Visualisierung", "visualise.py"))
visualise = importlib.util.module_from_spec(spec)
spec.loader.exec_module(visualise)


@pytest.fixture
def trained(workdir):
    """30 training steps (2 episodes of 13 steps), models checked every 10 steps"""
    try:
        reinforcement.main(["--city", CITY, "--steps", "30", "--learning-starts", "10", "--check-freq", "10",
                            "--metrics-freq", "10", "--root", str(workdir)])
    finally:
        torch.use_deterministic_algorithms(False)
    return workdir


def test_training_saves_models_of_the_city(capsys, trained):  # capsys first: it also records the fixture
    out = capsys.readouterr().out
    assert [line.split(":")[0] for line in out.splitlines() if "avg travel time" in line] == ["step 10", "step 20",
                                                                                              "step 30"]
    models = sorted(name for name in os.listdir(trained / ("tmp_" + CITY)) if name.endswith(".zip"))
    assert models and all(name.startswith("best_model_{}_".format(CITY)) for name in models)
    assert os.path.exists(trained / ("tmp_" + CITY) / "monitor.csv")


def test_latest_model_has_most_steps(tmp_path):
    for steps in [20, 100, 30]:
        (tmp_path / "best_model_{}_{}.zip".format(CITY, steps)).write_bytes(b"")
    (tmp_path / "best_model_Other_500.zip").write_bytes(b"")
    assert model_eval.latest_model(str(tmp_path), CITY) == str(tmp_path / "best_model_{}_100.zip".format(CITY))
    with pytest.raises(FileNotFoundError, match="reinforcement.py --city Nowhere"):
        model_eval.latest_model(str(tmp_path), "Nowhere")


def test_plan_metrics_and_map_of_the_city(trained, capsys):
    capsys.readouterr()
    plan, node_list = model_eval.main(["--city", CITY, "--root", str(trained), "--metrics-freq", "5"])
    progress = [line for line in capsys.readouterr().out.splitlines() if "avg travel time" in line]
    assert [line.split(":")[0] for line in progress] == ["step 5", "step 10", "step 13"]  # every 5 steps and the end
    assert progress[-1].startswith("step 13: {} stations".format(len(plan)))
    results = trained / "Results" / CITY
    with open(results / "plan_RL.pkl", "rb") as file:
        assert len(pickle.load(file)) == len(plan) >= 2
    assert (results / "nodes_RL.txt").read_text() == str(node_list)

    ts.main(["--city", CITY])  # default plan folder Results/<city>, relative to the working directory
    out = capsys.readouterr().out
    assert "Number of already existing charging stations: 2" in out
    assert "Number of charging stations: {}".format(len(plan)) in out
    assert out.count("Travel time (minutes) - Average") == 2

    visualise.main(["--city", CITY])
    assert os.path.getsize(trained / "Images" / "Result_Plots" / CITY / ("RL_" + CITY + ".png")) > 0

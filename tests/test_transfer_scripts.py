import csv
import json
import os
import pickle
import socket
import sys
import threading
import numpy as np
import pytest
import torch
from stable_baselines3 import DQN
import gym_compat as gc
from transfer import beam_search as bs
from transfer import env as et
from transfer import evaluate
from transfer import train
from conftest import CITY

"""
Command-line scripts of the transfer experiments on the synthetic city: training (transfer/train.py), evaluation of
models and of the random baseline (transfer/evaluate.py) and the beam search (transfer/beam_search.py), plus the code
of the beam member processes. The scripts find Graph/<city> next to their own folder; here they are pointed at the
synthetic city by their __file__, everything is written below tmp_path.
"""


@pytest.fixture
def at_city(city_root, monkeypatch, workdir):
    """the scripts read Graph/<city> of the synthetic city root, outputs go to the test's working directory"""
    fake = str(city_root / "transfer" / "script.py")
    for module in (train, evaluate, bs):
        monkeypatch.setattr(module, "__file__", fake)
    return workdir


def run(module, monkeypatch, args):
    monkeypatch.setattr(sys, "argv", ["prog"] + args)
    module.main()


@pytest.fixture
def small_model(city_root, tmp_path):
    env = et.make_env(CITY, str(city_root))
    model = DQN("MlpPolicy", env, policy_kwargs=dict(net_arch=[8]), learning_starts=0, seed=0, device="cpu")
    path = str(tmp_path / "small.zip")
    model.save(path)
    return path


def test_train_writes_checkpoints_and_final_model(at_city, monkeypatch, capsys):
    monkeypatch.setenv("PYTHONHASHSEED", "0")  # set by the script, restored after the test
    try:
        run(train, monkeypatch, ["--city", CITY, "--out", "run", "--steps", "20", "--learning-starts", "5",
                                 "--batch-size", "4", "--buffer-size", "50", "--target-update", "5",
                                 "--checkpoint-freq", "10", "--threads", "1"])
    finally:
        torch.use_deterministic_algorithms(False)
    out = at_city / "run"
    assert json.loads((out / "config.json").read_text())["steps"] == 20
    assert sorted(os.listdir(out / "checkpoints")) == ["model_10_steps.zip", "model_20_steps.zip"]
    assert (out / "final_model.zip").exists() and (out / "monitor.csv").exists()
    assert "Saved final model" in capsys.readouterr().out


def test_train_steps_of(tmp_path):
    assert evaluate.train_steps_of("runs/x/checkpoints/model_1500_steps.zip") == 1500
    assert evaluate.train_steps_of(str(tmp_path / "final_model.zip")) == 0  # no config.json
    (tmp_path / "config.json").write_text(json.dumps(dict(steps=777)))
    assert evaluate.train_steps_of(str(tmp_path / "final_model.zip")) == 777
    assert evaluate.train_steps_of(str(tmp_path / "other.zip")) == 0


def read_rows(path):
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def test_evaluate_model_folder_and_plan(at_city, monkeypatch, small_model, tmp_path, capsys):
    folder = tmp_path / "models"
    folder.mkdir()
    for name in ["model_20_steps.zip", "model_10_steps.zip"]:
        (folder / name).write_bytes(open(small_model, "rb").read())
    run(evaluate, monkeypatch, ["--city", CITY, "--policy", "model", "--model", str(folder), "--label", "G1",
                                "--eval-seed", "2", "--csv", "res/results.csv", "--save-plan", "plan"])
    rows = read_rows(at_city / "res" / "results.csv")
    assert [r["train_steps"] for r in rows] == ["10", "20"]  # sorted by training steps
    assert all(r["label"] == "G1" and r["eval_seed"] == "2" and r["policy"] == "model" for r in rows)
    first = rows[0]
    assert float(first["improvement"]) == pytest.approx(float(first["score"]) - float(first["existing_score"]))
    with open(at_city / "plan" / "plan_RL.pkl", "rb") as file:
        assert len(pickle.load(file)) == int(rows[-1]["n_stations"])
    assert (at_city / "plan" / "nodes_RL.txt").read_text().startswith("[(")
    # one more model appends to the same file without a second header
    run(evaluate, monkeypatch, ["--city", CITY, "--model", small_model, "--label", "G2", "--csv", "res/results.csv"])
    assert [r["label"] for r in read_rows(at_city / "res" / "results.csv")] == ["G1", "G1", "G2"]


def test_evaluate_random_policy_is_reproducible(at_city, monkeypatch):
    for _ in range(2):
        run(evaluate, monkeypatch, ["--city", CITY, "--policy", "random", "--label", "random", "--seed", "3",
                                    "--csv", "random.csv"])
    first, second = read_rows(at_city / "random.csv")
    assert first["model"] == "" and first["train_steps"] == "0"
    assert first["score"] == second["score"] and first["episode_steps"] == second["episode_steps"]


def test_evaluate_rejects_non_finite_weights(at_city, monkeypatch, city_root, tmp_path):
    env = et.make_env(CITY, str(city_root))
    model = DQN("MlpPolicy", env, policy_kwargs=dict(net_arch=[8]), learning_starts=0, seed=0, device="cpu")
    with torch.no_grad():
        next(model.policy.parameters()).fill_(float("nan"))
    model.save(str(tmp_path / "nan.zip"))
    with pytest.raises(ValueError, match="non-finite weights"):
        run(evaluate, monkeypatch, ["--city", CITY, "--model", str(tmp_path / "nan.zip"), "--label", "x",
                                    "--csv", "x.csv"])


def test_beam_search_greedy_equals_evaluate(at_city, monkeypatch, small_model, capsys):
    """B = k = 1 follows the greedy episode of transfer/evaluate.py exactly (same score, stations and steps)"""
    run(evaluate, monkeypatch, ["--city", CITY, "--model", small_model, "--label", "zs", "--eval-seed", "1",
                                "--csv", "zs.csv"])
    run(bs, monkeypatch, ["--city", CITY, "--model", small_model, "--beam", "1", "--topk", "1", "--eval-seed", "1",
                          "--csv", "beam.csv", "--log-every", "0", "--max-mem-gb", "0", "--min-available-gb", "0"])
    greedy, = read_rows(at_city / "zs.csv")
    beam, = read_rows(at_city / "beam.csv")
    assert float(beam["score"]) == pytest.approx(float(greedy["score"]), abs=1e-12)
    assert beam["n_stations"] == greedy["n_stations"] and beam["episode_steps"] == greedy["episode_steps"]
    assert sum(map(int, beam["action_counts"].split())) == int(beam["episode_steps"])


def test_beam_search_wider_beam_saves_plan(at_city, monkeypatch, small_model, capsys):
    run(bs, monkeypatch, ["--city", CITY, "--model", small_model, "--beam", "3", "--topk", "2", "--eval-seed", "1",
                          "--csv", "out/beam.csv", "--save-plan", "plan", "--log-every", "2", "--patience", "3",
                          "--max-mem-gb", "0", "--min-available-gb", "0"])
    out = capsys.readouterr().out
    assert out.startswith("start: score") and "existing plan, step 0" in out
    assert "current plan of the best member" in out and "peak memory" in out and "saved plan plan" in out
    row, = read_rows(at_city / "out" / "beam.csv")
    assert row["beam"] == "3" and int(row["expansions"]) >= int(row["episode_steps"])
    with open(at_city / "plan" / "plan_RL.pkl", "rb") as file:
        assert len(pickle.load(file)) == int(row["n_stations"])
    assert float(row["improvement"]) >= 0


def test_beam_search_max_steps(at_city, monkeypatch, small_model, capsys):
    run(bs, monkeypatch, ["--city", CITY, "--model", small_model, "--beam", "2", "--topk", "2", "--max-steps", "2",
                          "--csv", "b.csv", "--log-every", "1", "--max-mem-gb", "0", "--min-available-gb", "0"])
    row, = read_rows(at_city / "b.csv")
    assert row["episode_steps"] == "2"
    assert "step 2 beam" in capsys.readouterr().out


# ------------------------------------------------------------------ beam member processes and memory guard


class Exited(Exception):
    pass


@pytest.fixture
def member(city_root, tmp_path, monkeypatch, small_model):
    """a Worker on the synthetic city whose fork / os._exit are replaced, so it runs inside the test process"""
    graph_dir = os.path.join(str(city_root), "Graph", CITY)
    env = bs.ScoredStationPlacement(os.path.join(graph_dir, CITY + ".graphml"),
                                    os.path.join(graph_dir, "node_list_" + CITY + ".txt"),
                                    os.path.join(graph_dir, "existingplan_" + CITY + ".pkl"))
    obs = gc.reset(env)
    path = str(tmp_path / "c.sock")
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(path)
    listener.listen(8)
    listener.settimeout(30)

    def exit_(code):
        raise Exited(code)
    monkeypatch.setattr(bs.os, "_exit", exit_)
    yield bs.Worker(env, gc.load_dqn(small_model, env), path), listener, obs
    listener.close()


def test_worker_reports_and_serves_commands(member, monkeypatch, tmp_path):
    worker, listener, obs = member
    coordinator = bs.Coordinator(listener)
    worker.report(0, None, None, obs, False).close()
    first, = coordinator.collect(1)
    assert first["id"] == 0 and first["q"].shape == (5,) and not first["done"]
    monkeypatch.setattr(bs, "fork_keep_random", lambda: 0)  # always continue as the child
    left, right = socket.socketpair()
    results = {}

    def coordinate():
        bs.send(right, ("expand", [(1, 7)]))
        child, = coordinator.collect(1)
        results["child"] = child
        results["metrics"] = coordinator.service(7, 3)
        coordinator.command(7, "save", str(tmp_path / "saved"))
        results["saved"] = bs.recv(coordinator.conns[7])
        coordinator.kill(7)
    thread = threading.Thread(target=coordinate)
    thread.start()
    with pytest.raises(Exited):
        worker.serve(left, 0)
    thread.join()
    assert results["child"]["parent"] == 0 and results["child"]["action"] == 1 and results["child"]["error"] is None
    assert results["metrics"].startswith("step 3: ") and results["saved"] == "saved"
    assert (tmp_path / "saved" / "plan_RL.pkl").exists() and (tmp_path / "saved" / "nodes_RL.txt").exists()


def test_worker_reports_failed_step_and_parent_leaves(member, monkeypatch):
    worker, listener, _ = member
    coordinator = bs.Coordinator(listener)
    monkeypatch.setattr(bs, "fork_keep_random", lambda: 0)
    monkeypatch.setattr(bs.gc, "step", lambda env, action: (_ for _ in ()).throw(RuntimeError("broken step")))
    left, right = socket.socketpair()
    bs.send(right, ("expand", [(2, 5)]))
    reports = []
    thread = threading.Thread(target=lambda: (reports.extend(coordinator.collect(1)), coordinator.kill(5)))
    thread.start()
    with pytest.raises(Exited):
        worker.serve(left, 0)
    thread.join()
    assert reports[0]["done"] and "broken step" in reports[0]["error"] and reports[0]["q"] is None
    # a parent whose expansions all live on in children leaves
    monkeypatch.setattr(bs, "fork_keep_random", lambda: 12345)
    left, right = socket.socketpair()
    bs.send(right, ("expand", [(0, 8), (1, 9)]))
    with pytest.raises(Exited):
        worker.serve(left, 0)


def test_coordinator_expand_and_kill_closed_member():
    sent = {}

    class Conn:
        def __init__(self, name):
            self.name = name

        def sendall(self, data):
            if self.name == "closed":
                raise OSError("closed")
            sent[self.name] = data

        def close(self):
            pass
    coordinator = bs.Coordinator(None)
    coordinator.conns = {0: Conn("a"), 1: Conn("b"), 2: Conn("closed")}
    beam = [dict(id=0, q=np.array([0.1, 0.5, 0.3, 0.0, 0.2])), dict(id=1, q=np.array([1.0, 0.0, 0.0, 0.0, 0.0]))]
    n, backbone = coordinator.expand(beam, 2, backbone=0)
    assert n == 4 and backbone == 1 and coordinator.next_id == 5 and set(coordinator.conns) == {2}
    n, backbone = coordinator.expand([], 2, backbone=None)
    assert (n, backbone) == (0, None)
    coordinator.kill(2)  # the member is already gone: no error
    assert coordinator.conns == {}


def test_memory_guard_kills_the_beam(monkeypatch, tmp_path, capsys):
    socket_dir = tmp_path / "beam"
    socket_dir.mkdir()
    killed = []
    monkeypatch.setattr(bs.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(bs.os, "killpg", lambda pgid, sig: killed.append(pgid))

    def exit_(code):
        raise Exited(code)
    monkeypatch.setattr(bs.os, "_exit", exit_)
    monkeypatch.setattr(bs, "mem_available_gb", lambda: 1.0)
    monkeypatch.setattr(bs, "group_pss_gb", lambda pgid, extra=(): 5.0)
    guard = bs.MemoryGuard(4242, max_gb=0, min_available_gb=2, socket_path=str(socket_dir / "c.sock"))
    with pytest.raises(Exited, match="3"):
        guard.run()
    assert killed == [4242] and not socket_dir.exists() and guard.peak_gb == 5.0
    assert "machine has only 1.0 GB available" in capsys.readouterr().out

    def gone(pgid, sig):
        raise ProcessLookupError
    monkeypatch.setattr(bs.os, "killpg", gone)
    monkeypatch.setattr(bs, "mem_available_gb", lambda: 100.0)
    guard = bs.MemoryGuard(4242, max_gb=4, min_available_gb=2, socket_path=str(socket_dir / "c.sock"))
    with pytest.raises(Exited):
        guard.run()
    assert "beam processes use 5.0 GB (--max-mem-gb 4)" in capsys.readouterr().out


def test_mem_available_without_meminfo_line(monkeypatch, tmp_path):
    fake = tmp_path / "meminfo"
    fake.write_text("MemTotal: 100 kB\n")
    real_open = open
    monkeypatch.setattr("builtins.open", lambda path, *a, **k: real_open(fake if path == "/proc/meminfo" else path,
                                                                         *a, **k))
    assert bs.mem_available_gb() == float("inf")

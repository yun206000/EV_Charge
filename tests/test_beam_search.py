import os
import random
import socket
import threading
import numpy as np
import gym_compat as gc
import pytest
from stable_baselines3 import DQN
from transfer import beam_search as bs
from conftest import CITY


def test_send_recv_large_message():
    left, right = socket.socketpair()
    payload = dict(q=np.arange(50000, dtype=float), text="x" * 100000)
    sender = threading.Thread(target=lambda: (bs.send(left, payload), left.close()))
    sender.start()  # the message is larger than the socket buffer, so send and receive must run at the same time
    received = bs.recv(right)
    sender.join()
    assert np.array_equal(received["q"], payload["q"]) and received["text"] == payload["text"]
    with pytest.raises(EOFError):
        bs.recv(right)
    right.close()


def test_plan_signature_ignores_station_order():
    a = [(1, {}), [0, 2, 0]]
    b = [(2, {}), [1, 0, 0]]
    assert bs.plan_signature([a, b]) == bs.plan_signature([b, a])
    assert bs.plan_signature([a, b]) != bs.plan_signature([a, [(2, {}), [2, 0, 0]]])


def test_forked_child_keeps_random_state():
    random.seed(42)
    read_end, write_end = os.pipe()
    pid = bs.fork_keep_random()
    if pid == 0:  # child: report its next random number and leave without running pytest teardown
        os.write(write_end, repr(random.random()).encode())
        os._exit(0)
    os.close(write_end)
    child_value = float(os.read(read_end, 100).decode())
    os.close(read_end)
    os.waitpid(pid, 0)
    assert child_value == random.random()


def test_memory_readings():
    assert bs.mem_available_gb() > 0
    own = bs.group_pss_gb(os.getpgid(0), (os.getpid(),))
    assert own > 0
    assert bs.group_pss_gb(-12345) == 0


def test_reap_without_children():
    bs.reap()


@pytest.fixture
def scored_env(city_root):
    graph_dir = os.path.join(str(city_root), "Graph", CITY)
    return bs.ScoredStationPlacement(os.path.join(graph_dir, CITY + ".graphml"),
                                     os.path.join(graph_dir, "node_list_" + CITY + ".txt"),
                                     os.path.join(graph_dir, "existingplan_" + CITY + ".pkl"))


def test_scored_env_remembers_last_score(scored_env):
    gc.reset(scored_env)
    _, reward, _, _ = gc.step(scored_env, 1)
    assert scored_env.last_score <= scored_env.best_score
    if reward > 0:
        assert scored_env.last_score == pytest.approx(scored_env.best_score)


def test_q_values_of_model(scored_env):
    model = DQN("MlpPolicy", scored_env, policy_kwargs=dict(net_arch=[8]), learning_starts=0, seed=0,
                device="cpu")  # the project runs on the CPU (transfer/beam_search.py loads models there)
    obs = gc.reset(scored_env)
    values = bs.q_values(model, obs)
    assert values.shape == (5,) and values.dtype == float
    assert int(np.argmax(values)) == int(model.predict(obs, deterministic=True)[0])

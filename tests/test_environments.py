import random
import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env
import evaluation_framework as ef
import env_plus as ev
from transfer import env as et
from conftest import CITY, EXISTING_IDS, build_node_list, node_id


def capacity(config):
    return np.sum(ef.CHARGING_POWER * np.asarray(config))


def cost(config):
    return np.sum(ef.INSTALL_FEE * np.asarray(config))


def test_prepare_config_cheapest_configurations():
    config_dict = ev.prepare_config()
    keys = sorted(config_dict)
    assert keys[0] == 0
    for key in keys:
        assert sum(config_dict[key]) <= ef.K
        assert capacity(config_dict[key]) >= key
    costs = [cost(config_dict[key]) for key in keys]
    assert costs == sorted(costs)


def test_initial_solution_covers_nearby_demand():
    config_dict = ev.prepare_config()
    nodes = build_node_list()
    config = ev.initial_solution(config_dict, nodes, nodes[0])
    # only the station node itself lies within 50 m: ceil(weakened demand) * BATTERY
    assert capacity(config) > np.ceil(ef.weak_demand(nodes[0])) * ef.BATTERY


def test_choose_nodes():
    nodes = [(1, {"covered": 2.0, "demand": 0.1, "private CS": 0}), (2, {"covered": 0.5, "demand": 0.3, "private CS": 0}),
             (3, {"covered": 0.5, "demand": 0.9, "private CS": 1})]
    assert ev.choose_node_new_benefit(nodes)[0] == 2
    assert ev.choose_node_bydemand(nodes)[0] == 3


def test_anti_choose_and_support_without_stations():
    nodes = build_node_list()
    assert ev.anti_choose_node_bybenefit(nodes, []) is None
    best = max(nodes, key=ef.weak_demand)
    assert ev.support_stations([], nodes)[0] == best[0]


def test_anti_choose_removes_station_with_lowest_bound():
    nodes = build_node_list()
    by_id = dict(nodes)
    cheap, dear = node_id(0, 0), node_id(2, 2)
    by_id[dear]["estate price"] = 1000.0
    plan = [[(cheap, by_id[cheap]), np.array([0, 1, 0]), {}], [(dear, by_id[dear]), np.array([0, 1, 0]), {}]]
    assert ev.anti_choose_node_bybenefit(nodes, plan) is plan[1]


def test_support_full_station_picks_nearest_free_node():
    nodes = build_node_list()
    by_id = dict(nodes)
    full = (node_id(2, 2), by_id[node_id(2, 2)])
    plan = [[full, np.array([0, ef.K, 0]), {"D_s": 1.0, "service rate": 2.0, "W_s": 0.1}]]
    free = [n for n in nodes if n[0] != full[0]]
    chosen = ev.support_stations(plan, free)
    assert chosen[0] in {node_id(1, 2), node_id(3, 2), node_id(2, 1), node_id(2, 3)}
    plan[0][1] = np.array([0, 3, 0])
    assert ev.support_stations(plan, free) is full


@pytest.fixture
def env(city_files):
    return ev.StationPlacement(city_files["graph"], city_files["nodes"], city_files["plan"])


def test_station_placement_passes_gym_checks(env):
    check_env(env, warn=False)


def test_station_placement_episode(env):
    obs = env.reset()
    assert obs.shape == env.observation_space.shape
    assert env.plan_length == 2
    best = env.best_score
    actions = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 4, 0, 1, 2, 3]
    done, steps = False, 0
    while not done:
        obs, reward, done, _ = env.step(actions[steps])
        steps += 1
        assert reward >= 0
        assert env.best_score == pytest.approx(best + reward)
        best = env.best_score
        assert all(sum(station[1]) <= ef.K for station in env.plan_instance.plan)
    assert steps == np.ceil(len(env.node_list) / 2)
    node_list, best_plan = env.render()
    assert best_plan and node_list
    # the existing stations are never removed
    assert set(EXISTING_IDS) <= {station[0][0] for station in env.plan_instance.plan}


def test_station_placement_budget_end(env):
    env.reset()
    env.budget = 1.0
    env.step(0)
    assert env.game_over
    env.reset()
    env.budget = 1.0
    env.step(2)
    assert env.game_over


def test_steal_column(env):
    env.reset()
    plan = env.plan_instance
    station = plan.plan[0]
    station[1] = np.array([1.0, 0.0, 1.0])
    ef.installment_fee(station)
    budget, index = plan.steal_column(station, 0)
    assert index == 2 and list(station[1]) == [1, 0, 0] and station in plan.plan
    assert budget == pytest.approx(ef.INSTALL_FEE[2])
    budget, index = plan.steal_column(station, 0)
    assert index == 0 and station not in plan.plan
    assert "charging plan" in repr(plan)
    assert "station" in repr(ev.Station())


def test_transfer_helpers():
    assert et.squash(np.array([0.0, 1.0, 3.0])) == pytest.approx([0, 0.5, 0.75])
    assert et.log_scale([0, et.NEED_CAP, 10 * et.NEED_CAP]) == pytest.approx([0, 1, 1])
    assert et.station_need([None, None, {"service rate": 0}]) == et.NEED_CAP
    assert et.station_need([None, None, {"service rate": 1.0, "D_s": 1.0, "W_s": np.inf}]) == et.NEED_CAP
    assert et.station_need([None, None, {"service rate": 2.0, "D_s": 1.0, "W_s": 0.5}]) == pytest.approx(1.0)


@pytest.fixture
def transfer_env(city_root):
    return et.make_env(CITY, str(city_root))


def test_transfer_observation(transfer_env):
    obs = transfer_env.reset()
    assert obs.shape == (et.OBS_SIZE,)
    assert transfer_env.observation_space.contains(obs)
    assert obs[0] == pytest.approx(1.0)  # whole budget left


def test_transfer_episode_is_reproducible(transfer_env):
    def episode():
        random.seed(3)
        obs, done, scores = transfer_env.reset(), False, []
        while not done:
            obs, _, done, _ = transfer_env.step(len(scores) % 5)
            assert np.all(np.isfinite(obs)) and np.all(np.abs(obs) <= 1)
            scores.append(transfer_env.best_score)
        return obs, scores
    first_obs, first = episode()
    second_obs, second = episode()
    assert first == second
    assert np.array_equal(first_obs, second_obs)


def test_transfer_rejects_non_finite_observation(transfer_env):
    transfer_env.reset()
    transfer_env.budget = np.nan
    with pytest.raises(ValueError, match="Non-finite observation"):
        transfer_env.establish_observation()

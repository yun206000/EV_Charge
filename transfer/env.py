import numpy as np
from gym import spaces
import evaluation_framework as ef
import env_plus as ev

"""
City-independent variant of the StationPlacement environment for transfer learning.

The original observation lists 13 values for every node, so its size depends on the city and it contains raw GPS
coordinates. Here the observation is a fixed-size summary in which every value is normalised per city:
coordinates relative to the city bounding box, counts as fractions, waiting/charging time on a log scale.
Actions, rewards and episode logic are unchanged, so the score stays comparable with env_plus.
"""

N_TOP_FREE = 32  # free nodes with the highest weakened demand
N_TOP_STATIONS = 16  # stations with the highest waiting + charging time
N_COVER_BINS = 8
N_GLOBAL = 16
N_CANDIDATE = 7  # features per action candidate
N_FREE_FEAT = 5
N_STATION_FEAT = 7
OBS_SIZE = N_GLOBAL + 4 * N_CANDIDATE + N_TOP_FREE * N_FREE_FEAT + N_TOP_STATIONS * N_STATION_FEAT + N_COVER_BINS
STEP_SCALE = 1000  # episodes end on budget after roughly 500-700 steps
SCORE_SCALE = 10
NEED_CAP = 10 ** 6


def squash(values):
    """map [0, inf) to [0, 1)"""
    return values / (1 + values)


def log_scale(values):
    values = np.minimum(np.asarray(values, dtype=float), NEED_CAP)
    return np.log1p(values) / np.log1p(NEED_CAP)


def station_need(my_station):
    """
    waiting + charging time; a station without chargers (service rate 0) gives inf or nan, count it as maximal need
    """
    if my_station[2]["service rate"] <= 0:
        return NEED_CAP
    need = ev.support_stations_hilfe(my_station)
    return need if np.isfinite(need) else NEED_CAP


class TransferStationPlacement(ev.StationPlacement):
    """StationPlacement with a fixed-size, city-normalised observation"""

    def __init__(self, my_graph_file, my_node_file, my_plan_file):
        super(TransferStationPlacement, self).__init__(my_graph_file, my_node_file, my_plan_file)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(OBS_SIZE,), dtype=np.float32)
        # static node features, in node_list order
        self.node_index = {my_node[0]: j for j, my_node in enumerate(self.node_list)}
        x = np.array([my_node[1]['x'] for my_node in self.node_list], dtype=float)
        y = np.array([my_node[1]['y'] for my_node in self.node_list], dtype=float)
        self.x_norm = 2 * (x - x.min()) / max(x.max() - x.min(), 1e-9) - 1
        self.y_norm = 2 * (y - y.min()) / max(y.max() - y.min(), 1e-9) - 1
        self.weak_demand = np.array([ef.weak_demand(my_node) for my_node in self.node_list], dtype=float)
        self.price = np.array([my_node[1]['estate price'] for my_node in self.node_list], dtype=float)
        self.total_demand = max(self.weak_demand.sum(), 1e-9)
        self.initial_node_attributes = [dict(my_node[1]) for my_node in self.node_list]

    def reset(self):
        """
        Start every episode with empty distance/cost caches. The caches of env_plus/evaluation_framework are never
        emptied, so in a long training run they grow to several GB per process. The cost cache is also keyed by
        station position only, so without this an episode depends on the earlier episodes of the same process;
        now each episode behaves like the first episode of a fresh process (the setting of transfer/evaluate.py).
        The node attributes (assigned station, distance, travel time, ...) are restored for the same reason.
        """
        for my_node, attributes in zip(self.node_list, self.initial_node_attributes):
            my_node[1].clear()
            my_node[1].update(attributes)
        for my_dict in (ev.StationPlacement.node_dict, ev.StationPlacement.cost_dict):
            for node_cache in my_dict.values():
                node_cache.clear()
        ef._OSM_ROUTE_CACHE.clear()
        return super(TransferStationPlacement, self).reset()

    def node_features(self, j, covered):
        return [self.weak_demand[j], squash(covered[j]), self.price[j], self.x_norm[j], self.y_norm[j]]

    def establish_observation(self):
        """
        Build the fixed-size observation
        """
        plan = self.plan_instance.plan
        existing = set(my_node[0] for my_node in self.plan_instance.existing_plan)
        covered = np.array([my_node[1].get("covered", 0) for my_node in self.node_list], dtype=float)
        station_idx = np.array([self.node_index[s[0][0]] for s in plan], dtype=int)
        chargers = np.array([s[1] for s in plan], dtype=float).reshape(-1, len(ef.CHARGING_POWER))
        n_chargers = chargers.sum(axis=1)
        is_station = np.zeros(len(self.node_list), dtype=bool)
        is_station[station_idx] = True
        not_full = np.zeros(len(self.node_list), dtype=bool)
        not_full[station_idx[n_chargers < ef.K]] = True
        neediness = np.array([station_need(s) for s in plan], dtype=float)
        overloaded = np.array([s[2]["W_s"] >= ef.my_inf for s in plan], dtype=bool)
        cover_squash = squash(covered)

        # global state
        n_stations = len(plan)
        total_chargers = max(n_chargers.sum(), 1)
        global_feat = np.zeros(N_GLOBAL)
        global_feat[0] = self.budget / ef.BUDGET
        global_feat[1] = min(self.schritt / STEP_SCALE, 1)
        global_feat[2] = np.log1p(n_stations) / np.log1p(STEP_SCALE)
        if n_stations:
            global_feat[3] = np.mean([s[0][0] not in existing for s in plan])
            global_feat[4:7] = chargers.sum(axis=0) / total_chargers
            global_feat[7] = n_chargers.mean() / ef.K
            global_feat[8] = np.mean(n_chargers >= ef.K)
            global_feat[13] = log_scale(neediness).mean()
            global_feat[14] = log_scale(neediness).max()
            global_feat[15] = overloaded.mean()
        global_feat[9] = np.sum(self.weak_demand * cover_squash) / self.total_demand
        global_feat[10] = np.mean(covered == 0)
        global_feat[11] = np.sum(self.weak_demand[covered == 0]) / self.total_demand
        global_feat[12] = np.tanh(self.best_score / SCORE_SCALE)

        # the node each heuristic action would pick (same tie-breaking as env_plus: first in node_list order)
        candidate_feat = np.zeros((4, N_CANDIDATE))
        station_chargers = np.zeros(len(self.node_list))
        station_chargers[station_idx] = n_chargers
        for row, (mask, values, pick_max) in enumerate([(~is_station, covered, False),
                                                        (~is_station, self.weak_demand, True),
                                                        (not_full, covered, False),
                                                        (not_full, self.weak_demand, True)]):
            if not mask.any():
                continue
            masked = np.where(mask, values, -np.inf if pick_max else np.inf)
            j = int(np.argmax(masked) if pick_max else np.argmin(masked))
            candidate_feat[row] = self.node_features(j, covered) + [station_chargers[j] / ef.K, 1]

        # free nodes with the highest weakened demand
        free_feat = np.zeros((N_TOP_FREE, N_FREE_FEAT))
        free_idx = np.flatnonzero(~is_station)
        top_free = free_idx[np.argsort(-self.weak_demand[free_idx], kind="stable")[:N_TOP_FREE]]
        for row, j in enumerate(top_free):
            free_feat[row] = self.node_features(j, covered)

        # stations that need support the most
        station_feat = np.zeros((N_TOP_STATIONS, N_STATION_FEAT))
        for row, k in enumerate(np.argsort(-neediness, kind="stable")[:N_TOP_STATIONS]):
            j = station_idx[k]
            station_feat[row] = [log_scale(neediness[k]), *(chargers[k] / ef.K), self.x_norm[j], self.y_norm[j],
                                 plan[k][0][0] in existing]

        # demand-weighted coverage histogram
        hist, _ = np.histogram(cover_squash, bins=N_COVER_BINS, range=(0, 1), weights=self.weak_demand)
        hist = hist / self.total_demand

        obs = np.concatenate([global_feat, candidate_feat.ravel(), free_feat.ravel(), station_feat.ravel(), hist])
        if not np.all(np.isfinite(obs)):
            # a nan in the observation silently turns the Q-network weights into nan
            raise ValueError("Non-finite observation at indices {}".format(np.flatnonzero(~np.isfinite(obs))))
        obs = np.clip(obs, -1, 1).astype(self.observation_space.dtype)
        return obs


def make_env(location, repository_root):
    graph_dir = repository_root + "/Graph/" + location
    graph_file = graph_dir + "/" + location + ".graphml"
    node_file = graph_dir + "/node_list_" + location + ".txt"
    plan_file = graph_dir + "/existingplan_" + location + ".pkl"
    return TransferStationPlacement(graph_file, node_file, plan_file)

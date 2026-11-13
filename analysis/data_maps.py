import argparse
import os
import pickle
import numpy as np
import pandas as pd
import osmnx as ox
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import evaluation_framework as ef

"""
Road network and existing charging stations of a city after the Open Charge Map mapping (Preprocessing), written to
Images/Result_Plots/Data/<city>_network_stations.png:

    light grey lines  road network (Graph/<city>/<city>.graphml)
    blue dots         existing stations of the plan (Graph/<city>/existingplan_<city>.pkl), placed on their road node;
                      the dot size gives the number of chargers (OCM sites on the same node are merged)
    hollow circles    OCM coordinates more than --far-m metres from their road node, joined to it by a dashed line
                      (sites outside the area of the road network are mapped to the nearest node at its edge)

    python -m analysis.data_maps                       # Tainan and Taipei
    python -m analysis.data_maps --cities Taipei_Daan
"""

OUT_DIR = "Images/Result_Plots/Data"
STATION, LINK = "#2a78d6", "#8a8984"
INK, MUTED, ROAD, SURFACE = "#0b0b0b", "#52514e", "#d6d5d0", "#fcfcfb"
SIZE_BINS = [(1, 3, 14), (4, 7, 30), (8, None, 60)]  # chargers from, to (None = open), marker area


def marker_area(chargers):
    for low, high, area in SIZE_BINS:
        if chargers >= low and (high is None or chargers <= high):
            return area
    return SIZE_BINS[0][2]


def load_city(city, mapping_file=None):
    """road graph, existing plan as {node: chargers per type} and the OCM -> road node mapping"""
    graph_dir = os.path.join("Graph", city)
    graph = ox.load_graphml(os.path.join(graph_dir, city + ".graphml"))
    with open(os.path.join(graph_dir, "existingplan_" + city + ".pkl"), "rb") as file:
        plan = pickle.load(file)
    stations = {}
    for station in plan:
        node = station[0][0] if isinstance(station[0], (tuple, list)) else station[0]
        stations[node] = np.asarray(station[1], dtype=int)
    mapping = pd.read_csv(mapping_file or os.path.join(graph_dir, "ocm_osm_mapping_" + city + ".csv"))
    return graph, stations, mapping


def summary(graph, stations, mapping, far_m):
    chargers = np.sum(list(stations.values()), axis=0) if stations else np.zeros(len(ef.CHARGING_POWER), dtype=int)
    far = mapping[mapping.nearest_distance_m > far_m]
    return dict(nodes=len(graph.nodes), edges=len(graph.edges), ocm_sites=len(mapping), stations=len(stations),
                chargers=int(chargers.sum()),
                chargers_by_power={int(p): int(n) for p, n in zip(ef.CHARGING_POWER, chargers) if n},
                far_sites=len(far), max_distance_m=float(mapping.nearest_distance_m.max()) if len(mapping) else 0.0)


def draw_city(city, out_file, far_m=500, mapping_file=None):
    graph, stations, mapping = load_city(city, mapping_file)
    info = summary(graph, stations, mapping, far_m)
    fig, ax = plt.subplots(figsize=(8, 8.6), facecolor=SURFACE)
    ox.plot_graph(graph, ax=ax, node_size=0, edge_color=ROAD, edge_linewidth=0.3, bgcolor=SURFACE,
                  show=False, close=False)
    ax.set_facecolor(SURFACE)

    far = mapping[mapping.nearest_distance_m > far_m]
    for row in far.itertuples():
        node = graph.nodes[row.nearest_node]
        ax.plot([row.longitude, node["x"]], [row.latitude, node["y"]], color=LINK, lw=0.7, ls=(0, (3, 2)), zorder=2)
    if len(far):
        ax.scatter(far.longitude, far.latitude, s=22, facecolors="none", edgecolors=LINK, linewidths=0.9, zorder=3)

    nodes = list(stations)
    ax.scatter([graph.nodes[n]["x"] for n in nodes], [graph.nodes[n]["y"] for n in nodes],
               s=[marker_area(int(stations[n].sum())) for n in nodes], color=STATION, edgecolors=SURFACE,
               linewidths=0.6, zorder=4)
    # view: road network plus the OCM sites outside it, degrees scaled to equal ground distance
    xs = [data["x"] for _, data in graph.nodes(data=True)] + far.longitude.tolist()
    ys = [data["y"] for _, data in graph.nodes(data=True)] + far.latitude.tolist()
    pad_x, pad_y = 0.03 * (max(xs) - min(xs)), 0.03 * (max(ys) - min(ys))
    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
    ax.set_aspect(1 / np.cos(np.radians(np.mean(ys))))

    powers = ", ".join("{} x {} kW".format(n, p) for p, n in sorted(info["chargers_by_power"].items(), reverse=True))
    ax.set_title("{}: road network and existing charging stations".format(city), loc="left", fontsize=13,
                 color=INK, pad=34)
    ax.text(0, 1.045, "{:,} road nodes, {:,} edges | {} OCM sites mapped to {} stations, {} chargers ({})".format(
        info["nodes"], info["edges"], info["ocm_sites"], info["stations"], info["chargers"], powers),
        transform=ax.transAxes, fontsize=8.5, color=MUTED)
    if info["far_sites"]:
        note = "{} OCM sites lie more than {} m from the road network (farthest {:.1f} km) and are mapped to the " \
               "nearest node at its edge".format(info["far_sites"], far_m, info["max_distance_m"] / 1000)
    else:
        note = "every OCM site lies within {} m of its road node (farthest {:.0f} m)".format(far_m,
                                                                                          info["max_distance_m"])
    ax.text(0, 1.015, note, transform=ax.transAxes, fontsize=8.5, color=MUTED)

    legend = [Line2D([0], [0], marker="o", lw=0, color=STATION, markeredgecolor=SURFACE,
                     markersize=np.sqrt(area), label="{}{} chargers".format(low, "+" if high is None else
                                                                            "-{}".format(high)))
              for low, high, area in SIZE_BINS]
    legend.append(Line2D([0], [0], color=ROAD, lw=1.5, label="Road"))
    if info["far_sites"]:
        legend.append(Line2D([0], [0], marker="o", color=LINK, ls=(0, (3, 2)), lw=0.7, markerfacecolor="none",
                             markersize=5, label="OCM location > {} m away".format(far_m)))
    fig.legend(handles=legend, loc="lower center", ncol=len(legend), frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    fig.savefig(out_file, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return info


def main(argv=None):
    parser = argparse.ArgumentParser(description="Maps of the road network and the existing charging stations.")
    parser.add_argument("--cities", nargs="+", default=["Tainan", "Taipei"])
    parser.add_argument("--far-m", type=float, default=500,
                        help="show the OCM location of sites farther than this from their road node (metres)")
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args(argv)
    for city in args.cities:
        out_file = os.path.join(args.out_dir, "{}_network_stations.png".format(city))
        info = draw_city(city, out_file, args.far_m)
        print("{}: {}".format(out_file, info))


if __name__ == '__main__':
    main()

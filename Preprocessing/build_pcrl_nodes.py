# Preprocessing/build_pcrl_nodes.py

import os
import argparse

import numpy as np
import osmnx as ox


# ============================================================
# Synthetic demand
# ============================================================

def build_synthetic_demand(G):
    """
    Generate synthetic charging demand from road-network
    connectivity.

    Higher street_count -> higher demand.

    This is NOT real EV demand.
    """

    values = []

    for node, data in G.nodes(data=True):

        street_count = data.get(
            "street_count",
            1
        )

        try:
            street_count = float(
                street_count
            )
        except (TypeError, ValueError):
            street_count = 1.0

        values.append(
            street_count
        )

    values = np.array(
        values,
        dtype=float
    )

    min_value = values.min()
    max_value = values.max()

    if max_value == min_value:

        normalized = np.ones_like(
            values
        )

    else:

        normalized = (
            values - min_value
        ) / (
            max_value - min_value
        )

    # Avoid exactly zero demand.
    normalized = 0.05 + 0.95 * normalized

    demand = {}

    for (node, _), value in zip(
        G.nodes(data=True),
        normalized
    ):
        demand[node] = float(value)

    return demand


# ============================================================
# Synthetic estate price
# ============================================================

def build_synthetic_estate_price(G):
    """
    Generate a normalized synthetic estate-price proxy.

    Current version:
        based on node connectivity.

    Replace this function later if real land-price data
    becomes available.
    """

    values = {}

    for node, data in G.nodes(data=True):

        street_count = data.get(
            "street_count",
            1
        )

        try:
            street_count = float(
                street_count
            )
        except (TypeError, ValueError):
            street_count = 1.0

        values[node] = street_count

    arr = np.array(
        list(values.values()),
        dtype=float
    )

    min_value = arr.min()
    max_value = arr.max()

    if max_value == min_value:

        normalized = np.ones_like(
            arr
        )

    else:

        normalized = (
            arr - min_value
        ) / (
            max_value - min_value
        )

    # Avoid zero.
    normalized = 0.1 + 0.9 * normalized

    result = {}

    for node, value in zip(
        values.keys(),
        normalized
    ):

        result[node] = float(
            value
        )

    return result


# ============================================================
# Build PCRL node list
# ============================================================

def build_node_list(
    G,
    synthetic_demand,
    estate_price
):

    node_list = []

    # First create base node representation.
    for node_id, data in G.nodes(data=True):

        node = dict(data)

        # ----------------------------------------------------
        # Required PCRL fields
        # ----------------------------------------------------

        node["x"] = float(
            data["x"]
        )

        node["y"] = float(
            data["y"]
        )

        node["demand"] = float(
            synthetic_demand[node_id]
        )

        node["estate price"] = float(
            estate_price[node_id]
        )

        # No private home charging data available.
        node["private CS"] = 0

        # These are useful for debugging.
        node["osmid"] = node_id

        node_list.append(
            (
                node_id,
                node
            )
        )
    return node_list


# ============================================================
# Save exactly in PCRL format
# ============================================================

def save_node_list(
    node_list,
    output_file
):

    os.makedirs(
        os.path.dirname(output_file),
        exist_ok=True
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        # PCRL loads this using:
        #
        # my_node_list = eval(file.readline())
        #
        f.write(
            repr(node_list)
        )

        f.write("\n")

    print(
        f"Saved: {output_file}"
    )


# ============================================================
# Main
# ============================================================

def build_pcrl_nodes(
    graph_file,
    output_file
):

    print("=" * 70)
    print("PCRL node preprocessing")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load graph
    # --------------------------------------------------------

    print("[1/4] Loading OSM graph...")

    G = ox.load_graphml(
        graph_file
    )

    print(
        f"Nodes: {len(G.nodes)}"
    )

    # --------------------------------------------------------
    # 2. Demand
    # --------------------------------------------------------

    print(
        "[2/4] Generating synthetic demand..."
    )

    synthetic_demand = (
        build_synthetic_demand(G)
    )

    # --------------------------------------------------------
    # 3. Estate price
    # --------------------------------------------------------

    print(
        "[3/4] Generating synthetic estate price..."
    )

    estate_price = (
        build_synthetic_estate_price(G)
    )

    # --------------------------------------------------------
    # 4. Node list
    # --------------------------------------------------------

    print(
        "[4/4] Building PCRL node list..."
    )

    node_list = build_node_list(
        G,
        synthetic_demand,
        estate_price
    )

    save_node_list(
        node_list,
        output_file
    )

    print()
    print("=" * 70)
    print("Finished")
    print("=" * 70)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--graph",
        required=True
    )

    parser.add_argument(
        "--output",
        required=True
    )

    args = parser.parse_args()

    build_pcrl_nodes(
        graph_file=args.graph,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
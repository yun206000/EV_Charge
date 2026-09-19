# Preprocessing/build_ocm_plan.py

import os
import json
import pickle
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import osmnx as ox


# ============================================================
# Charger power mapping
# ============================================================

def power_to_pcrl_type(power_kw):
    """
    Map OCM charging power to PCRL charger categories.

    PCRL categories:
        7 kW
        22 kW
        50 kW
    """

    if power_kw is None:
        return None

    try:
        power_kw = float(power_kw)
    except (TypeError, ValueError):
        return None

    if power_kw <= 11:
        return 0       # 7 kW

    elif power_kw <= 35:
        return 1       # 22 kW

    else:
        return 2       # 50 kW


# ============================================================
# Extract charger counts from OCM
# ============================================================

def extract_charger_counts(station):
    """
    Extract [n_7kw, n_22kw, n_50kw] from an OCM station.

    Priority:
        1. Connections[].PowerKW + Quantity
        2. NumberOfPoints as 22 kW
    """

    counts = np.array(
        [0, 0, 0],
        dtype=int
    )

    connections = station.get("Connections")

    if isinstance(connections, list):

        for connection in connections:

            power_kw = connection.get("PowerKW")
            quantity = connection.get("Quantity", 0)

            try:
                quantity = int(quantity or 0)
            except (TypeError, ValueError):
                quantity = 0

            charger_type = power_to_pcrl_type(power_kw)

            if charger_type is not None and quantity > 0:
                counts[charger_type] += quantity

    # If no usable connection information exists,
    # fall back to NumberOfPoints.
    if counts.sum() == 0:

        number_of_points = station.get(
            "NumberOfPoints",
            0
        )

        try:
            number_of_points = int(
                number_of_points or 0
            )
        except (TypeError, ValueError):
            number_of_points = 0

        # Same assumption as original PCRL:
        # NumberOfPoints -> 22 kW.
        counts[1] = number_of_points

    # Conservative fallback
    if counts.sum() == 0:
        counts[1] = 1

    return counts


# ============================================================
# Load OCM JSON
# ============================================================

def load_ocm_json(input_file):
    """
    Load an OCM API JSON file.
    """

    with open(
        input_file,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    # OCM API responses usually contain a list.
    if isinstance(data, list):
        return data

    # Some saved API responses may use Data.
    if isinstance(data, dict):

        if "Data" in data:
            return data["Data"]

        if "data" in data:
            return data["data"]

    raise ValueError(
        "Cannot find OCM station list in JSON."
    )


# ============================================================
# OCM -> nearest OSM node
# ============================================================

def map_ocm_to_osm(
    stations,
    G,
    max_distance_m=100
):
    """
    Map every OCM station to the nearest OSM node.

    Returns
    -------
    mapped_rows : list
    """

    rows = []

    for station in stations:

        address_info = station.get(
            "AddressInfo",
            {}
        )

        latitude = address_info.get("Latitude")
        longitude = address_info.get("Longitude")

        if latitude is None or longitude is None:
            continue

        try:
            latitude = float(latitude)
            longitude = float(longitude)
        except (TypeError, ValueError):
            continue

        nearest_node, distance = ox.distance.nearest_nodes(
            G,
            X=longitude,
            Y=latitude,
            return_dist=True
        )

        distance = float(distance)

        # Quality control.
        if distance > max_distance_m:
            print(
                f"WARNING: OCM station {station.get('ID')} "
                f"is {distance:.1f} m from nearest OSM node. "
                f"Skipping."
            )
            continue

        counts = extract_charger_counts(
            station
        )

        rows.append({
            "ocm_id": station.get("ID"),
            "latitude": latitude,
            "longitude": longitude,
            "nearest_node": int(nearest_node),
            "nearest_distance_m": distance,
            "n_7kw": int(counts[0]),
            "n_22kw": int(counts[1]),
            "n_50kw": int(counts[2]),
            "numberofpoints": int(counts.sum())
        })

    return rows


# ============================================================
# Aggregate stations mapped to same OSM node
# ============================================================

def aggregate_by_osm_node(mapped_rows):

    aggregated = defaultdict(
        lambda: np.array(
            [0, 0, 0],
            dtype=int
        )
    )

    for row in mapped_rows:

        node = row["nearest_node"]

        aggregated[node] += np.array([
            row["n_7kw"],
            row["n_22kw"],
            row["n_50kw"]
        ])

    return aggregated


# ============================================================
# Build PCRL existing plan
# ============================================================

def build_existing_plan(
    G,
    aggregated
):
    """
    Build PCRL existingplan structure.

    Each station is:

        [
            node_tuple,
            np.array([n7, n22, n50]),
            {}
        ]
    """

    existing_plan = []

    for node_id, charger_counts in aggregated.items():

        if node_id not in G.nodes:
            print(
                f"WARNING: node {node_id} "
                f"not found in graph."
            )
            continue

        node_data = dict(
            G.nodes[node_id]
        )

        node_tuple = (
            node_id,
            node_data
        )

        s_x = np.array(
            charger_counts,
            dtype=int
        )

        station = [
            node_tuple,
            s_x,
            {}
        ]

        existing_plan.append(
            station
        )

    return existing_plan


# ============================================================
# Save mapping CSV
# ============================================================

def save_mapping_csv(
    mapped_rows,
    output_file
):

    df = pd.DataFrame(
        mapped_rows
    )

    df.to_csv(
        output_file,
        index=False
    )

    print(
        f"Mapping CSV saved to: {output_file}"
    )


# ============================================================
# Main pipeline
# ============================================================

def build_ocm_plan(
    graph_file,
    ocm_file,
    output_pkl,
    mapping_csv,
    max_distance_m=100
):

    print("=" * 70)
    print("PCRL OCM preprocessing")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load graph
    # --------------------------------------------------------

    print("[1/5] Loading OSM graph...")

    G = ox.load_graphml(
        graph_file
    )

    print(
        f"Nodes: {len(G.nodes)}"
    )

    # --------------------------------------------------------
    # 2. Load OCM
    # --------------------------------------------------------

    print("[2/5] Loading OCM data...")

    stations = load_ocm_json(
        ocm_file
    )

    print(
        f"OCM stations: {len(stations)}"
    )

    # --------------------------------------------------------
    # 3. Map to OSM
    # --------------------------------------------------------

    print("[3/5] Mapping OCM stations to OSM nodes...")

    mapped_rows = map_ocm_to_osm(
        stations,
        G,
        max_distance_m=max_distance_m
    )

    print(
        f"Mapped stations: {len(mapped_rows)}"
    )

    # --------------------------------------------------------
    # 4. Aggregate duplicate nodes
    # --------------------------------------------------------

    print("[4/5] Aggregating stations...")

    aggregated = aggregate_by_osm_node(
        mapped_rows
    )

    print(
        f"Unique OSM station nodes: "
        f"{len(aggregated)}"
    )

    # --------------------------------------------------------
    # 5. Build pickle
    # --------------------------------------------------------

    print("[5/5] Building PCRL existing plan...")

    existing_plan = build_existing_plan(
        G,
        aggregated
    )

    os.makedirs(
        os.path.dirname(output_pkl),
        exist_ok=True
    )

    with open(
        output_pkl,
        "wb"
    ) as f:

        pickle.dump(
            existing_plan,
            f
        )

    save_mapping_csv(
        mapped_rows,
        mapping_csv
    )

    print()
    print("=" * 70)
    print("Finished")
    print("=" * 70)

    print(
        f"existingplan : {output_pkl}"
    )

    print(
        f"mapping CSV  : {mapping_csv}"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--graph",
        required=True
    )

    parser.add_argument(
        "--ocm",
        required=True
    )

    parser.add_argument(
        "--output-pkl",
        required=True
    )

    parser.add_argument(
        "--mapping-csv",
        required=True
    )

    parser.add_argument(
        "--max-distance",
        type=float,
        default=100,
        help="Maximum OCM -> OSM matching distance in meters"
    )

    args = parser.parse_args()

    build_ocm_plan(
        graph_file=args.graph,
        ocm_file=args.ocm,
        output_pkl=args.output_pkl,
        mapping_csv=args.mapping_csv,
        max_distance_m=args.max_distance
    )


if __name__ == "__main__":
    main()
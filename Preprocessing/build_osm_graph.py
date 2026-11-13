# Preprocessing/build_osm_graph.py

import os
import argparse
import osmnx as ox


def build_osm_graph(
    place: str,
    location: str,
    output_dir: str = "Graph",
    from_graph: str = None
):
    """
    Download OSM driving network.

    Parameters
    ----------
    place:
        OSM place query, e.g. "Taipei, Taiwan"

    location:
        Internal PCRL location name, e.g. "Taipei"

    output_dir:
        Root directory for PCRL graph files.

    from_graph:
        Optional GraphML of a larger, already downloaded area (e.g. Graph/Tainan/Tainan.graphml). The graph is then
        cut to the place polygon (from Nominatim) instead of downloading it from Overpass, which helps when the
        Overpass servers are overloaded; like graph_from_place(retain_all=False) only the largest connected part is
        kept.
    """

    location_dir = os.path.join(output_dir, location)
    os.makedirs(location_dir, exist_ok=True)

    print("=" * 70)
    print("PCRL OSM preprocessing")
    print("=" * 70)
    print(f"OSM place   : {place}")
    print(f"Location    : {location}")
    print()

    if from_graph:
        print(f"[1/2] Cutting the place polygon out of {from_graph}...")
        places = ox.geocode_to_gdf(place)
        # geopandas >= 1.0 renamed unary_union to union_all
        polygon = places.union_all() if hasattr(places, "union_all") else places.unary_union
        G = ox.load_graphml(from_graph)
        if int(ox.__version__.split(".")[0]) >= 2:
            # osmnx 2 removed retain_all: keep the largest weakly connected part, as retain_all=False did
            G = ox.truncate.largest_component(
                ox.truncate.truncate_graph_polygon(G, polygon),
                strongly=False
            )
        else:
            G = ox.truncate.truncate_graph_polygon(
                G,
                polygon,
                retain_all=False
            )
    else:
        print("[1/2] Downloading OSM graph...")

        G = ox.graph_from_place(
            place,
            network_type="drive",
            simplify=True,
            retain_all=False
        )

    print(f"Nodes       : {len(G.nodes)}")
    print(f"Edges       : {len(G.edges)}")

    # Validate node coordinates.
    for node, data in G.nodes(data=True):
        if "x" not in data or "y" not in data:
            raise ValueError(
                f"Node {node} is missing x/y coordinates."
            )

    # Make sure edge length exists.
    missing_length = 0

    for u, v, key, data in G.edges(keys=True, data=True):
        if "length" not in data:
            missing_length += 1

    if missing_length > 0:
        print(
            f"WARNING: {missing_length} edges do not have length."
        )

    output_file = os.path.join(
        location_dir,
        f"{location}.graphml"
    )

    print()
    print("[2/2] Saving GraphML...")

    ox.save_graphml(
        G,
        filepath=output_file
    )

    print()
    print("Finished.")
    print(f"Output: {output_file}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--place",
        required=True,
        help='OSM place query, e.g. "Taipei, Taiwan"'
    )

    parser.add_argument(
        "--location",
        required=True,
        help='PCRL location name, e.g. "Taipei"'
    )

    parser.add_argument(
        "--output-dir",
        default="Graph",
        help="Root Graph directory"
    )

    parser.add_argument(
        "--from-graph",
        default=None,
        help=(
            "GraphML of a larger downloaded area to cut the place out of, instead of downloading from Overpass, "
            'e.g. "Graph/Tainan/Tainan.graphml"'
        )
    )

    args = parser.parse_args()

    build_osm_graph(
        place=args.place,
        location=args.location,
        output_dir=args.output_dir,
        from_graph=args.from_graph
    )


if __name__ == "__main__":
    main()
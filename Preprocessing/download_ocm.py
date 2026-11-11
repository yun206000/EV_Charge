import os
import json
import time
import argparse
import requests


ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docker", ".env")


def read_env_file(path=ENV_FILE):
    """KEY=VALUE lines of docker/.env (the file docker run --env-file reads); {} if it does not exist"""
    values = {}
    if os.path.exists(path):
        with open(path) as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    name, value = line.split("=", 1)
                    values[name.strip()] = value.strip().strip('"').strip("'")
    return values

"""
Download charging station data from Open Charge Map (OCM) for a given location.

Two ways to select the area:
    1. --latitude / --longitude / --distance : a circle around a center point.
    2. --place : the same OSM place query used in build_osm_graph.py.
       The bounding box of the place polygon is sent to OCM, and the result is
       then filtered by the polygon itself, so the OCM area matches the OSM graph.
"""


OCM_API_URL = "https://api.openchargemap.io/v3/poi/"


def fetch_ocm(
    api_key,
    latitude=None,
    longitude=None,
    distance_km=20,
    max_results=100,
    country_code="TW",
    status_type_id=50,
    bounding_box=None,
    verbose=True,
):
    """
    Download charging stations from Open Charge Map.

    Parameters
    ----------
    api_key : str
        Open Charge Map API key.

    latitude : float
        Center latitude.

    longitude : float
        Center longitude.

    distance_km : float
        Search radius in kilometers.

    max_results : int
        Number of results requested per API call.

    country_code : str or None
        ISO country code. Taiwan = TW.

    status_type_id : int or None
        OCM status filter.
        50 = Operational.

        Set to None if you do not want to filter
        by operational status.

    bounding_box : tuple or None
        (south, west, north, east). If given, it is used
        instead of latitude / longitude / distance_km.

    Returns
    -------
    stations : list
        List of OCM station dictionaries.
    """

    headers = {
        "User-Agent": "PCRL-OCM-Downloader/1.0"
    }

    params = {
        "key": api_key,
        "maxresults": max_results,
    }

    if bounding_box is not None:
        south, west, north, east = bounding_box
        params["boundingbox"] = f"({south},{west}),({north},{east})"
    else:
        params["latitude"] = latitude
        params["longitude"] = longitude
        params["distance"] = distance_km
        params["distanceunit"] = "KM"

    if country_code:
        params["countrycode"] = country_code

    if status_type_id is not None:
        params["statustypeid"] = status_type_id

    if verbose:
        print("=" * 70)
        print("Open Charge Map downloader")
        print("=" * 70)
        if bounding_box is not None:
            print(f"Bounding box   : {bounding_box}")
        else:
            print(f"Latitude       : {latitude}")
            print(f"Longitude      : {longitude}")
            print(f"Distance       : {distance_km} km")
        print(f"Country        : {country_code}")
        print(f"StatusTypeID   : {status_type_id}")
        print()

    print("Requesting OCM API...")

    response = requests.get(
        OCM_API_URL,
        params=params,
        headers=headers,
        timeout=60,
    )

    print(
        f"HTTP status: {response.status_code}"
    )

    if response.status_code != 200:
        print()
        print("OCM API request failed.")
        print(response.text[:1000])

        response.raise_for_status()

    data = response.json()

    if not isinstance(data, list):
        raise ValueError(
            "Unexpected OCM API response format. "
            "Expected a list of stations."
        )

    if verbose:
        print(
            f"Stations returned: {len(data)}"
        )

    if len(data) >= max_results:
        print(
            f"WARNING: OCM returned {len(data)} stations, which equals "
            f"--max-results. The result is probably truncated."
        )

    return data


def load_place_polygon(place):
    """
    Geocode the OSM place query to the same polygon that
    ox.graph_from_place uses in build_osm_graph.py.
    """

    import osmnx as ox

    gdf = ox.geocode_to_gdf(place)

    return gdf.unary_union


def filter_by_polygon(
    stations,
    polygon,
):
    """
    Keep only the stations located inside the polygon.
    """

    from shapely.geometry import Point

    kept = []

    for station in stations:

        address_info = station.get(
            "AddressInfo",
            {}
        )

        lat = address_info.get("Latitude")
        lon = address_info.get("Longitude")

        if lat is None or lon is None:
            continue

        if polygon.covers(Point(float(lon), float(lat))):
            kept.append(station)

    print(
        f"Stations inside place polygon: {len(kept)} / {len(stations)}"
    )

    return kept


def save_json(
    stations,
    output_file,
):
    """
    Save raw OCM response.
    """

    output_dir = os.path.dirname(
        output_file
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True
        )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            stations,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        f"Saved OCM JSON: {output_file}"
    )


def generate_summary(
    stations,
):
    """
    Generate basic statistics for data-quality checking.
    """

    total = len(stations)

    valid_coordinates = 0
    number_of_points = 0
    connections = 0
    power_kw = 0

    for station in stations:

        address_info = station.get(
            "AddressInfo",
            {}
        )

        lat = address_info.get(
            "Latitude"
        )

        lon = address_info.get(
            "Longitude"
        )

        if lat is not None and lon is not None:
            valid_coordinates += 1

        if station.get(
            "NumberOfPoints"
        ) is not None:
            number_of_points += 1

        station_connections = station.get(
            "Connections"
        )

        if isinstance(
            station_connections,
            list
        ):

            if len(station_connections) > 0:
                connections += 1

            for connection in station_connections:

                if connection.get(
                    "PowerKW"
                ) is not None:

                    power_kw += 1

    return {
        "total_stations": total,
        "valid_coordinates": valid_coordinates,
        "has_number_of_points": number_of_points,
        "has_connections": connections,
        "connections_with_power_kw": power_kw,
    }


def save_summary(
    summary,
    output_file,
):

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"Saved summary: {output_file}"
    )


def print_summary(
    summary,
):

    print()
    print("-" * 70)
    print("OCM data summary")
    print("-" * 70)

    for key, value in summary.items():

        print(
            f"{key:30s}: {value}"
        )

    print("-" * 70)


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Download charging station data "
            "from Open Charge Map."
        )
    )

    # --------------------------------------------------------
    # API key
    # --------------------------------------------------------

    parser.add_argument(
        "--api-key",
        default=None,
        help=(
            "Open Charge Map API key. "
            "If omitted, the OCM_API_KEY environment "
            "variable is used, else OCM_API_KEY in docker/.env."
        ),
    )

    # --------------------------------------------------------
    # Location
    # --------------------------------------------------------

    parser.add_argument(
        "--place",
        default=None,
        help=(
            'OSM place query, e.g. "Tainan, Taiwan". '
            "Use the same query as build_osm_graph.py so that "
            "the OCM area is aligned with the OSM graph. "
            "Overrides --latitude / --longitude / --distance."
        ),
    )

    parser.add_argument(
        "--latitude",
        type=float,
        default=None,
        help="Center latitude.",
    )

    parser.add_argument(
        "--longitude",
        type=float,
        default=None,
        help="Center longitude.",
    )

    # --------------------------------------------------------
    # Search parameters
    # --------------------------------------------------------

    parser.add_argument(
        "--distance",
        type=float,
        default=20,
        help="Search radius in km. Default: 20",
    )

    parser.add_argument(
        "--max-results",
        type=int,
        default=100,
        help=(
            "Maximum results requested per API call. "
            "Default: 100"
        ),
    )

    parser.add_argument(
        "--country-code",
        default="TW",
        help=(
            "OCM country code. "
            "Taiwan = TW. "
            "Use empty string to disable."
        ),
    )

    parser.add_argument(
        "--status-type-id",
        type=int,
        default=50,
        help=(
            "OCM status type ID. "
            "50 = Operational. "
            "Use -1 to disable filtering."
        ),
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON file.",
    )

    parser.add_argument(
        "--summary-output",
        default=None,
        help=(
            "Optional summary JSON file. "
            "Default: <output>.summary.json"
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # API key
    # --------------------------------------------------------

    api_key = args.api_key

    if api_key is None:
        api_key = os.environ.get(
            "OCM_API_KEY"
        )

    if api_key is None:
        api_key = read_env_file().get("OCM_API_KEY")

    if not api_key:

        raise RuntimeError(
            "No OCM API key found.\n\n"
            "Use either:\n"
            "  --api-key YOUR_API_KEY\n\n"
            "or set:\n"
            "  export OCM_API_KEY=YOUR_API_KEY\n\n"
            "or write OCM_API_KEY=YOUR_API_KEY into docker/.env "
            "(see docker/.env.example)"
        )

    # --------------------------------------------------------
    # Status filter
    # --------------------------------------------------------

    status_type_id = args.status_type_id

    if status_type_id == -1:
        status_type_id = None

    # --------------------------------------------------------
    # Area
    # --------------------------------------------------------

    polygon = None
    bounding_box = None

    if args.place:

        polygon = load_place_polygon(
            args.place
        )

        west, south, east, north = polygon.bounds
        bounding_box = (south, west, north, east)

    elif args.latitude is None or args.longitude is None:

        parser.error(
            "Either --place or both --latitude and --longitude are required."
        )

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    stations = fetch_ocm(
        api_key=api_key,
        latitude=args.latitude,
        longitude=args.longitude,
        distance_km=args.distance,
        max_results=args.max_results,
        country_code=(
            args.country_code
            if args.country_code
            else None
        ),
        status_type_id=status_type_id,
        bounding_box=bounding_box,
    )

    if polygon is not None:

        stations = filter_by_polygon(
            stations,
            polygon,
        )

    # --------------------------------------------------------
    # Save raw JSON
    # --------------------------------------------------------

    save_json(
        stations,
        args.output,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = generate_summary(
        stations
    )

    print_summary(
        summary
    )

    summary_output = (
        args.summary_output
    )

    if summary_output is None:
        summary_output = (
            args.output
            + ".summary.json"
        )

    save_summary(
        summary,
        summary_output,
    )

    print()
    print("=" * 70)
    print("Download finished.")
    print("=" * 70)


if __name__ == "__main__":
    main()
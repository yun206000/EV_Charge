import importlib.util
import json
import os
import pickle
import runpy
import sys
import networkx as nx
import numpy as np
import osmnx as ox
import pytest
from shapely.geometry import Point, box
from conftest import build_graph

"""
Preprocessing scripts (OSM graph, OCM charging stations, PCRL node list) without network access: the Overpass /
Nominatim / Open Charge Map requests are replaced by small fakes, all files are written below tmp_path.
"""

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRE = os.path.join(REPO, "Preprocessing")


def load(name):
    spec = importlib.util.spec_from_file_location("pre_" + name, os.path.join(PRE, name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ocm = load("download_ocm")
plan = load("build_ocm_plan")
nodes = load("build_pcrl_nodes")
osm = load("build_osm_graph")
grid = load("load_graph")
prep = load("nodes_preparation")


def station(sid, lon, lat, **extra):
    return dict(ID=sid, AddressInfo=dict(Latitude=lat, Longitude=lon), **extra)


# synthetic grid of conftest: x 120.200-120.216, y 23.000-23.016
STATIONS = [
    station(1, 120.204, 23.004, Connections=[dict(PowerKW=7, Quantity=2), dict(PowerKW=50, Quantity=1)]),
    station(2, 120.2041, 23.0041, Connections=[dict(PowerKW=22, Quantity="3")]),  # same node as station 1
    station(3, 120.212, 23.012, NumberOfPoints=4, Connections=[]),
    station(4, 120.50, 23.50, NumberOfPoints=1),  # far away from every node
    dict(ID=5, AddressInfo=dict(Latitude=None, Longitude=120.2)),  # no coordinates
    station(6, "x", "y"),  # coordinates that are not numbers
]


class FakeResponse:
    def __init__(self, status, data, text=""):
        self.status_code, self._data, self.text = status, data, text

    def json(self):
        return self._data

    def raise_for_status(self):
        raise RuntimeError("HTTP {}".format(self.status_code))


@pytest.fixture
def fake_get(monkeypatch):
    calls = []

    def get(url, params, headers, timeout):
        calls.append(dict(url=url, params=params, headers=headers, timeout=timeout))
        return FakeResponse(200, list(STATIONS))
    monkeypatch.setattr(ocm.requests, "get", get)
    return calls


# ------------------------------------------------------------------ download_ocm


def test_read_env_file(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# comment\n\nOCM_API_KEY = 'abc'\nOTHER=\"x=y\"\nnot a pair\n")
    assert ocm.read_env_file(str(env)) == {"OCM_API_KEY": "abc", "OTHER": "x=y"}
    assert ocm.read_env_file(str(tmp_path / "missing")) == {}


def test_fetch_ocm_circle(fake_get, capsys):
    data = ocm.fetch_ocm("KEY", latitude=23.0, longitude=120.2, distance_km=5, max_results=6)
    assert data == STATIONS
    params = fake_get[0]["params"]
    assert params["latitude"] == 23.0 and params["distance"] == 5 and params["distanceunit"] == "KM"
    assert params["countrycode"] == "TW" and params["statustypeid"] == 50 and fake_get[0]["timeout"] == 60
    assert fake_get[0]["url"].startswith("https://")
    out = capsys.readouterr().out
    assert "Distance       : 5 km" in out and "probably truncated" in out  # 6 results = --max-results


def test_fetch_ocm_bounding_box_without_filters(fake_get, capsys):
    ocm.fetch_ocm("KEY", bounding_box=(23.0, 120.2, 23.1, 120.3), country_code=None, status_type_id=None,
                  verbose=False)
    params = fake_get[0]["params"]
    assert params["boundingbox"] == "(23.0,120.2),(23.1,120.3)"
    assert "countrycode" not in params and "statustypeid" not in params and "latitude" not in params
    assert "Bounding box" not in capsys.readouterr().out
    ocm.fetch_ocm("KEY", bounding_box=(23.0, 120.2, 23.1, 120.3))
    assert "Bounding box   : (23.0, 120.2, 23.1, 120.3)" in capsys.readouterr().out


def test_fetch_ocm_errors(monkeypatch, capsys):
    monkeypatch.setattr(ocm.requests, "get", lambda *a, **k: FakeResponse(403, None, "forbidden"))
    with pytest.raises(RuntimeError, match="HTTP 403"):
        ocm.fetch_ocm("KEY", latitude=1, longitude=2)
    assert "forbidden" in capsys.readouterr().out
    monkeypatch.setattr(ocm.requests, "get", lambda *a, **k: FakeResponse(200, {"error": "x"}))
    with pytest.raises(ValueError, match="Expected a list"):
        ocm.fetch_ocm("KEY", latitude=1, longitude=2)


def test_place_polygon_and_filter(monkeypatch, capsys):
    import geopandas as gpd
    area = box(120.20, 23.00, 120.21, 23.01)
    monkeypatch.setattr(ox, "geocode_to_gdf", lambda place: gpd.GeoDataFrame(geometry=[area], crs="epsg:4326"))
    polygon = ocm.load_place_polygon("Synth, Taiwan")
    assert polygon.equals(area)
    kept = ocm.filter_by_polygon(STATIONS[:5], polygon)
    assert [s["ID"] for s in kept] == [1, 2]
    assert "Stations inside place polygon: 2 / 5" in capsys.readouterr().out


def test_summary_and_files(tmp_path, capsys):
    summary = ocm.generate_summary(STATIONS)
    assert summary == dict(total_stations=6, valid_coordinates=5, has_number_of_points=2, has_connections=2,
                           connections_with_power_kw=3)
    ocm.save_json(STATIONS, str(tmp_path / "sub" / "ocm.json"))
    assert json.loads((tmp_path / "sub" / "ocm.json").read_text(encoding="utf-8")) == STATIONS
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        ocm.save_json([], "plain.json")  # no folder part
    finally:
        os.chdir(cwd)
    ocm.save_summary(summary, str(tmp_path / "summary.json"))
    assert json.loads((tmp_path / "summary.json").read_text())["total_stations"] == 6
    ocm.print_summary(summary)
    assert "valid_coordinates" in capsys.readouterr().out


def run_main(module, monkeypatch, args):
    monkeypatch.setattr(sys, "argv", ["prog"] + args)
    return module.main()


def test_download_main_with_key_argument(fake_get, monkeypatch, tmp_path):
    out = tmp_path / "ocm.json"
    run_main(ocm, monkeypatch, ["--api-key", "K1", "--latitude", "23", "--longitude", "120.2", "--output", str(out),
                                "--status-type-id", "-1", "--country-code", ""])
    assert fake_get[0]["params"]["key"] == "K1" and "statustypeid" not in fake_get[0]["params"]
    assert "countrycode" not in fake_get[0]["params"]
    assert json.loads((tmp_path / "ocm.json.summary.json").read_text())["total_stations"] == 6


def test_download_main_key_sources(fake_get, monkeypatch, tmp_path):
    args = ["--latitude", "23", "--longitude", "120.2", "--output", str(tmp_path / "o.json"),
            "--summary-output", str(tmp_path / "s.json")]
    monkeypatch.setenv("OCM_API_KEY", "FROM_ENV")
    run_main(ocm, monkeypatch, args)
    assert fake_get[-1]["params"]["key"] == "FROM_ENV"
    monkeypatch.delenv("OCM_API_KEY")
    monkeypatch.setattr(ocm, "read_env_file", lambda: {"OCM_API_KEY": "FROM_FILE"})
    run_main(ocm, monkeypatch, args)
    assert fake_get[-1]["params"]["key"] == "FROM_FILE" and (tmp_path / "s.json").exists()
    monkeypatch.setattr(ocm, "read_env_file", lambda: {})
    with pytest.raises(RuntimeError, match="No OCM API key"):
        run_main(ocm, monkeypatch, args)


def test_download_main_place(fake_get, monkeypatch, tmp_path):
    monkeypatch.setattr(ocm.requests, "get", lambda url, params, **kwargs: fake_get.append(dict(params=params)) or
                        FakeResponse(200, STATIONS[:5]))
    monkeypatch.setattr(ocm, "load_place_polygon", lambda place: box(120.20, 23.00, 120.21, 23.01))
    run_main(ocm, monkeypatch, ["--api-key", "K", "--place", "Synth", "--output", str(tmp_path / "o.json")])
    assert fake_get[0]["params"]["boundingbox"] == "(23.0,120.2),(23.01,120.21)"
    assert [s["ID"] for s in json.loads((tmp_path / "o.json").read_text())] == [1, 2]


def test_download_main_needs_area(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        run_main(ocm, monkeypatch, ["--api-key", "K", "--latitude", "23", "--output", str(tmp_path / "o.json")])


# ------------------------------------------------------------------ build_ocm_plan


@pytest.mark.parametrize("power, expected", [(None, None), ("abc", None), (7, 0), (11, 0), ("22", 1), (35, 1),
                                             (50, 2), (350.0, 2)])
def test_power_to_pcrl_type(power, expected):
    assert plan.power_to_pcrl_type(power) == expected


@pytest.mark.parametrize("data, expected", [
    (dict(Connections=[dict(PowerKW=7, Quantity=2), dict(PowerKW=50)]), [2, 0, 0]),
    (dict(Connections=[dict(PowerKW=22, Quantity="x"), dict(PowerKW=None, Quantity=3)], NumberOfPoints="4"), [0, 4, 0]),
    (dict(NumberOfPoints="many"), [0, 1, 0]),
    (dict(Connections="none"), [0, 1, 0]),
])
def test_extract_charger_counts(data, expected):
    assert plan.extract_charger_counts(data).tolist() == expected


def test_load_ocm_json(tmp_path):
    for content in [[1, 2], {"Data": [1, 2]}, {"data": [1, 2]}]:
        (tmp_path / "f.json").write_text(json.dumps(content))
        assert plan.load_ocm_json(str(tmp_path / "f.json")) == [1, 2]
    (tmp_path / "f.json").write_text(json.dumps({"other": 1}))
    with pytest.raises(ValueError, match="Cannot find"):
        plan.load_ocm_json(str(tmp_path / "f.json"))


def test_build_ocm_plan(tmp_path, capsys):
    graph_file = tmp_path / "Synth.graphml"
    ox.save_graphml(build_graph(), filepath=str(graph_file))
    (tmp_path / "ocm.json").write_text(json.dumps(STATIONS))
    plan.build_ocm_plan(str(graph_file), str(tmp_path / "ocm.json"), str(tmp_path / "out" / "existingplan.pkl"),
                        str(tmp_path / "mapping.csv"), max_distance_m=100)
    with open(tmp_path / "out" / "existingplan.pkl", "rb") as file:
        existing = pickle.load(file)
    assert [(s[0][0], s[1].tolist(), s[2]) for s in existing] == [(1001 + 5, [2, 3, 1], {}), (1008 + 10, [0, 4, 0], {})]
    assert existing[0][0][1]["x"] == pytest.approx(120.204)
    lines = (tmp_path / "mapping.csv").read_text().splitlines()
    assert lines[0].startswith("ocm_id,latitude,longitude,nearest_node") and len(lines) == 4
    out = capsys.readouterr().out
    assert "OCM station 4 is" in out and "Unique OSM station nodes: 2" in out


def test_existing_plan_skips_unknown_node(capsys):
    graph = build_graph()
    assert plan.build_existing_plan(graph, {12345: np.array([1, 0, 0])}) == []
    assert "node 12345 not found" in capsys.readouterr().out


def test_build_ocm_plan_main(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(plan, "build_ocm_plan", lambda **kwargs: calls.append(kwargs))
    run_main(plan, monkeypatch, ["--graph", "g", "--ocm", "o", "--output-pkl", "p", "--mapping-csv", "m",
                                 "--max-distance", "50"])
    assert calls == [dict(graph_file="g", ocm_file="o", output_pkl="p", mapping_csv="m", max_distance_m=50.0)]


# ------------------------------------------------------------------ build_pcrl_nodes


def street_graph(counts):
    graph = nx.MultiDiGraph()
    for i, count in enumerate(counts):
        graph.add_node(i, x=120.0 + i, y=23.0, street_count=count)
    return graph


def test_synthetic_demand_and_estate_price():
    graph = street_graph([1, 3, "bad", 5])
    demand = nodes.build_synthetic_demand(graph)
    assert demand == pytest.approx({0: 0.05, 1: 0.525, 2: 0.05, 3: 1.0})
    price = nodes.build_synthetic_estate_price(graph)
    assert price == pytest.approx({0: 0.1, 1: 0.55, 2: 0.1, 3: 1.0})
    flat = street_graph([2, 2])
    assert nodes.build_synthetic_demand(flat) == {0: 1.0, 1: 1.0}
    assert nodes.build_synthetic_estate_price(flat) == {0: 1.0, 1: 1.0}


def test_build_pcrl_nodes(tmp_path):
    graph_file = tmp_path / "g.graphml"
    ox.save_graphml(street_graph([1, 2, 3]), filepath=str(graph_file))
    out = tmp_path / "out" / "node_list.txt"
    nodes.build_pcrl_nodes(str(graph_file), str(out))
    import ast
    node_list = ast.literal_eval(out.read_text().splitlines()[0])  # the format evaluation_framework.py reads
    assert [n[0] for n in node_list] == [0, 1, 2]
    assert node_list[1][1]["demand"] == pytest.approx(0.525) and node_list[1][1]["private CS"] == 0
    assert node_list[2][1]["estate price"] == pytest.approx(1.0) and node_list[2][1]["osmid"] == 2


def test_build_pcrl_nodes_main(monkeypatch):
    calls = []
    monkeypatch.setattr(nodes, "build_pcrl_nodes", lambda **kwargs: calls.append(kwargs))
    run_main(nodes, monkeypatch, ["--graph", "g", "--output", "o"])
    assert calls == [dict(graph_file="g", output_file="o")]


# ------------------------------------------------------------------ build_osm_graph


def test_build_osm_graph_download(monkeypatch, tmp_path, capsys):
    graph = build_graph()
    del graph.edges[1000, 1005, 0]["length"]
    monkeypatch.setattr(ox, "graph_from_place", lambda place, **kwargs: graph)
    osm.build_osm_graph("Synth, Taiwan", "Synth", output_dir=str(tmp_path))
    saved = ox.load_graphml(str(tmp_path / "Synth" / "Synth.graphml"))
    assert len(saved.nodes) == len(graph.nodes)
    assert "WARNING: 1 edges do not have length." in capsys.readouterr().out


def test_build_osm_graph_rejects_node_without_coordinates(monkeypatch, tmp_path):
    graph = build_graph()
    graph.add_node(5)
    monkeypatch.setattr(ox, "graph_from_place", lambda place, **kwargs: graph)
    with pytest.raises(ValueError, match="Node 5 is missing"):
        osm.build_osm_graph("Synth", "Synth", output_dir=str(tmp_path))


def test_build_osm_graph_from_larger_graph(monkeypatch, tmp_path):
    import geopandas as gpd
    big = tmp_path / "big.graphml"
    ox.save_graphml(build_graph(), filepath=str(big))
    area = box(120.1995, 22.9995, 120.2085, 23.0085)  # the lower left 3 x 3 grid nodes
    monkeypatch.setattr(ox, "geocode_to_gdf", lambda place: gpd.GeoDataFrame(geometry=[area], crs="epsg:4326"))
    osm.build_osm_graph("Part", "Part", output_dir=str(tmp_path), from_graph=str(big))
    part = ox.load_graphml(str(tmp_path / "Part" / "Part.graphml"))
    assert sorted(part.nodes) == [1000, 1001, 1002, 1005, 1006, 1007, 1010, 1011, 1012]


def test_build_osm_graph_main(monkeypatch):
    calls = []
    monkeypatch.setattr(osm, "build_osm_graph", lambda **kwargs: calls.append(kwargs))
    run_main(osm, monkeypatch, ["--place", "P", "--location", "L", "--from-graph", "G"])
    assert calls == [dict(place="P", location="L", output_dir="Graph", from_graph="G")]


# ------------------------------------------------------------------ original PCRL scripts (load_graph, nodes_preparation)


def test_grid_cells_of_nodes():
    grids = grid.grid_preparation(grid.rec)
    assert len(grids) == grid.n_x * grid.n_y
    inside = (1, dict(x=9.61, y=52.31))
    grid.grid_location(inside, grids)
    assert 0 <= inside[1]["row"] < grid.n_x and 0 <= inside[1]["column"] < grid.n_y
    outside = (2, dict(x=0.0, y=0.0))
    grid.grid_location(outside, grids)
    assert (outside[1]["row"], outside[1]["column"]) == (None, None)


def test_load_graph_script(monkeypatch, tmp_path, capsys):
    """__main__ of load_graph.py with a fake download, writing ../Graph/Toy_Example below tmp_path"""
    (tmp_path / "Graph" / "Toy_Example").mkdir(parents=True)
    (tmp_path / "work").mkdir()
    monkeypatch.chdir(tmp_path / "work")
    graph = nx.MultiDiGraph(crs="epsg:4326")
    graph.add_node(1, x=9.61, y=52.31)
    graph.add_node(2, x=9.90, y=52.45)
    monkeypatch.setattr(ox, "graph_from_address", lambda *args, **kwargs: graph)
    runpy.run_path(os.path.join(PRE, "load_graph.py"), run_name="__main__")
    with open(tmp_path / "Graph" / "Toy_Example" / "grid_density_Toy_Example.pkl", "rb") as file:
        assert pickle.load(file).sum() == 2
    assert "Grid density is calculated." in capsys.readouterr().out


def test_node_preparation_functions(monkeypatch):
    monkeypatch.setattr(prep, "grid_density", np.ones((prep.n_x, prep.n_y)) * 2, raising=False)
    monkeypatch.setattr(prep, "demand_min", 0.05, raising=False)
    matrix = np.ones((prep.n_x, prep.n_y)) * 10
    matrix[0][0] = 1000
    prep.grid_density[1][1] = 0
    result = prep.modify_demand_dict(matrix)
    assert result.max() == 1 and result[1][1] == pytest.approx(0.05 / 250)
    monkeypatch.setattr(prep, "demand_matrix", result, raising=False)
    with_cell = (1, dict(row=0, column=0))
    prep.charging_demand(with_cell)
    assert with_cell[1]["demand"] == 1
    no_cell = (2, dict(row=None))
    prep.charging_demand(no_cell)
    assert no_cell[1]["demand"] == pytest.approx(np.mean(result))
    near = [(1, dict(x=120.0, y=23.0, **{"private CS": 0.5, "estate price": 2.0})), (2, dict(x=120.001, y=23.0)),
            (3, dict(x=121.0, y=23.0))]
    bound = prep.social_efficiency_upper_bound(near[0], near)
    assert near[0][1]["I1_max"] == 2
    assert bound == pytest.approx(prep.ef.my_lambda * 2 * (1 - 0.05) / 100 / 2.0)

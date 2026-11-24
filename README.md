# PCRL
# Reinforcement Learning-based Placement of Charging Stations in Urban Road Networks

<p align="center">
<img src="https://github.com/frantz03/PCRL/blob/main/RL_Dresden_mini.png" width="500">
</p>


# Description
This repository is based on the implementation of the project "Reinforcement Learning-based Placement of Charging
Stations in Urban Road Networks" by Leonie von Wahl, Nicolas Tempelmeier, Ashutosh Sao and Elena Demidova. A Deep Q
Network (DQN) learns where to build charging stations and how many chargers to install in a road network.

This version applies the method to **Tainan** and **Taipei** (OpenStreetMap road networks, existing stations from
Open Charge Map) in two stages:

1. **Training**: DQN models trained on Tainan.
2. **Generalisation and fine-tuning**: the Tainan model is applied to Taipei, without any Taipei training
   (zero-shot) and with model-guided beam search on the Taipei map.

The full technical documentation (Traditional Chinese, all settings, parameters and results) is
`README_for_doc.md`; the model card (performance, parameters and hyperparameters, training / inference time, hardware and software
requirements, user manual) is `MODEL_CARD.md`.


# Installation
The environment (Python 3.12.16, stable-baselines3 2.9.2, gymnasium 1.4.0, torch 2.15.0 CPU, osmnx 2.1.1; no known
vulnerabilities) is packaged as a Docker image; the files are in `docker/`.

```bash
docker build -f docker/Dockerfile -t pcrl:2.0 .        # from the repository root
docker run --rm pcrl:2.0                                # prints the package versions
docker run --rm -it -v "$PWD":/workspace/PCRL -u "$(id -u):$(id -g)" pcrl:2.0 python -m transfer.evaluate --help
```

Image with the trained model weights (`docker/Dockerfile.model`, built on `pcrl:2.0`; the weights are git-ignored, so
build it from a working copy that holds them):

```bash
docker build -f docker/Dockerfile.model -t pcrl:2.0-model .
docker run --rm pcrl:2.0-model                                # package versions and sha256 of both model files
docker run --rm pcrl:2.0-model python -m transfer.evaluate --city Taipei_Daan --policy model \
    --model runs/source_Tainan/best_source.zip --label zeroshot --eval-seed 1 --csv /tmp/zeroshot.csv
```

The models were trained and the reported results produced in the legacy environment (Python 3.8.20,
stable-baselines3 1.5.0, torch 1.12.0, gym 0.21.0, osmnx 1.2.1), kept in `docker/legacy-py38/` (`pcrl:1.0`) for exact
reproduction. The code runs in both (`gym_compat.py`); the trained models load unchanged in the new environment and give
the same plans (`README_for_doc.md` section 11.10). The scripts in `scripts/` use the new environment by default;
set `PY=$HOME/anaconda3/envs/PCRL/bin/python` to run them in the legacy one (training is reproduced bit for bit only
in the environment it was run in).

The Open Charge Map API key (needed only to download stations, preprocessing step 3) is kept in `docker/.env`
(`OCM_API_KEY=...`, template `docker/.env.example`). The file is git-ignored and not copied into the image; pass it
with `docker run --env-file docker/.env ...`. Outside Docker, `Preprocessing/download_ocm.py` reads it directly.

Without Docker, create a conda environment from the same file:

```bash
conda create -n PCRL-py312 python=3.12
conda activate PCRL-py312
pip install -r docker/requirements-pcrl.txt
```

Models are always loaded on the CPU (`gym_compat.load_dqn()`).


# Preprocessing (Tainan / Taipei)
The data in `Graph/<city>/` are built from OpenStreetMap and Open Charge Map with the scripts in `Preprocessing/`.
Run them from the repository root (osmnx caches downloads in `cache/`):

```bash
CITY=Taipei; PLACE="Taipei, Taiwan"; LC=taipei     # Tainan: CITY=Tainan; PLACE="Tainan, Taiwan"; LC=tainan

# 1. OSM driving network -> Graph/$CITY/$CITY.graphml (needs internet)
python Preprocessing/build_osm_graph.py --place "$PLACE" --location $CITY --output-dir Graph
# 2. PCRL node list (coordinates, street_count, demand, estate price) -> node_list_$CITY.txt
python Preprocessing/build_pcrl_nodes.py --graph Graph/$CITY/$CITY.graphml --output Graph/$CITY/node_list_$CITY.txt
# 3. Charging stations from Open Charge Map -> ocm_$LC.json (needs internet; API key from docker/.env)
python Preprocessing/download_ocm.py --place "$PLACE" --output Graph/$CITY/ocm_$LC.json
# 4. Map OCM stations to the nearest graph node -> existingplan_$CITY.pkl, ocm_osm_mapping_$CITY.csv
python Preprocessing/build_ocm_plan.py --graph Graph/$CITY/$CITY.graphml --ocm Graph/$CITY/ocm_$LC.json \
    --output-pkl Graph/$CITY/existingplan_$CITY.pkl --mapping-csv Graph/$CITY/ocm_osm_mapping_$CITY.csv \
    --max-distance 100000
```

OSM and OCM are updated continuously, so steps 1 and 3 return newer data on every download. To reproduce our results,
keep the files in `Graph/<city>/` and rerun only steps 2 and 4, which reproduce them bit for bit.


# Training
Original PCRL model on Tainan (environment `env_plus.py`, observation of 8 values per node):

```bash
python reinforcement.py --city Tainan          # DQN training -> tmp_Tainan/best_model_Tainan_<steps>.zip
python model_eval.py --city Tainan             # charging plan of the model -> Results/Tainan/plan_RL.pkl, nodes_RL.txt
python test_solution.py --city Tainan          # travel time, waiting time, budget of the plan
python Visualisierung/visualise.py --city Tainan
```

`--city` is the folder name in `Graph/` (default Tainan). Other options: `reinforcement.py --steps --learning-starts
--check-freq --seed --log-dir`, `model_eval.py --model --output` (default model: the `best_model_<city>_<steps>.zip`
with the most steps in `tmp_<city>/`), `test_solution.py --plan-dir`, `visualise.py --plan --output`. Example, 1,000
training steps directly on Taipei: `python reinforcement.py --city Taipei --steps 1000 --learning-starts 200 --check-freq 100`.
`--metrics-freq n` (default 50) prints the average travel and waiting time of the current plan every n steps.

Small demo map, Tainan West Central District (`Graph/Tainan_WestCentral/`, 652 nodes, 2 existing stations; the whole
pipeline takes about 2 minutes): `--city Tainan_WestCentral`. Its road network is cut out of the Tainan graph with
`Preprocessing/build_osm_graph.py --from-graph Graph/Tainan/Tainan.graphml` (no Overpass download needed). It is part
of Tainan, so it only demonstrates the pipeline. For a generalisation demo use Taipei Da'an District
(`Graph/Taipei_Daan/`, 1,046 nodes, 4 existing stations, cut from the Taipei graph): zero-shot + beam search take seconds
per seed (`README_for_doc.md` section 10.5).

City-independent source model on Tainan (environment `transfer/env.py`, fixed observation of 324 values, so the same
network runs on every city):

```bash
scripts/run_generalization.sh source    # 20,000 training steps -> runs/source_Tainan/
scripts/run_generalization.sh select    # best checkpoint -> runs/source_Tainan/best_source.zip
scripts/run_generalization.sh service   # Tainan travel / waiting time of every plan -> Results/Tainan/metrics.csv
python -m analysis.data_maps            # road network and existing stations of Tainan / Taipei -> Images/Result_Plots/Data/
python -m analysis.training_plots       # training curves, Tainan service metrics -> Images/Result_Plots/Training/
```


# Generalisation and fine-tuning (Taipei)
```bash
scripts/run_generalization.sh baselines  # Tainan model on Taipei (zero-shot) and random policy
scripts/run_generalization.sh analyze    # -> Results/Transfer/generalization_summary.txt, figure

scripts/run_beam_search.sh validate      # beam search B = k = 1 (reproduces the zero-shot episodes exactly)
scripts/run_beam_search.sh search        # beam search B = 4, k = 2 on episode seeds 1-5
scripts/run_beam_search.sh analyze       # -> Results/Transfer/beam_summary.txt, Images/Result_Plots/Transfer/beam/
                                         #    (incl. maps of the stations before / after fine-tuning, map_seed<seed>.png)
```

Beam search (`transfer/beam_search.py`): in every step each of the B best plans tries the k actions with the highest
Q-values of the Tainan model; the plans with the highest "score so far + max Q" are kept. One of the B places always
follows the model's greedy episode, the other B - 1 explore alternatives. Every run is started in its own
memory-limited cgroup (`systemd-run`, default 12 GB) and stops itself before the machine runs short of memory.

Service metrics of any saved plan: `python -m analysis.plan_metrics --city Taipei --plan <folder with plan_RL.pkl>`.

Training and fine-tuning logs for trackio: `python -m analysis.trackio_logs` writes CSV files (a `step` column, an
optional ISO `timestamp`, numeric metrics) to `Results/Trackio/training/` and `Results/Trackio/finetuning/`; each file is
one run for `trackio.import_csv(path, project=...)`.


# Results
Service target: average travel time and average waiting time of the charging plan below 15 minutes
(`test_solution.py` definitions).

**Tainan** (training city):

| | Existing stations | Original PCRL model | Source model |
|---|---|---|---|
| Charging stations | 17 | 755 | 405 |
| Average travel time (min) | 10.806 | **1.771** | **2.178** |
| Average waiting time (min) | 7.457 | **0.365** | **0.098** |

Both trained models meet the 15-minute target by a wide margin.

**Taipei** (Tainan model, no Taipei training; episode seeds 1-5, mean):

| | Existing stations | Before fine-tuning (zero-shot) | After fine-tuning (beam search B = 4, k = 2) |
|---|---|---|---|
| Score improvement | 0 | 2.3239 | **2.4012 (+3.3%)**, 4 of 5 seeds better, none worse |
| Average travel time (min) | 3.861 | 1.199 | **1.172** |
| Average waiting time (min) | 3.030 | 0.110 | 0.111 |
| Max waiting time (min) | 26.69 | 0.498 | **0.460 (-7.7%)** |

Without any Taipei training the Tainan model is 85% better than a random policy. Details: `README_for_doc.md`
sections 7 and 10, `Results/Tainan/metrics.csv`, `Results/Transfer/*_summary.txt`, figures in `Images/Result_Plots/`.

Security: the code is scanned with SonarQube (`sonar-project.properties`) and the dependencies with pip-audit
(`scripts/dependency_audit.sh`, imported into SonarQube): 0 issues, 0 vulnerabilities, 0 security hotspots, ratings
A / A / A, quality gate passed (`README_for_doc.md` sections 11.7, 11.9, 11.10, 11.11). Software bill of materials of
the model image (syft, SPDX 2.3 JSON, 175 components): `scripts/sbom.sh` -> `Results/Security/充電站選址服務_SBOM.json`
(section 11.12).

Tests: 140 unit tests on a small synthetic city (`tests/`, about 15 seconds), passing in both environments; 95.9%
coverage (`README_for_doc.md` sections 11.8, 11.11).

```
python -m pip install -r docker/requirements-test.txt
python -m pytest --cov --cov-report=xml
```


# Folder structure
All commands are run from the repository root; the modules in `transfer/` and `analysis/` are started with
`python -m <folder>.<module>`.

```
PCRL/
├── evaluation_framework.py   utility model (cost, coverage, waiting time, score), routing and caches
├── env_plus.py               original environment StationPlacement
├── gym_compat.py             gym 0.21 / gymnasium compatibility (both environments)
├── MODEL_CARD.md             model card
├── reinforcement.py          original PCRL: DQN training
├── model_eval.py             original PCRL: charging plan of a model
├── test_solution.py          original PCRL: travel time, waiting time, budget of a plan
├── Visualisierung/           map of a charging plan
├── Preprocessing/            OSM / OCM preprocessing
├── transfer/                 city-independent model
│   ├── env.py                    TransferStationPlacement (324-value observation), make_env()
│   ├── train.py                  training of the source model
│   ├── evaluate.py               greedy episodes of a model / random policy -> CSV, plan
│   └── beam_search.py            model-guided beam search
├── analysis/                 statistics and figures
│   ├── data_maps.py              road network and existing stations (after the OCM mapping)
│   ├── training_plots.py         training curves, Tainan service metrics
│   ├── generalization.py         zero-shot vs random policy
│   ├── beam.py                   beam search vs zero-shot, search progress, service metrics
│   ├── plan_maps.py              maps of the Taipei stations before / after fine-tuning
│   ├── plan_metrics.py           service metrics of saved plans
│   └── trackio_logs.py           training / fine-tuning logs -> trackio CSV files
├── scripts/                  run_generalization.sh, run_beam_search.sh, dependency_audit.sh / .py (dependency vulnerability scan),
│                             sbom.sh (SBOM of the model image)
├── tests/                    unit tests (pytest; settings in pytest.ini, .coveragerc)
├── docker/                   Dockerfile, Dockerfile.model (+ weights), requirements-pcrl.txt, requirements-test.txt, .env(.example); legacy-py38/
├── sonar-project.properties  SonarQube analysis settings
├── Graph/<city>/             input data
├── tmp_Tainan/               model and training log of reinforcement.py (tmp_Toy_Example/: original example)
├── runs/source_Tainan/       source model, checkpoints, monitor.csv, config.json
├── Results/                  Tainan/ plans and service metrics; Transfer/ result CSVs, summaries, plans/; Trackio/ logs;
│                             Security/ dependency scan, SBOM
├── Images/Result_Plots/      figures: Training/, Tainan/, Transfer/
└── logs/                     training/, beam/
```

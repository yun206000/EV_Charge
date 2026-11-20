# PCRL 技術文件：以強化學習進行電動車充電站佈建規劃，及跨城市泛化／微調驗證

> 本文件說明 PCRL 專案（`/home/angel030880/test/PCRL`）的系統架構、資料、效用模型、強化學習環境、模型結構、超參數、
> 訓練／評估流程與實驗結果。所有數字均取自本 repo 的程式碼、設定檔與實驗紀錄（`Results/`、`runs/`、`logs/`）。
> 最後更新：2026-11-20。模型的摘要說明（效能、超參數、時間、軟硬體需求、使用手冊）見 `MODEL_CARD.md`。

---

## 目錄

1. [專案概述](#1-專案概述)
2. [系統架構與目錄結構](#2-系統架構與目錄結構)
3. [執行環境（Conda / Docker）](#3-執行環境conda--docker)
4. [資料與前處理](#4-資料與前處理)
5. [效用模型（Utility Model）](#5-效用模型utility-model)
6. [強化學習環境 StationPlacement](#6-強化學習環境-stationplacement)
7. [模型訓練（預訓練階段）](#7-模型訓練預訓練階段)
8. [跨城市遷移版本（transfer/env.py）](#8-跨城市遷移版本transferenvpy)
9. [程式模組與參數一覽](#9-程式模組與參數一覽)
10. [泛化與微調驗證（第二階段）](#10-泛化與微調驗證第二階段)
11. [工程紀錄](#11-工程紀錄)
12. [影像紀錄與輸出檔清單](#12-影像紀錄與輸出檔清單)
13. [重現步驟](#13-重現步驟)
14. [後續規劃](#14-後續規劃)
15. [附錄](#15-附錄)

---

## 1. 專案概述

### 1.1 目標

PCRL（Placement of Charging stations with Reinforcement Learning）以 Deep Q-Network（DQN）強化學習，在城市道路網路上
決定電動車充電站的位置與充電樁配置。原始方法出自 von Wahl, Tempelmeier, Sao, Demidova,
*Reinforcement Learning-based Placement of Charging Stations in Urban Road Networks*（原始程式碼 `github.com/frantz03/PCRL`，
引用資訊見 `CITATION.cff`）。本 repo 在原始方法上做了以下調整：

| 項目 | 原始 PCRL | 本專案 |
|---|---|---|
| 城市 | Hannover、Dresden 等（未公開資料）、Toy_Example（紐約假資料） | **台南（Tainan）**、**台北（Taipei）**，OSM 路網 + Open Charge Map 既有充電站 |
| 路徑計算 | `networkx.shortest_path_length`（每對點一次） | scipy C 實作 Dijkstra + CSR 稀疏矩陣 + KD-tree，結果快取（`evaluation_framework.py`） |
| 覆蓋計算 | 雙層 Python 迴圈 | 向量化 haversine（`coverage_vector`，結果與原版逐位元相同） |
| 觀測 | 每個節點 8 個值，維度依城市而變 | 另新增固定 324 維、依城市正規化的觀測（`transfer/env.py`），可跨城市使用 |
| 實驗 | 單城市訓練 | 跨城市泛化（zero-shot）與微調驗證（模型引導的 beam search） |

**服務目標**：充電規劃的**平均移動時間（旅行時間）與平均等待時間皆低於 15 分鐘**（`test_solution.py` 定義，5.5）。
台南與台北的結果見 7.2、7.4、10.1、10.3。

### 1.2 計畫內容與本專案的對應

本計畫包含兩個階段：

| 階段 | 內容 | 本文件 |
|---|---|---|
| 一、模型訓練 | 以台南資料（OSM 路網 + OCM 既有充電站）從零訓練 DQN 充電站佈建模型，產生充電規劃並評估 | 第 7 節 |
| 二、泛化與微調驗證 | 把台南訓練完成的模型導入台北資料：不經任何台北訓練直接使用（zero-shot）驗證泛化能力；再以模型引導的 beam search 在台北資料上微調充電規劃，比較微調前後的效能 | 第 8、10 節 |

計畫書對微調的定義：

> 本計畫將驗證並交付訓練完成之 AI 模型泛化能力。為證明本計畫所建置之預測模組具備微調能力，將其他資料導入模組，
> 對預訓練模型進行局部參數更新訓練。實務上將記錄模型的微調過程，並交付相關的影像紀錄，藉此證明模型可在其他資料集上
> 進行微調，且微調後效能有顯著提升。

對應到本專案：

| 計畫書 | 本專案 |
|---|---|
| 預測模組 | PCRL 的 DQN 模型（Q-network，輸入固定 324 維狀態，輸出 5 個動作的 Q 值） |
| 訓練完成之 AI 模型（預訓練模型） | 以台南資料訓練的來源模型 `runs/source_Tainan/best_source.zip`（第 7 節）；另有原始 PCRL 台南模型 `tmp_Tainan/best_model_Tainan_11200.zip` |
| 其他資料 | 台北資料 `Graph/Taipei/`（12,030 個路網節點、29 個既有充電站） |
| 泛化能力 | 台南模型不經訓練直接用於台北（zero-shot），與隨機策略比較（10.2） |
| 微調 | 以台南模型的 Q 值引導 beam search，在台北地圖上搜尋更好的充電規劃（10.3） |
| 微調前 / 後 | 微調前 = zero-shot 貪婪規劃；微調後 = beam search 規劃；同一組 episode 種子 1–5 配對比較 |
| 微調過程紀錄 | 每次搜尋的逐步 log（`logs/beam/`）、結果 CSV、規劃檔（`Results/Transfer/plans/`） |
| 服務目標 | 平均移動時間、平均等待時間 < 15 分鐘：台南（7.2、7.4、10.1）、台北微調前後（10.3） |
| 影像紀錄 | 台南、台北路網與既有充電站分布圖（`Images/Result_Plots/Data/`）；訓練曲線、台南服務指標、台南充電站地圖（`Images/Result_Plots/Training/`、`Tainan/`）；泛化比較圖、微調前後比較、搜尋過程、服務指標、微調前後充電站分布圖（`Images/Result_Plots/Transfer/`） |

---

## 2. 系統架構與目錄結構

### 2.1 資料流程

```
OpenStreetMap ──build_osm_graph.py──> Graph/<city>/<city>.graphml          (道路網路)
       │                                     │
       └────────build_pcrl_nodes.py──────────┴─> Graph/<city>/node_list_<city>.txt   (節點：需求、地價…)
Open Charge Map ──download_ocm.py──> ocm_<city>.json
       └─────────build_ocm_plan.py──────────────> Graph/<city>/existingplan_<city>.pkl (既有充電站)

Graph/<city>/* ──> env_plus.StationPlacement / transfer.env.TransferStationPlacement (RL 環境)
                         │
  一、訓練   reinforcement.py / transfer/train.py（台南，DQN 訓練）──> 模型 .zip
                         │
  二、泛化   transfer/evaluate.py（台北，貪婪 episode = zero-shot）──> 規劃、Results/Transfer/generalization.csv
  二、微調   transfer/beam_search.py（台北，模型引導的 beam search）──> 規劃、Results/Transfer/beam.csv
                         │
  分析       test_solution.py / analysis/plan_metrics.py（服務指標）、visualise.py（地圖）、
             analysis/training_plots.py、analysis/generalization.py、analysis/beam.py（統計與影像紀錄）
```

### 2.2 目錄結構

原則：根目錄保留原始 PCRL 的程式（第一階段的原始模型），本專案新增的跨城市程式放在 `transfer/`，統計與影像紀錄放在
`analysis/`，流程腳本放在 `scripts/`，容器相關檔案放在 `docker/`。所有程式都在 repo 根目錄執行
（`transfer/`、`analysis/` 以 `python -m <資料夾>.<模組>` 執行）。

```
PCRL/
├── evaluation_framework.py   效用模型：成本、覆蓋、等候時間、分數；路徑計算與快取；全域常數
├── env_plus.py               原始 gym 環境 StationPlacement（觀測 8N+1 維）
├── gym_compat.py             gym 0.21 / gymnasium 介面相容（新舊兩個環境都能執行，3.1）
├── MODEL_CARD.md             Model card：效能、參數與超參數、訓練／推論時間、軟硬體需求、使用手冊（11.11）
├── reinforcement.py          原始 PCRL：DQN 訓練
├── model_eval.py             原始 PCRL：產生充電規劃
├── test_solution.py          原始 PCRL：規劃的評估指標
├── Visualisierung/           充電規劃地圖（visualise.py）
├── Preprocessing/            OSM / OCM 下載與轉換
├── transfer/                 跨城市模型（本專案）
│   ├── env.py                    TransferStationPlacement（固定 324 維觀測）、make_env()
│   ├── train.py                  來源模型訓練（台南）
│   ├── evaluate.py               貪婪 episode 評估：模型 / 隨機策略 → CSV、規劃
│   └── beam_search.py            模型引導的 beam search（台北微調）
├── analysis/                 統計與影像紀錄
│   ├── data_maps.py              台南、台北路網與既有充電站（OCM 對應後）分布圖
│   ├── training_plots.py         訓練曲線（原始模型、來源模型）、台南服務指標
│   ├── generalization.py         泛化統計與圖
│   ├── beam.py                   微調前後比較、搜尋過程、服務指標
│   ├── plan_maps.py              台北微調前後充電站分布圖
│   ├── plan_metrics.py           任一已存規劃的服務指標
│   └── trackio_logs.py           訓練與微調紀錄 → trackio 可匯入的 CSV（11.11）
├── scripts/                  run_generalization.sh、run_beam_search.sh（完整流程）；dependency_audit.sh／.py（相依套件弱點掃描，11.9）
├── tests/                    單元測試（pytest，合成小城市資料，11.8）；設定檔 pytest.ini、.coveragerc
├── docker/                   Dockerfile、Dockerfile.dockerignore、requirements-pcrl.txt、requirements-test.txt（測試套件）、
│                             Dockerfile.model（環境＋模型權重映像 pcrl:2.0-model，11.11）、
│                             .env（金鑰，不進 git）；legacy-py38/（舊環境：訓練模型與報告數值的產生環境）
├── Graph/<city>/             輸入資料（graphml、節點、既有充電站）；Toy_Example 為原始 PCRL 範例資料
├── cache/                    osmnx 下載快取
├── tmp_Tainan/               原始 reinforcement.py 的台南模型與訓練紀錄（tmp_Toy_Example/ 為原始範例）
├── runs/source_Tainan/       來源模型：best_source.zip、checkpoints/、monitor.csv、config.json
├── Results/
│   ├── Tainan/                   台南充電規劃（原始模型、source_model/ 來源模型）與服務指標 metrics.csv
│   ├── Transfer/                 泛化與微調的結果 CSV、摘要；plans/ 各規劃
│   └── Trackio/                  trackio 格式的訓練（training/）與微調（finetuning/）紀錄
├── Images/Result_Plots/      影像紀錄：Data/、Training/、Tainan/、Transfer/
└── logs/                     training/（訓練與評估 log）、beam/（beam search 逐步 log）
```

---

## 3. 執行環境（Conda / Docker）

### 3.1 軟體版本

專案有兩個環境，程式碼相同（`gym_compat.py` 處理 gym 0.21 與 gymnasium 的介面差異），兩者都通過全部單元測試：

| | 目前環境（2026-11-13 起） | 舊環境 |
|---|---|---|
| conda 環境 / Docker 映像 | `PCRL-py312` / `pcrl:2.0`（`docker/Dockerfile`） | `PCRL` / `pcrl:1.0`（`docker/legacy-py38/Dockerfile`） |
| Python | 3.12.16 | 3.8.20 |
| stable-baselines3 / RL 介面 | 2.9.2 / gymnasium 1.4.0 | 1.5.0 / gym 0.21.0 |
| torch | 2.15.0（CPU 版） | 1.12.0（+cu102 wheel，只用 CPU） |
| osmnx / numpy / pandas | 2.1.1 / 2.5.4 / 3.1.1 | 1.2.1 / 1.24.3 / 1.4.3 |
| 已知弱點（pip-audit） | **0** | 120 個（15 個套件，11.9） |
| 用途 | 預設環境（`scripts/run_*.sh` 的 `PY` 預設值）：訓練、評估、前處理；舊模型直接載入，結果與舊環境相同（11.10） | 訓練模型與本文件數值的產生環境，保留以便完整重現（`PY=$HOME/anaconda3/envs/PCRL/bin/python`） |

舊環境套件明細（conda 環境 `PCRL`）：

| 套件 | 版本 | 用途 |
|---|---|---|
| Python | 3.8.20 | |
| stable-baselines3 | 1.5.0 | DQN 實作 |
| torch | 1.12.0（+cu102 wheel） | 神經網路；**只使用 CPU**（見 3.3） |
| gym | 0.21.0 | RL 環境介面 |
| osmnx | 1.2.1 | 讀取 OSM 路網 |
| networkx | 2.8.4 | 圖結構 |
| numpy | 1.24.3 | 注意：主機 `pip list` 顯示 1.23.0，但實際載入的是 conda `numpy-base` 1.24.3 的檔案；模型檔以 1.24.3 序列化，用 1.23.0 載入會失敗（`__randomstate_ctor() takes from 0 to 1 positional arguments`），Docker 映像因此固定 1.24.3 |
| scipy | 1.10.1（conda） | Dijkstra（`scipy.sparse.csgraph`）、KD-tree、統計檢定 |
| pandas | 1.4.3 | |
| matplotlib | 3.5.2 | 繪圖 |
| geopandas / shapely / pyproj / fiona / rtree | 0.11.0 / 1.8.2 / 3.3.1 / 1.8.21 / 1.0.0 | 地理資料 |
| scikit-learn | 1.3.0（conda） | |

完整清單：目前環境 `docker/requirements-pcrl.txt`；舊環境 `docker/legacy-py38/environment.yml`（conda 套件）與 `docker/legacy-py38/requirements-pcrl.txt`（pip 套件）。

### 3.2 硬體

- CPU：28 執行緒；記憶體 31 GB。
- GPU：RTX 4090，但舊環境的 torch 1.12 + CUDA 10.2 不支援 sm_89（錯誤訊息 `no kernel image is available`），
  因此所有訓練與推論都在 CPU 上執行，載入模型一律指定 CPU（`gym_compat.load_dqn()`）；目前環境沿用 CPU 版 torch。
- 網路很小（15 萬個參數），瓶頸在環境模擬（每步約 1–2 秒），GPU 不會帶來明顯加速。

### 3.3 Docker 封裝

檔案（`docker/`）：`Dockerfile`、`Dockerfile.dockerignore`、`requirements-pcrl.txt`、`requirements-test.txt`、`.env.example`；舊環境在
`docker/legacy-py38/`。建置時以 repo 根目錄為 build context（`-f docker/Dockerfile`），BuildKit 會自動使用與 Dockerfile 同名的 `.dockerignore`。

- 基底映像：`python:3.12.16-slim-bookworm`（舊環境：`continuumio/miniconda3:24.9.2-0`）。
- 建置步驟：
  1. 安裝 `pip==26.3 setuptools==84.2.1 wheel==0.48.0`，再以 `pip install -r docker/requirements-pcrl.txt` 安裝全部套件
     （torch 取自 PyTorch 的 CPU 套件庫）。
     舊環境：`conda env create -f docker/legacy-py38/environment.yml`，再以舊版 pip 工具且關閉 build isolation 安裝
     `docker/legacy-py38/requirements-pcrl.txt`（gym 0.21.0 的 `setup.py` 不符合新版 setuptools 規範）。
  2. 建立非 root 使用者 `pcrl`（uid 1000）作為預設使用者。
  3. 逐一列出要複製的程式碼與 `Graph/` 輸入資料（`*.py`、`transfer/`、`analysis/`、`Preprocessing/`、`Visualisierung/`、
     `scripts/`、`Graph/`）到 `/workspace/PCRL`，檔案屬於 root、對執行使用者唯讀；`runs/`、`logs/`、`Results/`、
     `Images/`、`tmp_*` 等輸出不放進映像（另有 `docker/Dockerfile.dockerignore` 把關），執行時以 volume 掛載。
- 環境變數：`MPLBACKEND=Agg`（無螢幕繪圖）、`PYTHONUNBUFFERED=1`（即時輸出 log）、`PY=python`（`scripts/run_*.sh` 使用容器內的
  Python；舊環境映像未設定，執行腳本時加 `-e PY=python`）。
- 金鑰：OCM API key 放在 `docker/.env`（格式見 `docker/.env.example`：`OCM_API_KEY=...`），執行時以
  `--env-file docker/.env` 傳入容器。`docker/.env` 不進 git（`.gitignore`）也不會複製進映像（`Dockerfile.dockerignore`）。

```bash
# 建置（在 repo 根目錄執行）；舊環境：docker build -f docker/legacy-py38/Dockerfile -t pcrl:1.0 .
docker build -f docker/Dockerfile -t pcrl:2.0 .

# 檢查環境（預設 CMD 會印出各套件版本）
docker run --rm pcrl:2.0

# 掛載整個 repo，輸出直接寫回主機，並以主機使用者身分寫檔
docker run --rm -it -v "$PWD":/workspace/PCRL -u "$(id -u):$(id -g)" pcrl:2.0 \
    python -m transfer.evaluate --city Taipei --policy model --model runs/source_Tainan/best_source.zip \
    --label docker_check --seed 0 --eval-seed 1 --csv Results/Transfer/docker_check.csv

# 長時間實驗（背景執行）
docker run -d --name pcrl_generalization -v "$PWD":/workspace/PCRL -u "$(id -u):$(id -g)" pcrl:2.0 \
    scripts/run_generalization.sh all

# 前處理下載 OCM 充電站（金鑰由 docker/.env 傳入）
docker run --rm -it --env-file docker/.env -v "$PWD":/workspace/PCRL -u "$(id -u):$(id -g)" pcrl:2.0 \
    python Preprocessing/download_ocm.py --place "Taipei, Taiwan" --output Graph/Taipei/ocm_taipei.json
```

**含模型權重的映像 `pcrl:2.0-model`**（`docker/Dockerfile.model`，以 `pcrl:2.0` 為基底）：另外放入原始 PCRL 模型
`tmp_Tainan/best_model_Tainan_11200.zip`、來源模型 `runs/source_Tainan/best_source.zip`（含 `config.json`）與 `MODEL_CARD.md`，
不需掛載 repo 即可直接推論。模型權重不在 git 中（`*.zip`），需在含權重的工作目錄建置；build context 只含這幾個檔案
（`docker/Dockerfile.model.dockerignore`）。模型檔屬於 root、唯讀；所在資料夾屬於使用者 `pcrl`（`model_eval.py` 會在模型旁寫
評估 episode 的 `monitor.csv`）。驗證結果見 11.11。

```bash
docker build -f docker/Dockerfile.model -t pcrl:2.0-model .    # 先建好 pcrl:2.0
docker run --rm pcrl:2.0-model                                  # 套件版本與兩個模型檔的 sha256
docker run --rm pcrl:2.0-model python -m transfer.evaluate --city Taipei_Daan --policy model \
    --model runs/source_Tainan/best_source.zip --label zeroshot --eval-seed 1 --csv /tmp/zeroshot.csv
```

每個環境行程約佔 1–1.4 GB 記憶體，平行執行數（`JOBS`）需依主機記憶體調整。`scripts/run_beam_search.sh` 使用 `systemd-run --user --scope`（cgroup 記憶體上限），需在主機上執行。
映像約 1.7 GB（舊環境 3.0 GB，建置時需要約 6–8 GB 暫存空間）。

**驗證**（`pcrl:2.0`，2026-11-13）：台南來源模型在大安區的貪婪 episode（種子 1）improvement 1.8588917694329221、74 站、128 步，
與主機 `PCRL-py312` 逐位元相同（與舊環境的比較見 11.10）；金鑰傳入、非 root 使用者、程式檔唯讀等檢查同下表。

舊環境 `pcrl:1.0`（2026-11-09；2026-11-11 依新目錄結構與安全性修正重建後再次驗證）：

| 檢查 | 結果 |
|---|---|
| 套件版本 | Python 3.8.20、torch 1.12.0+cu102、stable-baselines3 1.5.0、gym 0.21.0、osmnx 1.2.1、numpy 1.24.3、scipy 1.10.1、pandas 1.4.3、networkx 2.8.4、matplotlib 3.5.2、scikit-learn 1.3.0、geopandas 0.11.0，與主機相同 |
| 端對端重現 | 容器內以台南來源模型在台北跑貪婪 episode（episode 種子 1）：improvement **2.4756452779578244**、403 站、688 步，與主機結果逐位元相同 |
| 金鑰傳入 | `--env-file docker/.env` 後容器內可讀到 `OCM_API_KEY`；映像中不含 `docker/.env` |
| 權限 | 預設使用者為 `pcrl`（uid 1000，非 root）；映像內程式檔不可被修改，工作目錄可建立輸出資料夾 |

---

## 4. 資料與前處理

### 4.1 前處理腳本（`Preprocessing/`）

| 腳本 | 功能 | 主要參數 |
|---|---|---|
| `build_osm_graph.py` | `ox.graph_from_place(place, network_type="drive", simplify=True, retain_all=False)` 下載可行車路網，存成 `Graph/<city>/<city>.graphml` | `--place "Taipei, Taiwan"`、`--location Taipei`、`--output-dir Graph` |
| `build_pcrl_nodes.py` | 由路網節點產生 PCRL 節點清單 `node_list_<city>.txt`（Python `repr`，以 `eval` 讀回） | graph 檔、輸出檔 |
| `download_ocm.py` | 由 Open Charge Map API 下載充電站（圓形範圍或 OSM place 多邊形） | `--place` 或 `--latitude/--longitude/--distance`、API key、`country_code="TW"` |
| `build_ocm_plan.py` | OCM 充電站對應到最近的 OSM 節點，彙整成既有充電規劃 `existingplan_<city>.pkl` | `--graph --ocm --output-pkl --mapping-csv --max-distance 100`（現有資料用 100000，見 4.3） |
| `load_graph.py`, `nodes_preparation.py` | 原始 PCRL 的格網密度／需求計算（Hannover 範例），本專案台南、台北未使用 | |

### 4.2 節點屬性（`node_list_<city>.txt`）

每個節點為 `(osm_id, {屬性})`：

| 屬性 | 說明 | 產生方式 |
|---|---|---|
| `x`, `y` | 經度、緯度 | OSM |
| `street_count` | 連接的街道數 | OSM |
| `demand` | 充電需求（0.05–1） | **合成資料**：`street_count` min-max 正規化後 `0.05 + 0.95 × norm`（`build_synthetic_demand`） |
| `estate price` | 地價代理值（0.1–1） | **合成資料**：`street_count` min-max 正規化後 `0.1 + 0.9 × norm`（`build_synthetic_estate_price`） |
| `private CS` | 私人充電樁數 | 無資料，全部設為 0 |
| `osmid` | OSM 節點 id | |

需求與地價皆由路口連接數（`street_count`）推估：連接數越多的路口，車流與土地價值越高。

### 4.3 既有充電站（`existingplan_<city>.pkl`）

- 每個充電站：`[ (node_id, node_attrs), np.array([n_7kW, n_22kW, n_50kW]), {} ]`。
- OCM 功率對應 PCRL 三種充電樁：≤ 11 kW → 7 kW；≤ 35 kW → 22 kW；其餘 → 50 kW。
- 對應最近 OSM 節點，同一節點的多個 OCM 站合併。所有 OCM 充電站都納入（`--max-distance 100000`，見 4.5）。
- 對應明細：`Graph/<city>/ocm_osm_mapping_<city>.csv`。
- 分布圖（`python -m analysis.data_maps`）：`Images/Result_Plots/Data/Tainan_network_stations.png`、`Taipei_network_stations.png`。藍點 = 既有充電站（畫在對應的路網節點上，大小 = 充電樁數），空心圓 + 虛線 = 距離對應節點超過 500 m 的 OCM 原始位置。
  - 台南：17 個 OCM 站 → 17 座既有充電站、86 支充電樁（50 kW 85 支、7 kW 1 支），全部在對應節點 332 m 以內。
  - 台北：33 個 OCM 站 → 29 座既有充電站、142 支充電樁（50 kW 136 支、7 kW 6 支）。其中 10 個 OCM 站位於台北市路網範圍外（距對應節點超過 500 m，最遠 9.5 km），對應到路網邊緣最近的節點；同一節點的站已合併。

### 4.4 資料集統計

| | 台南 Tainan | 台北 Taipei |
|---|---|---|
| 路網節點數（= RL 候選位置數 N） | 15,983 | 12,030 |
| 路網邊數 | 42,978 | 28,486 |
| 經度範圍 | 120.0532 – 120.6559 | 121.4632 – 121.6581 |
| 緯度範圍 | 22.8914 – 23.4088 | 24.9674 – 25.1918 |
| OCM 下載充電站 | 17 | 33 |
| 既有充電站（對應後） | 17 | 29 |
| 既有充電樁 7 / 22 / 50 kW | 1 / 0 / 85（共 86） | 6 / 0 / 136（共 142） |
| 平均需求 `demand` | 0.394 | 0.375 |
| 平均地價 `estate price` | 0.426 | 0.408 |
| 既有規劃的正規化分數 | 0.2333 | 0.2333 |
| 每個 episode 步數上限 N/2 | 7,992 | 6,015 |

**示範用小地圖：台南中西區（`Graph/Tainan_WestCentral/`，2026-11-12）**。整個流程（訓練 1,000 步 → 規劃 → 指標 → 地圖）
約 2 分鐘，適合示範。原本預定的安平區在 OCM 上沒有任何充電站（環境需要至少一座既有充電站做分數正規化），
因此改用相鄰、面積更小的中西區。

| | 台南中西區 Tainan_WestCentral |
|---|---|
| 面積 | 6.3 km² |
| 路網節點數 N / 邊數 | 652 / 1,846 |
| 經度範圍 / 緯度範圍 | 120.1658 – 120.2117 / 22.9813 – 23.0075 |
| 既有充電站（OCM） | 2 座，充電樁 7 / 22 / 50 kW：1 / 0 / 8 |
| 平均需求 / 平均地價 | 0.531 / 0.556 |
| 每個 episode 步數上限 N/2 | 326 |
| 路網來源 | 由 `Graph/Tainan/Tainan.graphml` 依中西區行政界線切出（`build_osm_graph.py --from-graph`，4.5） |
| 前處理 log | `logs/preprocessing/Tainan_WestCentral.log` |

中西區是台南的一部分（652 個節點、2 座既有充電站都在台南的訓練資料中），因此只用來示範流程，不用來評估泛化能力。

**示範泛化用的小地圖：台北大安區（`Graph/Taipei_Daan/`，2026-11-12）**。台北從未用於訓練或 checkpoint 選擇，
可示範台南來源模型 zero-shot 與 beam search（10.5）。

| | 台北大安區 Taipei_Daan |
|---|---|
| 面積 | 11.3 km² |
| 路網節點數 N / 邊數 | 1,046 / 2,148 |
| 經度範圍 / 緯度範圍 | 121.5218 – 121.5634 / 25.0099 – 25.0453 |
| 既有充電站（OCM） | 4 座，充電樁 7 / 22 / 50 kW：0 / 0 / 23 |
| 平均需求 / 平均地價 | 0.483 / 0.510 |
| 路網來源 | 由 `Graph/Taipei/Taipei.graphml` 依大安區行政界線切出（`build_osm_graph.py --from-graph`，4.5） |
| 前處理 log | `logs/preprocessing/Taipei_Daan.log` |

（既有規劃分數在兩市都是 0.2333，因為分數是以既有規劃本身做正規化，見 5.4。）

### 4.5 執行前處理（產生 `Graph/<city>/`）

在 repo 根目錄執行（`conda activate PCRL-py312` 或 Docker 容器內）。osmnx 的下載快取寫在根目錄的 `cache/`（已納入 git）。
兩個環境的前處理結果完全相同（11.10）。

| 步驟 | 腳本 | 輸入 | 輸出 | 需要網路 |
|---|---|---|---|---|
| 1 | `build_osm_graph.py` | OSM 地名 | `Graph/<city>/<city>.graphml` | 是（OSM） |
| 2 | `build_pcrl_nodes.py` | graphml | `Graph/<city>/node_list_<city>.txt` | 否 |
| 3 | `download_ocm.py` | 地名或中心點，OCM API key | `Graph/<city>/ocm_<city>.json`、`.summary.json` | 是（OCM API） |
| 4 | `build_ocm_plan.py` | graphml、OCM JSON | `Graph/<city>/existingplan_<city>.pkl`、`ocm_osm_mapping_<city>.csv` | 否 |

```bash
CITY=Tainan; PLACE="Tainan, Taiwan"; LC=tainan      # 台北：CITY=Taipei; PLACE="Taipei, Taiwan"; LC=taipei

# 1. 下載可行車路網（network_type="drive", simplify=True, retain_all=False）
python Preprocessing/build_osm_graph.py --place "$PLACE" --location $CITY --output-dir Graph

# 2. 節點清單：座標、street_count，合成的 demand / estate price，private CS = 0（4.2）
python Preprocessing/build_pcrl_nodes.py --graph Graph/$CITY/$CITY.graphml --output Graph/$CITY/node_list_$CITY.txt

# 3. 下載 OCM 充電站（--place 用與步驟 1 相同的地名：以地名多邊形的外框查詢，再用多邊形過濾）
#    OCM API key 自動從 docker/.env 讀取（見下方注意事項）
python Preprocessing/download_ocm.py --place "$PLACE" --output Graph/$CITY/ocm_$LC.json
#    或以中心點＋半徑：--latitude 25.04 --longitude 121.53 --distance 20
#    其他參數：--country-code TW（預設）、--status-type-id 50（只取營運中，-1 = 不過濾）、--max-results 100

# 4. OCM 充電站對應到最近的路網節點，產生既有充電規劃
python Preprocessing/build_ocm_plan.py --graph Graph/$CITY/$CITY.graphml --ocm Graph/$CITY/ocm_$LC.json \
    --output-pkl Graph/$CITY/existingplan_$CITY.pkl --mapping-csv Graph/$CITY/ocm_osm_mapping_$CITY.csv \
    --max-distance 100000
```

示範用小地圖（台南中西區）：步驟 1 加上 `--from-graph Graph/Tainan/Tainan.graphml`，從已下載的台南路網依行政界線
（Nominatim）切出，不需連 Overpass（Overpass 公用伺服器滿載時也能完成；與 `graph_from_place(retain_all=False)`
相同，只保留最大的連通路網）；步驟 2–4 不變：

```bash
CITY=Tainan_WestCentral; PLACE="West Central District, Tainan, Taiwan"; LC=tainan_westcentral
python Preprocessing/build_osm_graph.py --place "$PLACE" --location $CITY --output-dir Graph --from-graph Graph/Tainan/Tainan.graphml
```

驗證（2026-11-10）：以現有的 graphml 與 OCM JSON 重跑步驟 2 和 4，台南、台北的 `node_list_<city>.txt`、
`existingplan_<city>.pkl`、`ocm_osm_mapping_<city>.csv` 都與 repo 中的檔案**逐位元相同**（步驟 4 需 `--max-distance 100000`）。

注意事項：

- OSM 與 OCM 持續更新，重跑步驟 1、3 會下載到最新的路網與充電站。要重現本文件的結果，請直接使用 repo 中的
  `Graph/<city>/`，只重跑步驟 2、4。
- OCM API key 寫在 `docker/.env`（`OCM_API_KEY=...`，範本 `docker/.env.example`）。`download_ocm.py` 依序使用
  `--api-key`、環境變數 `OCM_API_KEY`、`docker/.env`；在容器內以 `--env-file docker/.env` 傳入。程式碼中不含金鑰。
- 原始 PCRL 的 `load_graph.py`、`nodes_preparation.py`（格網密度需求計算）為原始範例使用，本專案以 `build_pcrl_nodes.py` 產生節點。

---

## 5. 效用模型（Utility Model）

實作於 `evaluation_framework.py`。

### 5.1 常數

| 名稱 | 值 | 說明 |
|---|---|---|
| `K` | 15 | 每站最多充電樁數 |
| `CHARGING_POWER` | [7, 22, 50] kW | 三種充電樁功率 |
| `INSTALL_FEE` | [300, 750, 28000] | 三種充電樁安裝費 |
| `BUDGET` | 5,000,000 | 總預算 |
| `price_parkingplace` | 200 × 3.5 × 2 = 1,400 | 車位價格係數（乘以節點地價） |
| `BATTERY` | 85 kWh | 電池容量 |
| `RADIUS_MAX` | 1,000 m | 最大服務半徑 |
| `VELOCITY` | 23,000 m/h | 無路網時間時的預設車速 |
| `alpha` | 0.4 | 旅行成本 vs 充電＋等候成本的權重 |
| `my_lambda` | 0.5 | 效益 vs 成本的權重 |
| `my_inf` | 1,000,000 | 等候時間「無限大」的代替值 |
| `time_unit`, `capacity_unit` | 1 | 單位換算 |

### 5.2 單一充電站 s 的屬性（`s_dictionnary`）

| 屬性 | 公式 |
|---|---|
| 安裝費 `fee` | `price_parkingplace × estate_price(s) + Σ INSTALL_FEE · x_s` |
| 充電能力 `capability` | `Σ CHARGING_POWER · x_s`（kW） |
| 服務半徑 `radius` | `RADIUS_MAX / (1 + exp(−capability / 100))` |
| 到站車流 `D_s` | `Σ_{v 指派到 s} weak_demand(v) / distance(v, s)` |
| 服務率 `service rate` | `capability / BATTERY`（輛/小時） |
| 等候時間 `W_s`（M/D/1） | `τ = 1/service rate`，`ρ = D_s · τ`；`ρ ≥ 1` 時 `W_s = my_inf`，否則 `W_s = ρτ / (2(1−ρ))` |

其中 `weak_demand(v) = demand(v) × (1 − 0.1 × private_CS(v))`。

### 5.3 節點指派（`station_seeking`）

每個節點指派到「社會成本」最小的充電站：

```
cost(v, s) = weak_demand(v) × [ alpha × travel_time(v, s) + (1 − alpha) / distance(v, s) × (W_s + 1 / service_rate_s) ]
```

- `distance`、`travel_time`：OSM 路網最短路徑（長度 m、時間 h）。路段時間 = 長度 / `maxspeed`（無資料時用 `VELOCITY`）。
  有向圖不可達時改用無向圖。
- 實作：每個充電站位置跑一次 scipy Dijkstra（一對全部），結果快取；節點以 KD-tree 對應到最近路網節點。
- `prepare_score()` 重複「指派 → 更新 D_s、W_s」兩次。

### 5.4 分數（`norm_score`）

```
benefit     = Σ_v coverage(v) / N,  coverage(v) = H(I_v) × (1 − 0.1 × private_CS(v))
              I_v = 服務半徑內涵蓋 v 的充電站數，H(n) = 1 + 1/2 + … + 1/n（調和數）
travel      = Σ_v travel_time(v) × weak_demand(v)
charging    = Σ_s D_s / service_rate_s
waiting     = Σ_s D_s × W_s

cost  = [ alpha × travel/travel₀ + (1 − alpha) × (charging/charging₀ + waiting/waiting₀) ] / 3
score = my_lambda × benefit/benefit₀ − (1 − my_lambda) × cost
```

下標 0 表示既有規劃的值（`existing_score`），因此既有規劃的分數為 0.5 × 1 − 0.5 × (0.4 + 0.6 × 2) / 3 = **0.2333**。
分數下限截在 −25。

### 5.5 評估指標（`test_solution.py`）

- 移動時間（旅行時間，分鐘）：開車到被指派充電站的時間，依需求加權的平均值。
- 等待時間（等候時間，分鐘）：在充電站排隊等候的時間，依 D_s 加權的平均與最大值。
- 預算使用率：`(Σ fee − 既有規劃 fee) / BUDGET`。
- 限制檢查：預算、每站 ≤ K 樁、每節點只指派一站、等候時間有限。

---

## 6. 強化學習環境 StationPlacement

實作於 `env_plus.py`（gym 0.21 / gymnasium 介面，`gym_compat.py`）。

### 6.1 Episode 流程

1. `reset()`：預算 = 5,000,000；載入既有規劃並計算正規化基準；`best_score` = 0.2333。
2. 每一步選一個動作（5 種，見 6.2），修改規劃並扣預算。
3. 預算不足以執行該動作時 `game_over = True`（該動作不執行）。
4. 規劃站數 = 節點數，或步數 ≥ N/2 時結束。

### 6.2 動作空間（`Discrete(5)`，啟發式巨集動作）

| 動作 | 意義 | 選點規則 |
|---|---|---|
| 0 | 新建充電站 | 尚無充電站的節點中，覆蓋度最低者 |
| 1 | 新建充電站 | 尚無充電站的節點中，`weak_demand` 最高者 |
| 2 | 既有站加一支 22 kW 樁 | 未滿 K 樁的充電站中，覆蓋度最低者 |
| 3 | 既有站加一支 22 kW 樁 | 未滿 K 樁的充電站中，`weak_demand` 最高者 |
| 4 | 移樁 | 從「非既有」充電站中效益上界最低者拆一支最貴的樁（退回預算），加到最需要支援（等候＋充電時間最大）的充電站；該站已滿則加到最近的空節點 |

細節：

- 新建充電站的初始配置（`initial_solution`）：50 m 內需求總和 × `BATTERY` 為最低容量，
  在「容量 → 最便宜配置」表（`prepare_config`，三種樁合計 ≤ K）中取第一個超過該容量的配置。
- 動作 2/3 在沒有未滿的充電站時、動作 4 在沒有可拆的充電站時，會用 `random.choice` 隨機選一個空節點，
  這是環境中唯一的隨機來源。要重現結果需設定 `random.seed()`。
- 同分時取節點清單中的第一個。

### 6.3 獎勵

```
new_score = norm_score(目前規劃)
reward    = max(0, new_score − best_score)；若 new_score > best_score 則更新 best_score 與 best_plan
```

- 只有分數創新高時才有正獎勵（避免來回震盪的 "jojo learning"）。
- 一個 episode 的獎勵總和 = 最佳分數 − 0.2333，本文件稱為 **improvement（相對既有規劃的提升量）**，是所有實驗的主要指標。

### 6.4 原始觀測（`env_plus`）

- 每個節點 8 個值：x、y、demand、estate price、private CS、三種樁數；最後加上剩餘預算。
- 全部除以 `BUDGET`，型別 float16。維度 = 8N + 1（台南 127,865；台北 96,241）。
- 維度依城市節點數而變、且含原始經緯度，所以**不同城市的模型無法互用**。這是設計 `transfer/env.py` 的原因。

---

## 7. 模型訓練（預訓練階段）

本計畫的模型開發分兩個階段：

1. **訓練（預訓練）**：在台南資料上以 DQN 從零訓練充電站佈建模型（本節）。
2. **泛化與微調驗證**：把台南模型用到台北，測試不經訓練的表現（zero-shot），並以模型引導的 beam search 微調（第 10 節）。

台南共訓練了兩個模型：

| 模型 | 腳本 | 環境 / 觀測 | 用途 |
|---|---|---|---|
| 原始 PCRL 模型 | `reinforcement.py` | `env_plus`，每個節點 8 值，共 127,865 維 | 依原始論文方法訓練的台南模型，產生台南充電規劃 |
| 來源模型（source model） | `transfer/train.py` | `transfer/env.py`，固定 324 維 | 可跨城市使用的台南模型，作為泛化與微調的預訓練模型 |

### 7.1 訓練流程

```
初始化 Q-network θ 與 target network θ⁻ = θ，replay buffer D
for t = 1 … total_timesteps:
    以 ε-greedy 選動作 a_t（機率 ε 隨機，否則 argmax_a Q(s_t, a; θ)），ε 從 1.0 線性降到 0.05（前 10% 步數）
    在環境執行 a_t：修改充電規劃、扣預算、重新指派節點、計算分數 → r_t, s_{t+1}, done
    (s_t, a_t, r_t, s_{t+1}, done) 存入 D
    if t > learning_starts 且 t % train_freq == 0:
        從 D 抽 batch_size 筆，計算 y = r + γ (1 − done) max_a' Q(s', a'; θ⁻)
        loss = Huber(Q(s, a; θ), y)，Adam 更新 θ（梯度裁剪 10）
    if t % target_update_interval == 0: θ⁻ ← θ
    episode 結束（預算用完或步數 ≥ N/2）就 reset
    依設定保存模型（7.2：訓練獎勵創新高時；7.4：每 1,000 步 checkpoint）
```

- 每一步都要重算整個規劃的分數：節點指派（`station_seeking`，每個節點對每個充電站算成本）、到站車流、等候時間、覆蓋度，
  這是訓練時間的主要來源。
- 一個 episode 的獎勵總和 = 該 episode 最佳規劃分數 − 既有規劃分數（improvement）。
- 訓練環境以 `stable_baselines3.common.monitor.Monitor` 包裝，每個 episode 的獎勵 `r`、長度 `l`、經過時間 `t` 寫入 `monitor.csv`。

### 7.2 原始 PCRL 模型訓練（`reinforcement.py`）

**設定**

| 參數 | 值 |
|---|---|
| 城市 | Tainan（N = 15,983 個節點，既有 17 站） |
| 演算法 | DQN（stable-baselines3 1.5.0，舊環境） |
| Policy | `MlpPolicy`，預設隱藏層 [64, 64]，ReLU |
| 網路 | Linear(127865→64) → ReLU → Linear(64→64) → ReLU → Linear(64→5)，共 **8,187,909** 個參數 |
| total_timesteps | 20,000 |
| learning_rate | 1e-3（Adam，eps 1e-8） |
| buffer_size | 10,000 |
| batch_size | 128 |
| learning_starts | 1,000 |
| target_update_interval | 1,000（tau = 1.0，硬更新） |
| gamma | 0.99（預設） |
| train_freq / gradient_steps | 4 / 1（預設） |
| exploration | ε 1.0 → 0.05，前 10% 步數（2,000 步）線性下降（預設） |
| loss | Huber（smooth L1），max_grad_norm = 10 |
| seed | 1；`torch.manual_seed`、`np.random.seed`、`random.seed`，`torch.use_deterministic_algorithms(True)` |
| 裝置 | CPU |
| 模型保存 | `SaveOnBestTrainingRewardCallback`：每 400 步讀 `monitor.csv`，取最近 3 個 episode 的平均訓練獎勵，創新高就另存 `tmp_Tainan/best_model_Tainan_<steps>.zip`（不覆蓋舊模型；2026-11-10 清理後只保留實際使用的 11200） |

**命令列參數（2026-11-12 加入）**：原始 PCRL 的四支程式都以 `--city` 指定城市（`Graph/<city>/` 的資料夾名稱，預設 Tainan），
不加參數時與原本的台南設定完全相同（台南指標重算結果與 `logs/training/test_solution_Tainan.log` 逐字相同）。

| 程式 | 參數 | 預設 |
|---|---|---|
| `reinforcement.py` | `--city`、`--steps`（訓練步數）、`--learning-starts`、`--check-freq`、`--metrics-freq`、`--seed`、`--log-dir` | Tainan、20,000、1,000、400、50、1、`tmp_<city>/` |
| `model_eval.py` | `--city`、`--model`、`--output`、`--metrics-freq` | Tainan、`tmp_<city>/` 中步數最大的 `best_model_<city>_<steps>.zip`（即最後一次創新高的模型）、`Results/<city>/`、50 |
| `test_solution.py` | `--city`、`--plan-dir` | Tainan、`Results/<city>/` |
| `Visualisierung/visualise.py` | `--city`、`--plan`、`--output` | Tainan、`Results/<city>/plan_RL.pkl`、`Images/Result_Plots/<city>/RL_<city>.png` |

`--metrics-freq n`：每 n 步印出目前規劃（執行中的 episode）的平均移動時間與平均等待時間（`test_solution.py` 的定義），
例如 `step 400: 72 stations, avg travel time 0.702 min, avg waiting time 0.341 min`；0 = 不印。
`model_eval.py` 在 episode 結束時也會印一次。

例：在台北直接訓練原始 PCRL 模型 1,000 步（`learning_starts` 須小於總步數，網路才會更新；模型在第一個 episode
結束後才會存檔，台北一個 episode 約 600–700 步，所以把檢查間隔縮短為 100 步）：

```bash
python reinforcement.py --city Taipei --steps 1000 --learning-starts 200 --check-freq 100   # → tmp_Taipei/best_model_Taipei_<steps>.zip
python model_eval.py --city Taipei                                          # → Results/Taipei/
python test_solution.py --city Taipei
python Visualisierung/visualise.py --city Taipei                            # → Images/Result_Plots/Taipei/RL_Taipei.png
```

示範流程（台南中西區，4.4；全部約 2 分鐘）：

```bash
python reinforcement.py --city Tainan_WestCentral --steps 1000 --learning-starts 200 --check-freq 100   # 約 70 秒
python model_eval.py --city Tainan_WestCentral                                                          # 約 30 秒
python test_solution.py --city Tainan_WestCentral
python Visualisierung/visualise.py --city Tainan_WestCentral
```
| 輸出 | `tmp_Tainan/`（模型、`train.log`） |

**訓練過程**（`tmp_Tainan/train.log`，「最近 3 個 episode 平均訓練獎勵」，含探索）

| 步數 | 800 | 2,000 | 4,000 | 6,000 | 7,600 | 10,000 | 11,200 | 14,000 | 16,000 | 18,000 | 20,000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 平均訓練獎勵 | 7.02 | 7.18 | 7.32 | 7.69 | 7.79 | 7.61 | **7.79** | 7.57 | 7.59 | 7.48 | 7.58 |

- 共 49 次檢查，14 次創新高並保存：800、1,600、2,000、2,400、3,200、3,600、4,400、4,800、5,600、6,000、6,800、7,200、7,600、11,200 步。
- 0–7,600 步穩定上升（7.02 → 7.79），之後維持在 7.4–7.8；最終使用的最佳模型為 11,200 步（7.79）。
- 訓練時間：約 37.5 小時（800 步模型存於 2026-10-15 01:22，11,200 步模型存於 10-15 21:14，`train.log` 最後寫入 10-16 13:21），
  約 6.9 秒/步。之後覆蓋計算已向量化（11.4），同樣的訓練現在約快 2 倍。
- 圖：`Images/Result_Plots/Training/original_Tainan_training.png`。
- 訓練曲線以 `tmp_Tainan/train.log` 為準（`tmp_Tainan/monitor.csv` 為 `model_eval.py` 評估 episode 的紀錄）。

**評估（`model_eval.py`，模型 `best_model_Tainan_11200.zip`，貪婪 episode）**

| 項目 | 值 |
|---|---|
| improvement | **7.9091**（最佳分數 8.1424，既有 0.2333） |
| 充電站數 | 755（既有 17 站 + 新建 738 站） |
| episode 步數 | 739 |
| 輸出 | `Results/Tainan/plan_RL.pkl`、`Results/Tainan/nodes_RL.txt` |
| 地圖 | `Images/Result_Plots/Tainan/RL_Tainan.png`（`Visualisierung/visualise.py`；紅點 = 充電站，灰點 = 一般路口） |

**評估指標（`test_solution.py`，2026-11-09 以目前程式重算，log：`logs/training/test_solution_Tainan.log`）**

| 指標 | 既有規劃（17 站） | 原始 PCRL 模型規劃（755 站） | 改善 |
|---|---|---|---|
| 平均移動時間（旅行時間，分鐘，依需求加權） | 10.806 | **1.771** | −83.6% |
| 平均等待時間（分鐘，依車流加權） | 7.457 | **0.365** | −95.1% |
| 最大等待時間（分鐘） | 56.842 | 8.456 | −85.1% |
| 預算使用率 | 0% | 99.994% | |

**服務目標達成**：原始 PCRL 模型的台南規劃，平均移動時間 **1.771 分鐘**、平均等待時間 **0.365 分鐘**，
兩項皆**遠低於 15 分鐘目標**（同一指標也以 `analysis/plan_metrics.py` 重算，結果與上表逐位元相同，`Results/Tainan/metrics.csv`）。

- 兩者都出現 `More chargers at the station than admitted: 18 chargers`：來自 OCM 資料中一個有 18 支樁的既有充電站
  （超過 K = 15），屬於輸入資料，不是模型造成的。

**與隨機策略比較（同一環境，`Results/Transfer/random_Tainan.csv`）**

| 策略 | improvement | 充電站數 | 步數 |
|---|---|---|---|
| **原始 PCRL 模型** | **7.9091** | 755 | 739 |
| 隨機策略（種子 1 / 2 / 3） | 7.0070 / 0.8284 / 7.0912（平均 4.9757） | 347 / 36 / 351 | 655 / 671 / 651 |

- 原始模型比隨機策略平均高 59.0%，且高於隨機策略的每一個種子；台南規劃新建 738 站，平均移動時間降低 83.6%、
  平均等待時間降低 95.1%。

### 7.3 為什麼需要第二個（來源）模型

原始模型的觀測為每個節點 8 個值（經緯度、需求、地價、私人充電樁、三種樁數）＋預算，維度 = 8N + 1：

- 台南 127,865 維、台北 96,241 維，網路輸入層大小不同，**台南模型無法載入到台北**。
- 觀測含原始經緯度，台南（約 23°N, 120°E）與台北（約 25°N, 121.5°E）的數值範圍完全不同。
- 第一層 127,865 × 64 ≈ 818 萬個權重，絕大部分只對應單一節點，難以學到可轉移的規則。

因此設計固定 324 維、依城市正規化的觀測（第 8 節），在台南重新訓練一個可跨城市使用的來源模型。

### 7.4 來源模型訓練（`transfer/train.py`，`runs/source_Tainan/`）

**設定**（`runs/source_Tainan/config.json`）

| 參數 | 值 |
|---|---|
| 城市 | Tainan |
| 環境 / 觀測 | `transfer.env.TransferStationPlacement`，324 維（第 8 節） |
| 網路 | `MlpPolicy`，`net_arch = [256, 256]`：Linear(324→256) → ReLU → Linear(256→256) → ReLU → Linear(256→5)，共 **150,277** 個參數 |
| total_timesteps | 20,000 |
| learning_rate | 1e-3（Adam） |
| buffer_size | 50,000（大於總步數，保留全部經驗） |
| batch_size | 128 |
| learning_starts | 1,000 |
| target_update_interval | 1,000 |
| gamma | 0.99 |
| train_freq / gradient_steps | 4 / 1 |
| exploration | ε 1.0 → 0.05，前 10% 步數 |
| loss | Huber，max_grad_norm = 10 |
| seed | 1（同 7.2 的決定性設定） |
| torch threads | 8 |
| 模型保存 | `CheckpointCallback` 每 1,000 步存 `checkpoints/model_<N>_steps.zip`，結束時存 `final_model.zip` |
| 執行 | `scripts/run_generalization.sh source`，log 在 `logs/training/source_Tainan.log` |

**訓練過程**（`runs/source_Tainan/monitor.csv`）

- 共 31 個完整 episode，平均長度 633 步，平均訓練獎勵 5.742（含探索），最後 10 個 episode 平均 5.889。
- 大部分 episode 的獎勵在 6.4–7.6 之間（含 ε-greedy 探索）。
- 訓練時間：約 13.8 小時（2026-11-03 16:13 → 11-04 06:16），約 2.5 秒/步（含 8 threads 的網路更新）。

**Checkpoint 選擇**（`scripts/run_generalization.sh select`，`Results/Transfer/source_selection.csv`）

每個 checkpoint 在台南跑一個貪婪 episode（不探索）：

| 步數 | 1k* | 2k | 3k | 4k | 5k | 6k | 7k | 8k | 9k | 10k |
|---|---|---|---|---|---|---|---|---|---|---|
| improvement | 8.454 | 6.474 | 0.744 | 2.445 | 0.523 | 6.878 | 1.789 | 7.946 | 3.108 | 7.716 |
| **步數** | **11k** | **12k** | **13k** | **14k** | **15k** | **16k** | **17k** | **18k** | **19k** | **20k** |
| improvement | 3.296 | 0.523 | 2.929 | 3.345 | 3.272 | 6.765 | **8.532** | 8.130 | 6.639 | 5.978 |

\* 1k 為尚未開始更新的初始網路（≤ learning_starts），依規則排除。

- 選擇規則：排除 ≤ learning_starts 的 checkpoint，取 improvement 最高者 → **17,000 步（8.532，405 站，581 步）**，
  複製為 `runs/source_Tainan/best_source.zip`。
- 各 checkpoint 的貪婪分數差異大（0.5–8.5），因此以貪婪 episode 對所有 checkpoint 評分，選出最佳模型。
- 圖：`Images/Result_Plots/Training/source_Tainan_training.png`（橘點 = 訓練 episode，藍線 = 各 checkpoint 貪婪 episode）。

**結果**（服務指標：`scripts/run_generalization.sh service`，來源模型以貪婪 episode 產生台南規劃
`Results/Tainan/source_model/`，improvement 8.5323、405 站、581 步，與 checkpoint 選擇時的結果逐位元相同；
`Results/Tainan/metrics.csv`）

| 規劃 | 台南 improvement | 充電站數 | 平均移動時間（分鐘） | 平均等待時間（分鐘） | 最大等待時間（分鐘） |
|---|---|---|---|---|---|
| **來源模型（17k）** | **8.532** | 405 | **2.178** | **0.098** | **0.457** |
| 原始 PCRL 模型 | 7.909 | 755 | 1.771 | 0.365 | 8.456 |
| 既有充電站 | 0 | 17 | 10.806 | 7.457 | 56.842 |

- 來源模型以 1.8% 的參數量（150,277 vs 8,187,909）在台南超過原始模型 7.9%（improvement），且只用約一半的充電站
  （405 vs 755 站）。
- **服務目標達成**：來源模型的平均移動時間 2.178 分鐘、平均等待時間 0.098 分鐘，**兩項皆遠低於 15 分鐘**；
  平均等待時間比原始模型再降低 73%，最大等待時間由 8.456 降到 0.457 分鐘。
- 圖：`Images/Result_Plots/Training/Tainan_service_metrics.png`（既有、原始模型、來源模型，標示 15 分鐘目標線）。

### 7.5 訓練階段小結

| 項目 | 結果 |
|---|---|
| 原始 PCRL 模型（台南） | 20,000 步、約 37.5 小時；improvement 7.909，755 站 |
| 來源模型（台南，可跨城市） | 20,000 步、約 13.8 小時；選 17,000 步 checkpoint，improvement 8.532，405 站 |
| 服務目標（台南，< 15 分鐘） | **達成**：原始模型平均移動 1.771 分、平均等待 0.365 分；來源模型平均移動 2.178 分、平均等待 0.098 分（既有充電站為 10.806 / 7.457 分） |
| 影像紀錄 | 兩張訓練曲線、台南服務指標圖、台南充電站地圖（第 12 節） |
| 主要成果 | 兩個台南模型都完成訓練並產生充電規劃，平均移動與等待時間皆遠低於 15 分鐘；來源模型以 1.8% 的參數量超過原始模型 7.9%，作為第二階段的預訓練模型 |

---

## 8. 跨城市遷移版本（transfer/env.py）

### 8.1 設計原則

`TransferStationPlacement` 繼承 `StationPlacement`，**動作、獎勵、episode 邏輯完全不變**（分數可與 env_plus 直接比較），
只替換觀測：固定 324 維、每個值都依城市正規化並截到 [−1, 1]（float32）。

### 8.2 觀測向量（324 維）

| 區塊 | 維度 | 內容 |
|---|---|---|
| 全域狀態 | 16 | 見下表 |
| 動作候選 | 4 × 7 = 28 | 動作 0–3 各自會選到的節點的特徵（同 env_plus 的選點與同分規則） |
| 高需求空節點 | 32 × 5 = 160 | 尚無充電站、`weak_demand` 最高的 32 個節點 |
| 最需要支援的充電站 | 16 × 7 = 112 | 等候＋充電時間最大的 16 個充電站 |
| 覆蓋直方圖 | 8 | 依需求加權的覆蓋度分布 |

全域狀態 16 維：

| 索引 | 特徵 |
|---|---|
| 0 | 剩餘預算 / BUDGET |
| 1 | min(步數 / 1000, 1) |
| 2 | log(1 + 站數) / log(1001) |
| 3 | 新建站比例（非既有） |
| 4–6 | 7 / 22 / 50 kW 樁數占比 |
| 7 | 平均每站樁數 / K |
| 8 | 已滿站比例 |
| 9 | 需求加權覆蓋度 Σ weak_demand × squash(coverage) / Σ weak_demand |
| 10 | 未覆蓋節點比例 |
| 11 | 未覆蓋需求比例 |
| 12 | tanh(best_score / 10) |
| 13, 14 | 各站「需要度」log 尺度的平均、最大 |
| 15 | 超載站比例（W_s = my_inf） |

節點特徵（5 維）：weak_demand、squash(coverage)、estate price、x_norm、y_norm。
候選特徵（7 維）= 節點特徵 + 該點樁數 / K + 是否有候選（1/0）。
充電站特徵（7 維）：log 尺度需要度、三種樁數 / K、x_norm、y_norm、是否為既有站。

正規化方式：

- `x_norm`, `y_norm`：依該城市的經緯度範圍線性縮放到 [−1, 1]。
- `squash(v) = v / (1 + v)`。
- `log_scale(v) = log1p(min(v, 1e6)) / log1p(1e6)`。
- 「需要度」= D_s / service_rate + D_s × W_s；服務率為 0（樁數 [0,0,0]）或非有限值時設為 1e6（避免 NaN，見 11.1）。
- 觀測若出現非有限值直接拋出錯誤。

### 8.3 每個 episode 的狀態重設（2026-11-08 加入）

`reset()` 會清空 `StationPlacement.node_dict`、`cost_dict`、`ef._OSM_ROUTE_CACHE`，並把每個節點的屬性還原成初始值。
原因見 11.2。驗證：設定 `random.seed(1)` 時，同一行程內連續兩個 episode 的結果逐位元相同，
且與修改前在新行程中跑的結果逐位元相同。

### 8.4 網路結構

| 層 | 形狀 | 參數數 |
|---|---|---|
| hidden 1 | Linear(324→256) + ReLU | 83,200 |
| hidden 2 | Linear(256→256) + ReLU | 65,792 |
| output | Linear(256→5) | 1,285 |
| 合計 | | **150,277** |

（`NET_ARCH = [256, 256]`，target network 結構相同。）

---

## 9. 程式模組與參數一覽

### 9.1 `transfer/train.py`（來源模型訓練）

```bash
python -m transfer.train --city Tainan --out runs/source_Tainan --steps 20000 --seed 1 --threads 8
```

| 參數 | 預設 | 說明 |
|---|---|---|
| `--city` | （必填） | `Graph/<city>/` |
| `--out` | （必填） | 輸出資料夾：`config.json`、`monitor.csv`、`checkpoints/model_<N>_steps.zip`、`final_model.zip` |
| `--steps` | 20000 | 訓練步數 |
| `--seed` | 1 | 亂數種子（torch、numpy、random，決定性演算法） |
| `--lr` | 1e-3 | 學習率（Adam） |
| `--buffer-size` | 50000 | replay buffer |
| `--batch-size` | 128 | |
| `--learning-starts` | 1000 | 開始更新前收集的步數 |
| `--target-update` | 1000 | target network 更新間隔 |
| `--gamma` | 0.99 | 折扣因子 |
| `--eps-initial` / `--eps-final` / `--eps-fraction` | 1.0 / 0.05 / 0.1 | ε-greedy 探索 |
| `--checkpoint-freq` | 1000 | checkpoint 間隔 |
| `--threads` | 4 | torch CPU 執行緒 |

### 9.2 `transfer/evaluate.py`（貪婪 episode 評估）

```bash
python -m transfer.evaluate --city Taipei --policy model --model runs/source_Tainan/best_source.zip \
    --label zeroshot --seed 0 --eval-seed 1 --csv Results/Transfer/generalization.csv [--save-plan <dir>]
```

| 參數 | 說明 |
|---|---|
| `--policy` | `model`（模型或整個 checkpoint 資料夾）、`random`（隨機策略） |
| `--seed` | 模型的訓練種子（寫入 CSV） |
| `--eval-seed` | episode 種子（`random`、`np.random`），預設等於 `--seed` |
| `--save-plan` | 輸出 `plan_RL.pkl`、`nodes_RL.txt`（可給 `visualise.py`、`analysis/plan_metrics.py`） |

- 每個模型跑一個貪婪 episode（`deterministic=True`），`torch.set_num_threads(1)`。
- 權重含 NaN/Inf 的模型直接拒絕。
- CSV 欄位：`city, label, seed, policy, model, train_steps, existing_score, score, improvement, relative_pct, n_stations, episode_steps, seconds, eval_seed`；
  以 `fcntl.flock` 鎖檔，可多行程同時寫入。

### 9.3 `transfer/beam_search.py`（模型引導的 beam search）

```bash
python -m transfer.beam_search --city Taipei --beam 4 --topk 2 --eval-seed 1 --label B4k2 \
    --csv Results/Transfer/beam.csv --save-plan Results/Transfer/plans/B4k2_e1
```

| 參數 | 預設 | 說明 |
|---|---|---|
| `--city` | （必填） | 目標城市 |
| `--model` | `runs/source_Tainan/best_source.zip` | 引導搜尋的預訓練模型 |
| `--beam` | 4 | beam 寬度 B：每一步保留的候選規劃數 |
| `--topk` | 2 | 每個候選規劃嘗試 Q 值最高的 k 個動作 |
| `--value-weight` | 1.0 | 排序分數 = 目前最佳分數 + value_weight × max Q(下一個狀態) |
| `--eval-seed` | 1 | episode 種子（與 `transfer/evaluate.py` 相同的設定方式） |
| `--max-steps` | 3000 | 步數上限（正常情況下預算用完就結束） |
| `--patience` | 200 | 貪婪主線完成後，若最佳分數連續這麼多步沒有提升就結束搜尋 |
| `--max-mem-gb` | 16 | 協調行程＋所有候選行程的記憶體（PSS）上限，超過即中止 |
| `--min-available-gb` | 4 | 主機可用記憶體下限，低於即中止 |
| `--label`、`--csv`、`--save-plan`、`--log-every` | `beam`、`Results/Transfer/beam.csv`、無、25 | 結果標籤、CSV、規劃輸出資料夾、log 頻率（每 n 步印分數與目前最佳成員規劃的平均移動／等待時間；開始時印既有規劃、結束時印存檔規劃；0 = 不印中途進度） |

**演算法**

```
B_0 = { reset 後的狀態 }，貪婪主線 g = reset 後的狀態
for t = 1, 2, …:
    for 每個候選 c ∈ B_{t−1}:
        for 每個 a ∈ Q 值最高的 k 個動作 (Q(s_c, ·)，台南模型):
            c' = 在 c 的複本上執行 a；記錄 best_score(c')、max_a' Q(s_c', a')
    g ← g 的 Q 值最高動作的子節點（貪婪主線）
    預算用完的 c' 移到「完成」集合
    B_t = { g }（若未完成）＋ 其餘依 key = best_score(c') + value_weight × max Q(s_c') 排序，
          相同規劃只留一個，補滿到 B 個
    若 g 已完成且最佳分數連續 patience 步未提升 → 結束
結果 = 所有候選（完成與未完成）中 best_score 最高的規劃
```

- key 的意義：已達到的分數，加上模型估計「之後還能再提升多少」（環境的獎勵 = 超過目前最佳分數的增量，
  所以 Q 值正好估計之後還能提升多少）。
- 貪婪主線：B 個位置中固定保留一個給模型的貪婪路徑（每步取 Q 值最高的動作），其餘 B − 1 個位置探索其他動作組合，
  因此搜尋結果涵蓋 zero-shot 規劃並在其上尋找更好的規劃。B = k = 1 時即為模型的貪婪 episode。
- 實作：每個候選規劃是一個 fork 出來的行程，保有完整環境狀態（copy-on-write）；每個嘗試的動作在新的子行程中執行，
  子行程經 Unix socket 回報分數、Q 值與規劃簽章給協調行程，未被保留的行程立即結束。子行程沿用同一個指令迴圈
  （不遞迴），可支援任意長度的 episode。
- 亂數：fork 時保留 `random` 模組的狀態（`fork_keep_random`，見 11.5），所以結果可逐位元重現，且 B = k = 1 與
  `transfer/evaluate.py` 的 zero-shot 結果完全相同。
- 記憶體保護：所有候選行程屬於同一個 process group；監控執行緒每 2 秒檢查記憶體（PSS）與主機可用記憶體，
  超過 `--max-mem-gb` 或低於 `--min-available-gb` 就結束整個 process group（exit code 3）。
  `scripts/run_beam_search.sh` 另外把每個 run 放在獨立的 cgroup（`systemd-run --user --scope -p MemoryMax=12G`），
  到達上限時只有該 run 被終止，主機不受影響。
- 輸出欄位：`score`、`improvement`、`n_stations`、`episode_steps`、`expansions`（總共嘗試的動作數）、`seconds`、
  `action_counts`（最佳規劃路徑上動作 0–4 的次數）；log 最後一行為記憶體峰值。

### 9.4 `scripts/run_generalization.sh`

| 階段 | 內容 |
|---|---|
| `source` | 台南訓練 20,000 步（lr 1e-3、buffer 50,000、8 threads），每 1,000 步一個 checkpoint |
| `select` | 每個 checkpoint 在台南跑貪婪 episode，排除 ≤ learning_starts 的 checkpoint，取 improvement 最高者 → `best_source.zip` |
| `service` | 台南服務指標：來源模型產生台南規劃（`Results/Tainan/source_model/`），再計算既有充電站、原始 PCRL 模型、來源模型的平均移動時間與平均／最大等待時間 → `Results/Tainan/metrics.csv`、`Tainan_service_metrics.png` |
| `baselines` | 台北：zero-shot（episode 種子 1–5，存規劃）、隨機策略（種子 1–3） → `Results/Transfer/generalization.csv` |
| `analyze` | `analysis/generalization.py` → `Results/Transfer/generalization_summary.txt`、`Images/Result_Plots/Transfer/generalization.png` |
| `all` | 以上依序執行 |

環境變數：`PY`（預設 conda 環境 `PCRL-py312` 的 Python；舊環境 `$HOME/anaconda3/envs/PCRL/bin/python`）、`SOURCE_STEPS=20000`、`EVAL_SEEDS="1 2 3 4 5"`、`RANDOM_SEEDS="1 2 3"`、`JOBS=8`。

### 9.5 `scripts/run_beam_search.sh`

| 階段 | 內容 |
|---|---|
| `validate` | B = k = 1、episode 種子 1–5：確認與 zero-shot 結果完全相同 |
| `search` | B = `$BEAM`、k = `$TOPK`、episode 種子 1–5，規劃存到 `Results/Transfer/plans/B<B>k<k>_e<seed>` |
| `analyze` | `analysis/beam.py` → `Results/Transfer/beam_summary.txt`、`Images/Result_Plots/Transfer/beam/` 三張圖；`analysis/plan_maps.py` → 微調前後的台北充電站分布圖 `map_seed<seed>.png` |
| `all` | 以上依序執行 |

環境變數：`PY`（同 9.4）、`BEAM=4`、`TOPK=2`、`EVAL_SEEDS="1 2 3 4 5"`、`JOBS=2`（同時執行的 run 數）、`MEM_MAX=12G`（每個 run 的 cgroup 上限）、
`GUARD_GB=11`（傳給 `--max-mem-gb`）。

### 9.6 分析與影像紀錄

| 腳本 | 輸出 |
|---|---|
| `analysis/data_maps.py` | 路網與既有充電站（OCM 對應後）分布圖：`python -m analysis.data_maps [--cities Tainan Taipei] [--far-m 500]` → `Images/Result_Plots/Data/<city>_network_stations.png` |
| `analysis/training_plots.py` | `Images/Result_Plots/Training/original_Tainan_training.png`、`source_Tainan_training.png`、`Tainan_service_metrics.png`（讀 `Results/Tainan/metrics.csv`） |
| `analysis/generalization.py` | 泛化統計（Welch t 檢定）與 `generalization.png` |
| `analysis/beam.py` | 驗證（B = k = 1 vs zero-shot）、微調前後配對比較（paired t 檢定、Wilcoxon）、服務指標（平均與逐種子）、三張圖 |
| `analysis/plan_maps.py` | 台北微調前後充電站分布圖（每個種子一張）：`python -m analysis.plan_maps [--label B4k2] [--seeds 1 2 3 4 5]` |
| `analysis/trackio_logs.py` | 訓練與微調紀錄轉成 trackio 可匯入的 CSV：`python -m analysis.trackio_logs` → `Results/Trackio/training/`、`finetuning/`（11.11） |
| `analysis/plan_metrics.py` | 任一已存規劃的服務指標：`python -m analysis.plan_metrics --city <city> --plan <資料夾> --label <名稱> --csv <檔案>`（`--existing` = 既有規劃） |

---

## 10. 泛化與微調驗證（第二階段）

所有分數皆為 **improvement = 最佳規劃分數 − 既有規劃分數（台北 0.2333）**，越高越好。台北的比較都使用同一組
episode 種子 1–5（環境隨機選點的種子），微調前後逐一配對。

### 10.1 台南基準（預訓練模型）

來源模型的訓練與 checkpoint 選擇見 7.4（`Results/Transfer/source_selection.csv`、`random_Tainan.csv`、`Results/Tainan/metrics.csv`）：

| 策略 | improvement | 平均移動時間（分鐘） | 平均等待時間（分鐘） |
|---|---|---|---|
| **來源模型（17,000 步）** | **8.532** | **2.178** | **0.098** |
| 原始 PCRL 模型（`best_model_Tainan_11200`） | 7.909 | 1.771 | 0.365 |
| 隨機策略（種子 1 / 2 / 3） | 7.01 / 0.83 / 7.09（平均 4.98） | | |
| 既有充電站 | 0 | 10.806 | 7.457 |

- 來源模型分數最高，比隨機策略平均高 71.5%，作為第二階段的預訓練模型（`runs/source_Tainan/best_source.zip`）。
- **台南兩個模型的平均移動時間與平均等待時間皆低於 15 分鐘**（最高 2.178 分鐘）。

### 10.2 台北泛化（zero-shot）

**方法**：台南來源模型不做任何台北訓練，直接在台北地圖上跑貪婪 episode（`scripts/run_generalization.sh baselines`），
與隨機策略比較。

| 策略 | n | 平均 improvement | 範圍 |
|---|---|---|---|
| **zero-shot（台南模型）** | 5 | **2.3239** | 2.0695–2.4756 |
| 隨機策略 | 3 | 1.2563 | 0.1666–1.8570 |

- zero-shot 平均比隨機策略高 **85.0%**；5 個種子的結果（2.07–2.48）全部高於隨機策略的最佳結果（1.86）。
- 台南學到的佈建策略不需重新訓練即可直接用在台北，證明模型具備跨城市的泛化能力。
- 紀錄：`Results/Transfer/generalization.csv`、`generalization_summary.txt`；圖：`Images/Result_Plots/Transfer/generalization.png`。

### 10.3 台北微調（模型引導的 beam search）

**方法**（9.3）：以台南模型的 Q 值引導搜尋，在台北地圖上微調充電規劃。每一步保留 B = 4 個候選規劃，每個候選嘗試
Q 值最高的 k = 2 個動作，依「目前分數 + 模型估計的後續提升」排序；其中一個位置固定保留模型的貪婪路徑。
微調前 = zero-shot 規劃，微調後 = beam search 規劃，同一個 episode 種子配對比較
（`scripts/run_beam_search.sh validate / search / analyze`）。

**正確性驗證**：B = k = 1 的搜尋在 5 個種子上都與 zero-shot 結果**逐位元相同**（improvement 差 0、episode 步數相同），
確認搜尋程式與評估程式使用完全相同的環境與亂數。

**結果**（`Results/Transfer/beam.csv`、`beam_summary.txt`）：

| episode 種子 | 微調前（zero-shot） | 微調後（beam search） | 提升 | 站數（前 → 後） |
|---|---|---|---|---|
| 1 | 2.4756 | **2.6428** | **+6.8%** | 403 → 373 |
| 2 | 2.0695 | **2.0954** | +1.3% | 231 → 231 |
| 3 | 2.3019 | 2.3021 | +0.01% | 292 → 293 |
| 4 | 2.3604 | **2.5539** | **+8.2%** | 285 → 342 |
| 5 | 2.4119 | 2.4119 | 0 | 294 → 294 |
| **平均** | **2.3239** | **2.4012** | **+3.3%** | 301 → 307 |

- 5 個種子中 4 個提升、1 個持平，**沒有任何種子下降**；單一種子最高提升 8.2%。
- 種子 1 以較少的充電站（403 → 373 站）得到更高的分數，代表搜尋找到更有效率的佈建位置。

**微調前後的平均移動時間與平均等待時間**（`analysis/plan_metrics.py`，`test_solution.py` 的定義，5 個種子的規劃平均；
`Results/Transfer/plans/metrics.csv`、`metrics_B4k2.csv`，逐種子數值見 `beam_summary.txt`）：

| 指標 | 既有規劃（29 站） | 微調前（zero-shot） | 微調後（beam search） | 微調後 vs 微調前 |
|---|---|---|---|---|
| **平均移動時間**（旅行時間，分鐘） | 3.861 | **1.199** | **1.172** | −2.3% |
| **平均等待時間**（分鐘） | 3.030 | **0.110** | **0.111** | 持平（差 < 0.001 分鐘） |
| 最大等待時間（分鐘） | 26.692 | 0.498 | **0.460** | **−7.7%** |

- 台北微調前後的平均移動時間與平均等待時間**皆遠低於 15 分鐘**；5 個種子中每一個規劃也都低於 15 分鐘
  （平均移動時間 1.02–1.36 分、平均等待時間 0.09–0.13 分）。
- 與台北既有規劃相比，微調後的規劃平均移動時間降低 69.6%、平均等待時間降低 96.3%。

**微調前後的充電站分布**（`analysis/plan_maps.py`，`Images/Result_Plots/Transfer/beam/map_seed<seed>.png`）：
每張圖左邊為微調前、右邊為微調後；灰色 = 兩個規劃都有的充電站（含 29 個既有站），綠色 = 只在微調前，藍色 = 只在微調後。

| 種子 | 站數（前 → 後） | 微調新增的站點 | 微調移除的站點 | 兩者共有 |
|---|---|---|---|---|
| 1 | 403 → 373 | 161 | 191 | 212 |
| 2 | 231 → 231 | 5 | 5 | 226 |
| 3 | 292 → 293 | 4 | 3 | 289 |
| 4 | 285 → 342 | 304 | 247 | 38 |
| 5 | 294 → 294 | 0 | 0 | 294 |

- 種子 1：微調把北部、東部山區零散的充電站移到市區路網密集處，以較少的站數（−30 站）得到較高的分數（+6.8%）。
- 種子 4：微調幾乎重新配置整個規劃，原本集中在西南與南側的站點改為更平均地分布於全市，站數增加 57 站，
  分數提升 8.2%，平均移動時間由 1.22 降為 1.08 分鐘、平均等待時間由 0.113 降為 0.104 分鐘。
- 種子 2、3 只調整少數站點；種子 5 的搜尋沒有找到比原規劃更好的配置，規劃不變。
- 所有充電站最後都配置滿 15 支樁（K = 15）。

**微調過程紀錄**：

| 種子 | 步數 | 嘗試的動作數 | 執行時間 | 記憶體峰值（PSS） |
|---|---|---|---|---|
| 1 | 659 | 5,488 | 27.8 分 | 6.7 GB |
| 2 | 516 | 4,070 | 11.8 分 | 4.1 GB |
| 3 | 578 | 4,606 | 19.0 分 | 4.6 GB |
| 4 | 630 | 5,028 | 14.9 分 | 6.8 GB |
| 5 | 579 | 4,620 | 19.2 分 | 4.6 GB |

- 逐步 log：`logs/beam/B4k2_e<seed>.log`（每 25 步記錄候選數、最佳分數、站數、剩餘預算）；規劃：`Results/Transfer/plans/B4k2_e<seed>/`。
- 圖：
  - `Images/Result_Plots/Transfer/beam/before_after.png`：每個種子微調前後的分數。
  - `Images/Result_Plots/Transfer/beam/search_progress.png`：搜尋過程中最佳分數隨步數的變化（微調前後對照）。
  - `Images/Result_Plots/Transfer/beam/service_metrics.png`：既有規劃、微調前、微調後的平均移動時間與平均等待時間。

### 10.4 總結

| 驗證項目 | 結果 |
|---|---|
| 預訓練（台南） | 來源模型 improvement 8.532，高於原始模型（7.909）與隨機策略 |
| 服務目標（< 15 分鐘） | 台南：平均移動 2.178 分、平均等待 0.098 分（來源模型；原始模型 1.771 / 0.365 分）；台北微調前 1.199 / 0.110 分、微調後 1.172 / 0.111 分，全部達成 |
| 泛化（台北 zero-shot） | 不經台北訓練即達 2.3239，比隨機策略高 85.0%，5 個種子全部高於隨機策略的最佳結果 |
| 微調（台北 beam search） | 2.3239 → **2.4012（+3.3%）**；4 / 5 個種子提升、0 個下降，最高 +8.2%；平均移動時間 −2.3%、最大等待時間 −7.7% |
| 可重現性 | B = k = 1 與 zero-shot 逐位元相同；同一設定重跑結果逐位元相同 |
| 安全性 | 程式碼：SonarQube 0 個安全性問題、0 個 security hotspot、0 個 bug（11.7）；相依套件：升級後 49 個套件 **0 個已知弱點**（舊環境 120 個，11.9、11.10） |
| 程式品質 | 140 個單元測試在新舊兩個環境全部通過；SonarQube Quality Gate **通過**，0 個 issue，安全性／可靠性／可維護性評等 A / A / A，全專案測試覆蓋率 95.9%（11.11） |
| 交付物 | 含模型權重的 Docker 映像 `pcrl:2.0-model`（映像內重跑兩個模型的台南規劃，與報告逐站相同）、trackio 格式的訓練與微調紀錄（12 個 CSV，以 trackio 實際匯入驗證）、`MODEL_CARD.md`（11.11） |
| 環境升級 | Python 3.12、torch 2.15、SB3 2.9；既有模型直接載入，台北與大安區的規劃與舊環境逐站相同，分數差異 < 1e-12（11.10） |
| 影像紀錄 | 訓練曲線、台南服務指標、台南地圖、泛化比較、微調前後、搜尋過程、台北服務指標、台北微調前後充電站分布圖（第 12 節） |

### 10.5 示範：台北大安區（zero-shot + beam search）

台南來源模型直接用於大安區，再以 beam search（B = 4、k = 2）微調；每個種子 zero-shot 約 4 秒、beam search 8–47 秒，
記憶體峰值約 0.5 GB。2026-11-12 試跑結果（score improvement，相對既有規劃 0.2333）：

| episode 種子 | zero-shot | beam search | 變化 | 站數 |
|---|---|---|---|---|
| 1 | 1.8589 | 1.8987 | +2.1% | 74 → 77 |
| 2 | 1.9393 | 2.0319 | +4.8% | 93 → 99 |
| 3 | 1.5893 | 2.0163 | +26.9% | 52 → 100 |
| 4 | 1.9052 | 2.1158 | +11.1% | 80 → 111 |
| 5 | 2.2050 | 2.2485 | +2.0% | 131 → 138 |
| 平均 | 1.8995 | **2.0623** | **+8.6%** | |

種子 1 的平均移動／等待時間：既有 4 座 2.231 / 0.949 分，zero-shot 0.675 / 0.065 分，beam search 0.681 / 0.065 分。

```bash
C=Taipei_Daan; M=runs/source_Tainan/best_source.zip; OUT=Results/$C; mkdir -p $OUT
for e in 1 2 3 4 5; do
  python -m transfer.evaluate --city $C --policy model --model $M --label zeroshot --eval-seed $e \
      --csv $OUT/zeroshot.csv --save-plan $OUT/zeroshot_e$e
  python -m transfer.beam_search --city $C --model $M --beam 4 --topk 2 --eval-seed $e --label B4k2 \
      --csv $OUT/beam.csv --save-plan $OUT/B4k2_e$e
done
python -m analysis.plan_metrics --city $C --existing --csv $OUT/metrics.csv
python -m analysis.plan_metrics --city $C --plan $OUT/zeroshot_e1 --label zeroshot_e1 --csv $OUT/metrics.csv
python -m analysis.plan_metrics --city $C --plan $OUT/B4k2_e1 --label B4k2_e1 --csv $OUT/metrics.csv
python Visualisierung/visualise.py --city $C --plan $OUT/B4k2_e1/plan_RL.pkl --output Images/Result_Plots/$C/B4k2_e1.png
```

---


## 11. 工程紀錄

### 11.1 觀測的數值穩定性

- 樁數為 [0,0,0] 的充電站（動作 4 拆光）服務率為 0，等候＋充電時間為 inf/NaN。
- 處理：`station_need` 遇到服務率 ≤ 0 或非有限值時回傳 1e6；觀測含非有限值直接報錯；評估拒絕含 NaN 權重的模型；
  來源模型選擇排除 ≤ learning_starts 的 checkpoint（尚未訓練的初始網路）。

### 11.2 環境快取與 episode 獨立性

- env_plus / evaluation_framework 的距離與成本快取（`node_dict`、`cost_dict`、`_OSM_ROUTE_CACHE`）原本不會清空，
  長時間訓練時記憶體持續成長，且節點屬性（指派的充電站、距離、旅行時間）會殘留到下一個 episode。
- 處理：`TransferStationPlacement.reset()` 每個 episode 開始時清空快取並還原節點屬性（8.3）。
  每行程記憶體 ≤ 1.4 GB，連續 episode 結果相同，且第一個 episode 與原本逐位元相同。

### 11.3 平行寫入 CSV

多個評估行程同時寫入時，以 `fcntl.flock` 鎖檔，並在檔案為空時才寫標頭。

### 11.4 效能

- 覆蓋計算向量化後，台南單一 episode 由 2,508 秒降為 1,163 秒（約 2.2 倍），獎勵完全相同（7.023852，652 步）。
- 速度參考：單行程約 1.1 秒/步；14 行程平行時約 1.6–2.7 秒/步。

### 11.5 fork 與亂數狀態（beam search）

- Python 3.7 起，`random` 模組會在每個 fork 出來的子行程中以系統亂數重新設定種子（`os.register_at_fork`）。
  環境在所有充電站都已滿樁時（動作 2/3）或沒有可移的樁時（動作 4）會用 `random.choice` 選點，
  所以 fork 出來的候選行程會選到與原本 episode 不同的點。
- 處理：`transfer/beam_search.py` 的 `fork_keep_random()` 在 fork 前記下 `random.getstate()`，子行程立即 `setstate()`。
- 驗證：B = k = 1 在 episode 種子 1–5 的結果與 `transfer/evaluate.py` 的 zero-shot 結果逐位元相同
  （improvement、站數、步數全部一致，10.3）。

### 11.6 記憶體保護

beam search 同時有多個行程，每個行程各自保有環境複本。三層保護避免主機記憶體耗盡：

1. `transfer/beam_search.py` 的監控執行緒：所有候選行程的記憶體（PSS）超過 `--max-mem-gb`，或主機可用記憶體低於
   `--min-available-gb` 時，立即結束整個 process group。
2. `scripts/run_beam_search.sh`：每個 run 在獨立的 cgroup 中執行（`systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0`），
   到達上限時只有該 run 被終止。
3. 同時執行的 run 數由 `JOBS`（預設 2）控制。

實測記憶體峰值（PSS）：B = k = 1 為 1.1–1.6 GB／run；B = 4、k = 2 為 4.1–6.8 GB／run（10.3）。
保護機制實際運作過一次：上限設為 7 GB 時，種子 4 在 7.2 GB 被監控執行緒安全終止，其他 run 與主機均不受影響；
預設值隨後調整為 cgroup 12 GB、監控 11 GB，重跑後正常完成（結果與中止前的逐步紀錄完全一致）。

### 11.7 安全性掃描（SonarQube，2026-11-11）

- 工具：SonarQube Community Build 26.11（`sonarqube:community`）+ `sonarsource/sonar-scanner-cli`，設定檔
  `sonar-project.properties`（repo 根目錄）。
- 範圍：全部程式碼，24 個檔案、3,398 行（Python 23 個、`docker/Dockerfile`）；輸入資料、輸出、快取與 `docker/.env` 不在掃描範圍。

**第一次掃描**：安全性 5 項（高 2、中 2、低 1）、可靠性 1 項，已全部處理：

| 規則 | 位置 | 問題 | 處理 |
|---|---|---|---|
| S5443（高） | `transfer/beam_search.py` | Unix socket 建在共用的 `/tmp`，檔名可預測 | 改用 `tempfile.mkdtemp()` 建立只有本人可存取（0700）的暫存資料夾，結束時整個刪除 |
| S6470（高） | `docker/Dockerfile` | `COPY . ...` 遞迴複製整個 build context，可能把敏感檔案帶進映像 | 改為逐一列出要複製的程式碼與 `Graph/`（仍保留 `Dockerfile.dockerignore` 把關） |
| S6471（低） | `docker/Dockerfile` | 容器預設以 root 執行 | 新增非 root 使用者 `pcrl` 作為預設使用者 |
| S6504（低，修正後出現） | `docker/Dockerfile` | 複製進映像的程式可被執行使用者修改 | 程式檔屬於 root、對執行使用者唯讀；拿掉原本的 `chmod -R a+rwX` |
| S2245（中）× 2 | `env_plus.py` | 使用 `random` 偽亂數 | 用途是模擬中隨機選空節點，且必須固定種子才能重現結果，不涉及安全；在程式中註明理由（`# NOSONAR`） |
| S5855（可靠性） | `analysis/beam.py` | 正規表示式中多餘的分支 | 移除 |

另外主動強化（SonarQube 社群版未要求）：讀取節點清單（`node_list_<city>.txt`、`nodes_RL.txt`）時以
`ast.literal_eval` 取代 `eval`（`evaluation_framework.py`、`test_solution.py`、`analysis/plan_metrics.py`、`analysis/beam.py`
共 6 處），資料檔無法再執行任意程式碼；所有資料檔解析結果與原本完全相同，台南指標重算結果與原 log 逐位元一致。
OCM API key 已從程式碼移到 `docker/.env`（3.3、4.5）。

**重新掃描**：

| 指標 | 結果 |
|---|---|
| Vulnerabilities（安全性問題） | **0** |
| Security Hotspots | **0** |
| Bugs（可靠性問題） | **0** |
| Security / Reliability / Maintainability rating | **A / A / A** |

重新執行：啟動 SonarQube（`docker run -d -p 127.0.0.1:9000:9000 sonarqube:community`），在網頁或 API 建立分析 token，
先執行單元測試產生覆蓋率報告（11.8），再於 repo 根目錄執行 `sonar-project.properties` 開頭註解中的 scanner 指令。
2026-11-11 加入 `analysis/plan_maps.py` 與單元測試後重新掃描，程式碼結果同上表，Quality Gate 通過（11.8）。
2026-11-13 起掃描也匯入相依套件弱點（11.9）：舊環境的 15 個套件弱點一度使 Vulnerabilities = 15、安全性評等 E；
升級環境後回到 0 個問題、評等 A，Quality Gate 通過（11.10）。

### 11.8 單元測試與 Quality Gate（2026-11-11）

- 工具：pytest 9.2.0 + pytest-cov 7.1.0（目前環境，`docker/requirements-test.txt`；舊環境 pytest 8.3.5，`docker/legacy-py38/requirements-test.txt`）；
  測試在 `tests/`，設定檔 `pytest.ini`、`.coveragerc`。
- 測試資料：`tests/conftest.py` 產生一座合成小城市（5 × 5 路網、26 個節點、2 座既有充電站），檔案格式與
  `Graph/<city>/` 相同，因此測試只需數秒，也不依賴台南／台北的大型資料檔。分析腳本的測試在暫存資料夾中執行，
  不會覆寫 `Results/`、`Images/` 的正式結果。
- 執行：`python -m pytest --cov --cov-report=xml`，140 個測試在新舊兩個環境全部通過，約 12–15 秒（2026-11-20）。

| 測試檔 | 內容 |
|---|---|
| `test_evaluation_framework.py` | haversine、速限解析（含 mph）、路徑計算（單行道改用無向圖、無路可走時報錯）、向量化覆蓋與原版逐點計算相同、充電站屬性、超載時等待時間為無限大、限制檢查訊息、既有規劃的正規化分數 |
| `test_service_metrics.py` | 平均移動／等待時間的加權方式（手算驗證）、節點指派、`analysis/plan_metrics.py` 輸出的 CSV 欄位與數值 |
| `test_environments.py` | 充電配置表（最便宜配置）、啟發式選點、拆樁、預算用盡、原始環境通過 gym 介面檢查、完整 episode（5 種動作）、遷移環境觀測維度與範圍、同一種子 episode 逐位元重現 |
| `test_beam_search.py` | 行程間訊息傳遞（大於 socket 緩衝區）、規劃簽章與充電站順序無關、fork 後子行程保留亂數狀態（11.5）、記憶體讀取、Q 值與模型貪婪動作一致 |
| `test_pipeline.py` | 原始 PCRL 流程以 `--city` 指定城市：訓練 30 步並存模型（每 10 步印出平均移動／等待時間）→ 產生規劃（自動選最新模型）→ 服務指標 → 地圖 |
| `test_analysis.py` | 訓練 log 解析、訓練曲線、台南服務指標圖、泛化摘要、beam search 摘要（驗證、統計檢定、逐種子服務指標）、台北微調前後充電站分布圖、路網與既有充電站分布圖 |
| `test_dependency_audit.py` | 相依套件弱點報告（11.9）：CVSS 3.x 分數（標準範例向量）、嚴重度來源優先順序、修正版本、套件鎖定行號、SonarQube 匯入格式（不連網，以假資料測試） |
| `test_preprocessing.py` | 前處理腳本（OSM 路網、OCM 充電站、節點清單）：以假資料取代 Overpass／Nominatim／Open Charge Map 請求，不連網；金鑰來源順序、功率對應、既有規劃、合成需求與地價、由大路網切出小地圖 |
| `test_transfer_scripts.py` | 跨城市程式的命令列流程：來源模型訓練（checkpoint 與最終模型）、模型與隨機策略評估、拒絕含非有限值的權重、beam search（B = k = 1 與評估相同、較寬的 beam 存規劃、步數上限）、候選行程指令迴圈與記憶體保護 |
| `test_trackio_logs.py` | trackio 紀錄轉換（11.11）：欄名經 trackio 簡化規則後不變、探索率排程、原始模型與來源模型訓練紀錄、beam search 紀錄解析、空欄位省略、輸出檔 |

各模組覆蓋率（pytest-cov，以敘述計；2026-11-20，140 個測試）：

| 模組 | 覆蓋率 |
|---|---|
| `evaluation_framework.py` | 100% |
| `env_plus.py`、`gym_compat.py` | 97%、92% |
| `reinforcement.py`、`model_eval.py`、`test_solution.py` | 99%、98%、99% |
| `transfer/`（env、train、evaluate、beam_search） | 99%、98%、99%、95% |
| `analysis/`（beam、generalization、plan_maps、plan_metrics、training_plots、trackio_logs） | 98–99% |
| `analysis/data_maps.py` | 86% |
| `Preprocessing/`（build_osm_graph、build_pcrl_nodes、download_ocm、build_ocm_plan、load_graph） | 96–100% |
| `Preprocessing/nodes_preparation.py` | 44%（原始 PCRL 的 Hannover 範例程式，本專案未使用，4.1） |
| `Visualisierung/visualise.py` | 86% |
| `scripts/dependency_audit.py` | 96% |
| 全部 | **95.9%** |

SonarQube 結果（2026-11-11，加入覆蓋率報告後重新掃描；2026-11-13 的結果見 11.9）：

| 指標 | 結果 |
|---|---|
| Quality Gate | **通過（Passed）** |
| 新程式碼測試覆蓋率（門檻 ≥ 80%） | **93.8%** |
| 新程式碼重複率（門檻 ≤ 3%） | 0.0% |
| 新程式碼問題數（門檻 0） | 0 |
| 全專案測試覆蓋率 | 62.5%（前處理下載、來源模型訓練等長時間腳本以實際執行驗證，未納入單元測試） |

### 11.9 相依套件弱點掃描（pip-audit，2026-11-13）

- 範圍：Docker 映像實際安裝的 Python 套件（含 pip / setuptools / wheel）加上測試套件；Python 直譯器本身與 openssl 等
  非 Python 函式庫不在範圍內。本節是舊環境 `pcrl:1.0` 的結果（48 + 7 = 55 個套件，紀錄在 `Results/Security/legacy-py38/`）；
  升級後的結果見 11.10。torch 的 CPU 版（`2.15.0+cpu`）以 PyPI 上的同一版本 `2.15.0` 比對。
- 工具：pip-audit 2.9.0（PyPA），比對 PyPI / OSV 弱點資料庫；只送出套件名稱與版本。嚴重度取 GitHub advisory 評等
  （OSV 資料庫），沒有評等時以 CVSS 3.x 向量計算分數，兩者都沒有的標為未評級。
- 執行：`scripts/dependency_audit.sh`（約 30 秒）→ `Results/Security/pip_audit.json`（原始結果）、`dependency_audit.csv`
  （每個弱點一列）、`dependency_audit_summary.txt`（每個套件一列），以及給 SonarQube 匯入的 `dependency-audit-sonar.json`。
- SonarQube：`sonar-project.properties` 以 `sonar.externalIssuesReportPaths` 匯入，每個有弱點的套件一個 issue，標在鎖定該套件的那一行
  （`docker/requirements-*.txt` 或 `docker/Dockerfile`），之後每次掃描都會自動列管。

**結果**：55 個套件中 15 個有已知弱點，共 120 個（critical 11、high 57、medium 40、low 8、未評級 4）。

| 套件 | 版本 | 弱點數 | 最高嚴重度 | 全部修正的版本 |
|---|---|---|---|---|
| pillow | 9.1.1 | 54 | critical | 13.0.0（另 2 個尚無修正） |
| torch | 1.12.0 | 24 | critical | 2.13.0（另 5 個尚無修正） |
| certifi | 2022.6.15 | 4 | critical | 2026.10.18 |
| brotli | 1.0.9 | 2 | critical | 1.2.1 |
| fiona | 1.8.21 | 2 | critical | 1.10b2 |
| urllib3 | 1.26.9 | 10 | high | 2.8.0 |
| setuptools | 65.5.0 | 5 | high | 84.1.0 |
| fonttools | 4.33.3 | 2 | high | 4.60.2 |
| geopandas | 0.11.0 | 1 | high | 1.1.2 |
| pip | 23.1.2 | 7 | medium | 26.2 |
| requests | 2.28.1 | 4 | medium | 2.33.0 |
| idna | 3.3 | 2 | medium | 3.15 |
| pytest | 8.3.5 | 1 | medium | 9.0.3 |
| scikit-learn | 1.3.0 | 1 | medium | 1.5.0 |
| zipp | 3.8.0 | 1 | medium | 3.19.1 |

**SonarQube 掃描（舊環境，2026-11-13 升級前）**：

| 指標 | 結果 |
|---|---|
| Quality Gate | **未通過（Failed）**：新程式碼問題數 13（門檻 0），全部是匯入的相依套件弱點 |
| 新程式碼測試覆蓋率（門檻 ≥ 80%） | 90.3%（通過） |
| 新程式碼重複率（門檻 ≤ 3%） | 0.0%（通過） |
| Vulnerabilities / 安全性評等 | 15 / E（全部來自相依套件；程式碼本身 0 個） |
| Bugs / Security Hotspots | 0 / 0 |
| 全專案測試覆蓋率 | 64.7%（29 個檔案、3,707 行） |

這些版本原本是為了重現訓練結果而鎖定的（Python 3.8、torch 1.12、stable-baselines3 1.5、gym 0.21），多數修正版本已不支援
Python 3.8（例如 pillow 13、torch 2.13），因此改為升級整個環境並驗證既有模型與結果（11.10）。

### 11.10 環境升級與重現驗證（2026-11-13）

為修正 11.9 的相依套件弱點，環境升級為 Python 3.12（conda 環境 `PCRL-py312`、Docker 映像 `pcrl:2.0`，版本見 3.1）。
舊環境保留在 `docker/legacy-py38/`（`pcrl:1.0`），作為訓練模型與本文件數值的產生環境。

**程式修改**（新舊兩個環境都能執行）：

- `gym_compat.py`：依 stable-baselines3 版本選用 gym 0.21 或 gymnasium；環境的 `reset()` / `step()` 依所用函式庫回傳
  對應格式，腳本一律透過 `gym_compat.reset()` / `step()` 取得原本的格式；`load_dqn()` 在 CPU 上載入模型，並以環境的
  observation / action space 取代舊模型檔中 SB3 2.x 無法讀取的 gym 0.21 物件（網路權重不變）。
- 使用者：`env_plus.py`、`transfer/env.py`、`transfer/evaluate.py`、`transfer/beam_search.py`、`model_eval.py` 與測試。
- osmnx 2：`ox.config` 改為 `ox.settings`；`truncate_graph_polygon` 不再有 `retain_all`，改以 `largest_component`
  保留最大弱連通部分；geopandas 1.x 的 `unary_union` 改為 `union_all`（舊版仍用原寫法）。

**弱點掃描**：`pcrl:2.0` 的 43 個套件加上測試套件，共 49 個，**0 個已知弱點**（`Results/Security/`）。

**重現驗證**（同一模型檔 `runs/source_Tainan/best_source.zip`，結果在新環境重新產生後與舊環境紀錄比較）：

| 項目 | 舊環境 | 新環境 | 比較 |
|---|---|---|---|
| 台北 zero-shot（種子 1） | improvement 2.4756452779578244、403 站、688 步 | 2.4756452779580473、403 站、688 步 | 規劃逐站相同（站點與各類充電樁數），分數差 2.2e-13 |
| 大安區 zero-shot（種子 1、3） | 1.8588917694329208 / 1.5893022066346283 | 1.8588917694329221 / 1.5893022066346234 | 規劃逐站相同，分數差 < 1e-14 |
| 大安區 beam search B4k2（種子 1、3） | 1.8987150021890278、77 站 / 2.0163047408223203、100 站 | 1.898715002189029、77 站 / 2.0163047408223265、100 站 | 規劃逐站相同，搜尋步數相同 |
| 大安區服務指標（beam search 種子 1） | 既有 2.231 / 0.949 分，規劃 0.681 / 0.065 分 | 相同 | 相同 |
| 原始 PCRL 模型（`model_eval.py`，西中區模型 `best_model_Tainan_WestCentral_400.zip`） | score 5.676077559707788、328 站 | 5.676077559707786、328 站 | 規劃逐站相同（站點與各類充電樁數），分數差 2e-15 |
| 前處理（大安區路網切割、台北與大安區 OCM 對應、台北節點清單） | | | 輸出完全相同（節點與邊、對應表、既有規劃、節點清單檔逐位元相同） |
| 單元測試 | 79 個通過 | 79 個通過 | |
| 台北 zero-shot 執行時間 | 832.7 秒 | 631.0 秒 | 約快 24% |
| 訓練（`transfer/train.py` 大安區、`reinforcement.py` 西中區，各 300 步） | | 正常完成 | checkpoint 可由 `transfer/evaluate.py` 載入評估 |

模型的每一步決策都相同；分數只在第 13 位小數以後有差異，來自新版 numpy 的浮點加總順序。本文件的數值（舊環境產生）
因此不需重算。載入舊模型時 SB3 2.x 會提示 optimizer 狀態格式不同，只影響接續訓練，不影響推論。

驗證後 `scripts/run_generalization.sh`、`scripts/run_beam_search.sh` 的預設 `PY` 改為新環境。評估與 beam search 的結果與
舊環境相同；重新訓練則因 torch / SB3 版本不同，亂數序列與舊環境不同，訓練出的模型不會與本文件的模型逐位元相同，
要逐位元重現訓練請用舊環境（`PY=$HOME/anaconda3/envs/PCRL/bin/python`）。

**SonarQube 重新掃描**（升級後，2026-11-13）：

| 指標 | 結果 |
|---|---|
| Quality Gate | **通過（Passed）** |
| 新程式碼測試覆蓋率（門檻 ≥ 80%） | 88.9% |
| 新程式碼重複率（門檻 ≤ 3%） | 0.0% |
| 問題數（Issues） | **0**（相依套件弱點 0、程式碼 0） |
| Vulnerabilities / Bugs / Security Hotspots | 0 / 0 / 0 |
| Security / Reliability / Maintainability rating | **A / A / A** |
| 全專案測試覆蓋率 | 65.1%（28 個檔案、3,779 行） |

### 11.11 交付物：模型映像、trackio 紀錄、model card（2026-11-20）

**1. 含模型權重的 Docker 映像 `pcrl:2.0-model`**（3.3，`docker/Dockerfile.model`）

| 項目 | 內容 |
|---|---|
| 內容 | `pcrl:2.0` 的環境、程式碼與 `Graph/` 資料，加上原始 PCRL 模型（132.7 MB）、來源模型（2.4 MB）與 `MODEL_CARD.md` |
| 大小 | 1.83 GB |
| 權重檢查 | 映像內兩個模型檔的 sha256 與 repo 中的檔案相同（`docker run --rm pcrl:2.0-model` 印出） |
| 映像內推論（不掛載 repo，2026-11-20） | 原始 PCRL 模型台南規劃 755 站（score 8.142435537145035），與 `Results/Tainan/` 逐站相同，`nodes_RL.txt` 逐位元相同；來源模型台南 improvement 8.532322275639508（舊環境 8.532322275639117）、405 站、581 步，規劃逐站相同；大安區 zero-shot 1.8588917694329221、74 站 |
| 推論時間與記憶體（台南完整貪婪 episode） | 原始 PCRL 模型 3,031 秒、3.51 GB（舊環境紀錄 3,957 秒）；來源模型 945 秒、2.09 GB（舊環境 1,020 秒） |
| 短訓練（台南 300 步，第 100 步起更新網路） | 原始 PCRL 模型 381.7 秒、1.86 GB；來源模型（8 執行緒）300.0 秒、1.36 GB；replay buffer 填滿時另需 4.8 GB / 0.12 GB（依程式計算） |
| 交付檔 | `docker save pcrl:2.0-model \| gzip > dist/pcrl_2.0-model.tar.gz`（`dist/` 不進 git），以 `docker load -i` 載入 |

**2. trackio 格式的訓練與微調紀錄**（`analysis/trackio_logs.py` → `Results/Trackio/`）

trackio（Hugging Face）的 `trackio.import_csv()` 讀取第一欄 `step`、選用的 `timestamp`（ISO 8601）與數值指標欄，每個 CSV
匯入成一個 run；欄名只保留前 10 個英數字或 `/`，因此所有欄名都在這個範圍內，匯入後名稱不變。

| 檔案 | 列數 | 來源 | 主要欄位 |
|---|---|---|---|
| `training/original_Tainan.csv` | 49 | `tmp_Tainan/train.log`（每 400 步）＋評估 episode | `reward/avg`、`reward/max`、`saved`、`epsilon`、`eval/imp`、`eval/stns` |
| `training/source_Tainan.csv` | 51 | `runs/source_Tainan/monitor.csv`（31 個 episode）＋`source_selection.csv`（20 個 checkpoint） | `timestamp`、`hours`、`reward`、`length`、`epsilon`、`eval/imp`、`eval/stns` |
| `finetuning/Taipei_B4k2_e<1–5>.csv` | 22–29 | `logs/beam/B4k2_e*.log`（每 25 步） | `score`、`improve`、`live/best`、`done/best`、`key`、`beam`、`stations`、`budget`、`seconds` |
| `finetuning/Taipei_zeroshot_e<1–5>.csv` | 22–29 | `logs/beam/B1k1_e*.log`（B = k = 1 = zero-shot） | 同上 |

- 驗證：以 trackio 0.45.1 實際匯入全部 12 個檔案，從 trackio 資料庫讀回，每個檔案的筆數、指標名稱與數值都與 CSV 相同；
  來源模型 31 個訓練 episode 都帶有時間戳。
- `epsilon` 依訓練設定的線性排程計算；checkpoint 列沒有時間紀錄，`timestamp` 留空。
- 新的 beam search 紀錄若含平均移動／等待時間（9.3 的 `--log-every`），另外輸出 `travel`、`waiting` 欄。
- 單元測試 `tests/test_trackio_logs.py`（12 個）：欄名經 trackio 的簡化規則後不變、各紀錄的解析、空欄位省略、輸出檔。

**3. Model card**（`MODEL_CARD.md`）

依要求項目整理：各模組 AI 模型簡介與用途、預訓練資料集（來源、授權、統計）、參數數量與超參數意義、超參數對效能的
影響（原始模型：訓練步數與保存策略；來源模型：訓練步數與 checkpoint 選擇，選到的 17,000 步比最後一步高 42.7%；
微調：beam 寬度 B 與 k）、效能指標與綜合效能分數、訓練與測試細節及最終效能、訓練與推論參考時間、最低軟硬體需求、
使用手冊（執行環境、使用方式、輸入與輸出格式）。

**SonarQube 重新掃描**（2026-11-20，含上述程式與 `docker/Dockerfile.model`）：

| 指標 | 結果 |
|---|---|
| Quality Gate | **通過（Passed）** |
| 新程式碼測試覆蓋率 / 重複率 | 95.8% / 0.0% |
| 問題數（Issues） | **0**（掃描中發現的 1 個可靠性問題：`value != value` 判斷 NaN，已改為 `math.isnan`） |
| Vulnerabilities / Bugs / Security Hotspots | 0 / 0 / 0 |
| Security / Reliability / Maintainability rating | **A / A / A** |
| 全專案測試覆蓋率 | **95.9%**（30 個檔案、3,947 行） |
| 單元測試 | 140 個，新舊兩個環境全部通過 |

---

## 12. 影像紀錄與輸出檔清單

**影像紀錄**

| 檔案 | 階段 | 內容 |
|---|---|---|
| `Images/Result_Plots/Data/Tainan_network_stations.png` | 資料 | 台南路網（15,983 節點）與 17 座既有充電站（OCM 對應後，大小 = 充電樁數） |
| `Images/Result_Plots/Data/Taipei_network_stations.png` | 資料 | 台北路網（12,030 節點）與 29 座既有充電站；標出 10 個位於路網範圍外的 OCM 原始位置與其對應節點 |
| `Images/Result_Plots/Training/original_Tainan_training.png` | 訓練 | 原始 PCRL 模型在台南的訓練曲線（最近 3 個 episode 平均獎勵、保存的最佳模型） |
| `Images/Result_Plots/Training/source_Tainan_training.png` | 訓練 | 來源模型在台南的訓練 episode 獎勵與各 checkpoint 貪婪分數、選中的 17,000 步 |
| `Images/Result_Plots/Training/Tainan_service_metrics.png` | 訓練 | 台南既有充電站、原始 PCRL 模型、來源模型的平均移動時間與平均等待時間，標示 15 分鐘目標線 |
| `Images/Result_Plots/Tainan/RL_Tainan.png` | 訓練 | 原始 PCRL 模型產生的台南充電站地圖（755 站） |
| `Images/Result_Plots/Transfer/generalization.png` | 泛化 | 台南模型在台北（zero-shot）與隨機策略的比較 |
| `Images/Result_Plots/Transfer/beam/before_after.png` | 微調 | 每個 episode 種子的微調前（zero-shot）與微調後（beam search）分數 |
| `Images/Result_Plots/Transfer/beam/search_progress.png` | 微調 | 每個 episode 種子的搜尋過程：最佳分數隨步數的變化，微調前後對照 |
| `Images/Result_Plots/Transfer/beam/service_metrics.png` | 微調 | 台北既有規劃、微調前、微調後的平均移動時間與平均等待時間 |
| `Images/Result_Plots/Transfer/beam/map_seed<1–5>.png` | 微調 | 台北微調前後的充電站分布圖（共有／只在微調前／只在微調後） |

配色：微調後（beam search）#2a78d6（藍）、zero-shot #1baf7a（綠）、參考 #52514e（灰），背景 #fcfcfb。

**結果與紀錄檔**

| 檔案 | 內容 |
|---|---|
| `tmp_Tainan/train.log` | 原始 PCRL 模型訓練紀錄 |
| `Results/Tainan/plan_RL.pkl`, `nodes_RL.txt` | 原始 PCRL 模型的台南充電規劃 |
| `logs/training/test_solution_Tainan.log` | 台南規劃評估指標 |
| `logs/training/original_model.log` | 原始 PCRL 模型的台南貪婪 episode |
| `runs/source_Tainan/monitor.csv`, `config.json`, `logs/training/source_Tainan.log` | 來源模型訓練紀錄與完整參數 |
| `Results/Transfer/source_selection.csv` | 來源模型 checkpoint 選擇 |
| `Results/Tainan/metrics.csv` | 台南服務指標：既有充電站、原始 PCRL 模型、來源模型 |
| `Results/Tainan/source_model/`, `logs/training/source_model_Tainan.log` | 來源模型的台南充電規劃與貪婪 episode 紀錄 |
| `Results/Transfer/random_Tainan.csv` | 台南隨機策略基準 |
| `Results/Transfer/generalization.csv`, `generalization_summary.txt` | 台北泛化：zero-shot、隨機策略 |
| `Results/Transfer/beam.csv`, `beam_summary.txt` | 台北微調：beam search 驗證與結果 |
| `Results/Transfer/plans/` | 台北規劃（`plan_RL.pkl`、`nodes_RL.txt`）：`zeroshot_e*`、`B1k1_e*`、`B4k2_e*`；服務指標 `metrics.csv`、`metrics_B4k2.csv` |
| `logs/beam/<label>_e<seed>.log` | beam search 逐步紀錄（每 25 步：候選數、最佳分數、站數、剩餘預算）與記憶體峰值 |
| `Results/Trackio/training/*.csv`, `Results/Trackio/finetuning/*.csv` | trackio 格式的訓練與微調紀錄（11.11） |
| `MODEL_CARD.md` | Model card（11.11） |

---

## 13. 重現步驟

```bash
conda activate PCRL-py312    # 或在 Docker 容器內（第 3.3 節）；scripts/run_*.sh 預設也用此環境
# 逐位元重現訓練：conda activate PCRL（docker/legacy-py38/），並以 PY=$HOME/anaconda3/envs/PCRL/bin/python 執行 scripts/run_*.sh

# 單元測試（11.8；需先安裝 docker/requirements-test.txt）
python -m pytest --cov --cov-report=xml

# 前處理（4.5；重現本文件結果時只需步驟 2、4，直接使用 repo 中的 graphml 與 OCM JSON）
python Preprocessing/build_pcrl_nodes.py --graph Graph/Taipei/Taipei.graphml --output Graph/Taipei/node_list_Taipei.txt
python Preprocessing/build_ocm_plan.py --graph Graph/Taipei/Taipei.graphml --ocm Graph/Taipei/ocm_taipei.json \
    --output-pkl Graph/Taipei/existingplan_Taipei.pkl --mapping-csv Graph/Taipei/ocm_osm_mapping_Taipei.csv --max-distance 100000

# 一、訓練：原始 PCRL 模型（台南）
python reinforcement.py --city Tainan      # 訓練 → tmp_Tainan/（--city 可換成其他城市，7.2）
python model_eval.py --city Tainan         # 產生規劃 → Results/Tainan/
python test_solution.py --city Tainan      # 指標
python Visualisierung/visualise.py --city Tainan

# 一、訓練：來源模型（台南，transfer/env.py）
scripts/run_generalization.sh source   # 訓練 20,000 步（約 14 小時）
scripts/run_generalization.sh select   # checkpoint 選擇 → runs/source_Tainan/best_source.zip
scripts/run_generalization.sh service  # 台南平均移動／等待時間（既有、原始模型、來源模型）→ Results/Tainan/metrics.csv
python -m analysis.data_maps           # 台南、台北路網與既有充電站分布圖 → Images/Result_Plots/Data/
python -m analysis.training_plots      # 訓練曲線、台南服務指標圖 → Images/Result_Plots/Training/

# 二、泛化（台北）
scripts/run_generalization.sh baselines
scripts/run_generalization.sh analyze

# 二、微調（台北，beam search）
scripts/run_beam_search.sh validate    # B = k = 1，與 zero-shot 逐位元相同
scripts/run_beam_search.sh search      # B = 4、k = 2
scripts/run_beam_search.sh analyze

# 交付物（11.11）
python -m analysis.trackio_logs         # trackio 格式的訓練與微調紀錄 → Results/Trackio/
docker build -f docker/Dockerfile.model -t pcrl:2.0-model .
```

重現性：

- 訓練：`--seed` 設定 torch / numpy / random，並啟用 `torch.use_deterministic_algorithms(True)`。
- 評估：`--eval-seed` 設定 `random` / `np.random`（環境中動作 2–4 的隨機選點）；beam search 在 fork 時保留亂數狀態（11.5）。
- 同一環境、同一模型、同一 episode 種子結果可逐位元重現（例：zero-shot 種子 1 = 2.4756452779578244（舊環境）、
  2.4756452779580473（新環境）；兩環境的規劃逐站相同，11.10）。
- 訓練只在同一環境內逐位元重現；本文件的模型由舊環境訓練。

---

## 14. 後續規劃

1. **擴大搜尋**：增加 beam 寬度 B 與每步嘗試的動作數 k（記憶體保護已就緒，可依主機資源調整 `JOBS`、`MEM_MAX`）。
2. **更多城市**：以相同流程（前處理 → zero-shot → beam search）套用到其他城市，進一步驗證泛化能力。
3. **實際需求資料**：以實際電動車登記、車流或充電紀錄取代路口連接數推估的需求與地價。
4. **動作設計**：在 top-K 候選點中選擇，以共享的逐點評分網路處理，讓模型同時學習選點與選規則。

---

## 15. 附錄

### 15.1 主要模型檔

| 檔案 | 說明 |
|---|---|
| `tmp_Tainan/best_model_Tainan_11200.zip` | 原始 env_plus 台南模型（8.19M 參數） |
| `runs/source_Tainan/best_source.zip` | transfer/env.py 台南來源模型（= checkpoint 17,000 步，150,277 參數），泛化與微調使用的預訓練模型 |
| `runs/source_Tainan/checkpoints/model_<N>_steps.zip` | 來源訓練每 1,000 步的 checkpoint |

### 15.2 載入模型範例

```python
import os
import random
import numpy as np
import gym_compat as gc          # 新舊兩個環境都適用（3.1）
from transfer import env as et   # 在 repo 根目錄執行

random.seed(1); np.random.seed(1)                              # episode 種子（同 --eval-seed 1）
env = et.make_env("Taipei", os.getcwd())
model = gc.load_dqn("runs/source_Tainan/best_source.zip", env)  # CPU 上載入
obs, done = gc.reset(env), False
existing_score = env.best_score
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = gc.step(env, int(action))
print("improvement", env.best_score - existing_score, "stations", len(env.best_plan))
```

### 15.3 名詞對照

| 名詞 | 說明 |
|---|---|
| improvement | 最佳規劃分數 − 既有規劃分數，等於 episode 總獎勵 |
| zero-shot | 預訓練模型不經任何訓練直接用在新城市 |
| beam search | 每步保留 B 個最佳候選規劃、每個候選嘗試預訓練模型 Q 值最高的 k 個動作的搜尋 |
| 貪婪 episode | 每步選 Q 值最大的動作（不探索）跑完一個 episode |
| checkpoint | 訓練中定期保存的模型 |
| episode 種子 | 設定環境隨機選點的種子；微調前後使用同一組種子做配對比較 |

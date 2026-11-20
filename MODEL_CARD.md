# PCRL 充電站佈建模型 Model Card

以深度強化學習（DQN）在城市路網上規劃電動車充電站位置與充電樁數量的模型。

| 項目 | 內容 |
|---|---|
| 版本 | 2026-11-20 |
| Docker 映像（環境＋模型權重） | `pcrl:2.0-model`（`docker/Dockerfile.model`，建立於 `pcrl:2.0` 之上） |
| 模型檔 | `tmp_Tainan/best_model_Tainan_11200.zip`（原始 PCRL 模型）、`runs/source_Tainan/best_source.zip`（來源模型） |
| 訓練與微調紀錄 | `Results/Trackio/`（trackio 可匯入的 CSV，第 10 節） |
| 完整技術文件 | `README_for_doc.md`（本文件各節標註對應章節） |

**本文件各節與要求項目對照**

| 要求項目 | 章節 |
|---|---|
| 各模組之 AI 模型簡介與具體用途 | 1 |
| 用於各 AI 模型預訓練之資料集相關資訊 | 2 |
| 模型參數數量與重要超參數之意義說明 | 3 |
| 各 AI 模型之至少一個重要超參數對效能影響之說明 | 4 |
| 模型效能指標與綜合效能分數 | 5 |
| 各 AI 模型之訓練細節、測試細節與最終效能數據 | 6 |
| 模型進行訓練與推論所需之參考時間 | 7 |
| 各 AI 模型於訓練與推論階段所需之最低軟硬體需求 | 8 |
| AI 模型使用手冊（使用方式、執行環境、輸入與輸出格式規範） | 9 |

---

## 1. 各模組之 AI 模型簡介與具體用途

**問題**：在固定預算（5,000,000）下，決定在城市路網的哪些路口新建充電站、每站配置多少支 7 / 22 / 50 kW 充電樁（每站最多
15 支），讓充電服務涵蓋更多需求，並降低車主開車到充電站的平均移動時間與在站內的平均等待時間。

**做法**：把規劃過程建模為強化學習環境。每一步模型選擇 5 種佈建動作之一（新建充電站、在既有站加樁、移樁，見 3.1），
環境修改規劃並以效用模型計算綜合效能分數；分數創新高時給正獎勵。訓練好的模型一步一步執行到預算用完，即產生完整的
充電規劃。

| 模組 | 類型 | 檔案 | 參數量 | 具體用途 |
|---|---|---|---|---|
| **原始 PCRL 模型** | DQN（Q-network，多層感知器） | `tmp_Tainan/best_model_Tainan_11200.zip` | 8,187,909 | 依原始 PCRL 方法在台南訓練，產生台南的充電規劃（755 站）。觀測包含每個路口的資料，維度隨城市而變，只能用於台南 |
| **來源模型**（source model） | DQN（Q-network，多層感知器） | `runs/source_Tainan/best_source.zip` | 150,277 | 改用固定 324 維、依城市正規化的觀測，同一個模型可直接用於任何城市；產生台南規劃（405 站），並作為台北泛化（zero-shot）與微調的預訓練模型 |
| **微調模組**（模型引導 beam search） | 以來源模型 Q 值引導的搜尋 | `transfer/beam_search.py`（沿用來源模型，不另存權重） | 0（使用來源模型） | 在新城市以來源模型為基礎，同時保留多個候選規劃並嘗試 Q 值最高的動作，找出比 zero-shot 更好的規劃；模型權重不變，微調的對象是規劃 |
| 效用模型（評分） | 解析式模型（非學習式） | `evaluation_framework.py` | — | 計算規劃的綜合效能分數（覆蓋、移動、充電、等候），作為獎勵與所有評估的標準（5.1） |

**使用情境**：輸入一個城市的路網與既有充電站，輸出新的充電規劃（站點與三種充電樁數量），以及平均移動時間、平均等待時間
等服務指標，供充電站佈建方案的評估與比較。

**本專案的成果**（第 5、6 節）：

- 台南：兩個模型的規劃都把平均移動時間由 10.8 分鐘降到 2.2 分鐘以內、平均等待時間由 7.5 分鐘降到 0.4 分鐘以內，
  遠低於 15 分鐘的服務目標；來源模型以 1.8% 的參數量，綜合效能分數提升量（improvement）比原始模型高 7.9%。
- 台北（從未用於訓練）：來源模型不經任何台北訓練即達 improvement 2.3239，比隨機策略高 85.0%；
  再以 beam search 微調提升到 2.4012（+3.3%），5 個 episode 種子中 4 個提升、沒有任何一個下降。

---

## 2. 用於各 AI 模型預訓練之資料集相關資訊

強化學習不使用標註資料。模型的訓練樣本是模型與「城市環境」互動產生的經驗（狀態、動作、獎勵、下一個狀態），每個模型在
台南環境中收集 20,000 筆。城市環境由以下公開資料建構（`README_for_doc.md` 第 4 節）。

### 2.1 來源與授權

| 資料 | 來源 | 授權 | 在模型中的用途 |
|---|---|---|---|
| 可行車路網 | OpenStreetMap（osmnx `graph_from_place`，`network_type="drive"`、`simplify=True`、只保留最大連通路網） | ODbL 1.0（© OpenStreetMap contributors） | 路口 = 候選位置；道路長度與速限 → 最短路徑距離與旅行時間 |
| 既有充電站 | Open Charge Map API（國家 TW） | CC BY 4.0（Open Charge Map Contributors，資料紀錄中的授權欄位） | 既有充電規劃，作為分數的正規化基準與 episode 起點 |
| 充電需求、地價 | 由 OSM 的路口連接數 `street_count` 合成 | — | 路口屬性：`demand` = 0.05 + 0.95 × 正規化連接數；`estate price` = 0.1 + 0.9 × 正規化連接數 |

### 2.2 各資料集的角色與統計

| | 台南 Tainan | 台北 Taipei | 台北大安區 Taipei_Daan | 台南中西區 Tainan_WestCentral |
|---|---|---|---|---|
| 角色 | **預訓練**（兩個模型）與 checkpoint 選擇 | **泛化測試與微調**（從未用於訓練或選模型） | 泛化與微調的快速示範 | 訓練流程示範（屬於台南） |
| 路口數（候選位置 N） | 15,983 | 12,030 | 1,046 | 652 |
| 道路段數 | 42,978 | 28,486 | 2,148 | 1,846 |
| OCM 充電站 → 對應後既有充電站 | 17 → 17 | 33 → 29 | 4 | 2 |
| 既有充電樁 7 / 22 / 50 kW | 1 / 0 / 85 | 6 / 0 / 136 | 0 / 0 / 23 | 1 / 0 / 8 |
| 平均需求 / 平均地價 | 0.394 / 0.426 | 0.375 / 0.408 | 0.483 / 0.510 | 0.531 / 0.556 |
| 每個 episode 步數上限（N/2） | 7,992 | 6,015 | 523 | 326 |
| 收錄於 repo | 2026-10-19 | 2026-09-19 | 2026-11-12（由台北路網切出） | 2026-11-12（由台南路網切出） |

- OCM 充電站狀態更新日期：台南 2017-10-12 至 2025-03-04；台北 2019-04-11 至 2024-09-02。
- 前處理：OCM 充電站對應到最近的路網節點，同一節點的多個站合併；功率 ≤ 11 kW → 7 kW 樁、≤ 35 kW → 22 kW 樁、
  其餘 → 50 kW 樁。台北有 10 個 OCM 站位於台北市路網範圍外，對應到路網邊緣最近的節點。
- 檔案：`Graph/<city>/<city>.graphml`、`node_list_<city>.txt`、`existingplan_<city>.pkl`、`ocm_<city>.json`、
  `ocm_osm_mapping_<city>.csv`；分布圖 `Images/Result_Plots/Data/<city>_network_stations.png`。
- 重現性：以 repo 中的路網與 OCM 檔重跑前處理，輸出逐位元相同（新舊兩個執行環境皆同）。

---

## 3. 模型參數數量與重要超參數之意義說明

### 3.1 網路結構與參數數量

兩個模型都是 stable-baselines3 的 DQN `MlpPolicy`：輸入觀測，輸出 5 個動作各自的 Q 值（預估執行該動作後還能讓分數
提升多少），選 Q 值最大的動作。

| | 原始 PCRL 模型 | 來源模型 |
|---|---|---|
| 觀測 | 每個路口 8 個值（經緯度、需求、地價、私人充電樁、三種樁數）＋剩餘預算，共 8N + 1 = **127,865 維**（台南），float16 | 固定 **324 維**、依城市正規化到 [−1, 1]：全域狀態 16、動作候選點 28、高需求空路口 160、最需要支援的充電站 112、覆蓋分布 8，float32 |
| 網路 | Linear(127,865→64) → ReLU → Linear(64→64) → ReLU → Linear(64→5) | Linear(324→256) → ReLU → Linear(256→256) → ReLU → Linear(256→5) |
| 參數數量 | **8,187,909**（第一層佔 8,183,424） | **150,277**（83,200 + 65,792 + 1,285） |
| 權重檔大小 | 132.7 MB | 2.4 MB |
| 可用城市 | 只有台南（輸入維度隨路口數而變） | 任何城市 |

target network 與 Q-network 結構相同，只在訓練時使用。

**動作（兩個模型相同）**

| 動作 | 意義 |
|---|---|
| 0 | 在尚無充電站、覆蓋度最低的路口新建充電站 |
| 1 | 在尚無充電站、需求最高的路口新建充電站 |
| 2 | 在覆蓋度最低、未滿 15 支樁的充電站加一支 22 kW 樁 |
| 3 | 在需求最高、未滿 15 支樁的充電站加一支 22 kW 樁 |
| 4 | 從效益最低的新建站移一支樁到最需要支援的站 |

### 3.2 訓練超參數

| 超參數 | 原始 PCRL 模型 | 來源模型 | 意義 |
|---|---|---|---|
| total_timesteps | 20,000 | 20,000 | 訓練總步數（與環境互動的次數） |
| learning_rate | 1e-3 | 1e-3 | Adam 最佳化器的步長；越大更新越快但越不穩定 |
| net_arch | [64, 64] | [256, 256] | 隱藏層大小，決定模型容量與參數量 |
| buffer_size | 10,000 | 50,000 | replay buffer 容量：保留多少筆過去經驗供隨機抽樣訓練；來源模型大於總步數，保留全部經驗 |
| batch_size | 128 | 128 | 每次更新抽樣的經驗筆數 |
| learning_starts | 1,000 | 1,000 | 先收集這麼多步的經驗才開始更新網路 |
| train_freq / gradient_steps | 4 / 1 | 4 / 1 | 每 4 步做 1 次梯度更新 |
| target_update_interval | 1,000 | 1,000 | 每隔多少步把 Q-network 複製到 target network；間隔越長，學習目標越穩定 |
| gamma | 0.99 | 0.99 | 折扣因子：未來獎勵的權重，0.99 約等於看 100 步之後的效果 |
| exploration | ε 1.0 → 0.05（前 10% 步數） | 同左 | ε-greedy 探索：以機率 ε 隨機選動作，前 2,000 步由完全隨機線性降到 5% |
| loss / max_grad_norm | Huber / 10 | Huber / 10 | 損失函數與梯度裁剪，避免單次更新過大 |
| seed | 1 | 1 | torch / numpy / random 種子，並啟用決定性演算法 |
| 模型選擇 | 每 400 步檢查最近 3 個訓練 episode 的平均獎勵，創新高就存檔；使用最後存檔（11,200 步） | 每 1,000 步存 checkpoint；每個 checkpoint 在台南跑貪婪 episode，取 improvement 最高者（17,000 步） | 決定最後交付哪一個訓練階段的權重 |
| torch threads | 預設 | 8 | CPU 執行緒數 |

### 3.3 微調模組（beam search）超參數

| 超參數 | 值 | 意義 |
|---|---|---|
| B（`--beam`） | 4 | beam 寬度：每一步保留的候選規劃數；其中一個位置固定保留模型的貪婪路徑，因此結果不會比 zero-shot 差 |
| k（`--topk`） | 2 | 每個候選規劃嘗試 Q 值最高的 k 個動作 |
| value_weight | 1.0 | 候選排序分數 = 目前最佳分數 + value_weight × 模型預估的後續提升（max Q） |
| patience | 200 | 貪婪路徑結束後，最佳分數連續 200 步沒有提升就停止 |
| eval_seed | 1–5 | episode 種子（環境中少數隨機選點的種子），微調前後使用同一個種子配對比較 |

B = k = 1 時，搜尋等於模型的貪婪 episode（zero-shot），已驗證與 zero-shot 結果逐位元相同。

### 3.4 問題設定參數（效用模型）

| 參數 | 值 | 意義 |
|---|---|---|
| BUDGET | 5,000,000 | 總預算，用完 episode 即結束 |
| K | 15 | 每站最多充電樁數 |
| 充電樁功率 / 安裝費 | 7 / 22 / 50 kW；300 / 750 / 28,000 | 三種充電樁 |
| RADIUS_MAX | 1,000 m | 充電站最大服務半徑（隨充電能力增加） |
| alpha | 0.4 | 移動成本與（充電＋等候）成本的權重 |
| lambda | 0.5 | 效益（覆蓋）與成本的權重 |

---

## 4. 重要超參數對效能的影響

### 4.1 原始 PCRL 模型：訓練步數（total_timesteps）與模型保存

訓練過程中每 400 步檢查一次「最近 3 個訓練 episode 的平均獎勵」（含 ε = 0.05 的探索，`tmp_Tainan/train.log`）：

| 訓練步數 | 800 | 2,000 | 4,000 | 6,000 | 7,600 | 10,000 | **11,200** | 14,000 | 16,000 | 18,000 | 20,000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 平均訓練獎勵 | 7.02 | 7.18 | 7.32 | 7.69 | 7.79 | 7.61 | **7.79** | 7.57 | 7.59 | 7.48 | 7.58 |

- 800 → 7,600 步穩定上升（7.02 → 7.79，+11%），顯示訓練步數增加帶來明確的效能提升；7,600 步之後維持在 7.4–7.8。
- 因此 20,000 步的訓練長度已足以收斂；「創新高才存檔」的保存策略取得 11,200 步的最佳模型，比最後一步（7.58）高 2.8%。
  此模型的貪婪 episode improvement 為 **7.9091**。

### 4.2 來源模型：訓練步數與 checkpoint 選擇

每個 checkpoint 在台南跑一個貪婪 episode（不探索），improvement 如下（`Results/Transfer/source_selection.csv`）：

| 步數 | 1k* | 2k | 3k | 4k | 5k | 6k | 7k | 8k | 9k | 10k |
|---|---|---|---|---|---|---|---|---|---|---|
| improvement | 8.454 | 6.474 | 0.744 | 2.445 | 0.523 | 6.878 | 1.789 | 7.946 | 3.108 | 7.716 |
| **步數** | **11k** | **12k** | **13k** | **14k** | **15k** | **16k** | **17k** | **18k** | **19k** | **20k** |
| improvement | 3.296 | 0.523 | 2.929 | 3.345 | 3.272 | 6.765 | **8.532** | 8.130 | 6.639 | 5.978 |

\* 1k 為尚未開始更新的初始網路（≤ learning_starts），依規則排除。

- 訓練步數對貪婪策略的效能影響很大：相鄰 checkpoint 的 improvement 可在 0.5–8.5 之間變化。
- 以貪婪評估挑選 checkpoint，選到 17,000 步（**8.532**），比直接使用最後一步（20,000 步，5.978）高 **42.7%**，
  也高於原始 PCRL 模型（7.909）。訓練步數與 checkpoint 選擇是此模型最重要的超參數。

### 4.3 微調模組：beam 寬度 B 與每步嘗試動作數 k

台北 episode 種子 1–5，同一個來源模型（`Results/Transfer/beam.csv`）：

| 設定 | 平均 improvement | 各種子 improvement | 每個 run 執行時間 | 記憶體峰值 |
|---|---|---|---|---|
| B = k = 1（等於 zero-shot） | 2.3239 | 2.4756 / 2.0695 / 2.3019 / 2.3604 / 2.4119 | 341–856 秒（平均 531） | 1.1–1.6 GB |
| **B = 4、k = 2** | **2.4012（+3.3%）** | 2.6428 / 2.0954 / 2.3021 / 2.5539 / 2.4119 | 711–1,671 秒（平均 1,113） | 4.1–6.8 GB |

- 加大 B、k 讓搜尋涵蓋更多動作組合：5 個種子中 4 個提升（最高 +8.2%）、1 個持平、沒有任何一個下降；
  代價是執行時間約 2.1 倍、記憶體約 4 倍。
- 台北大安區同樣設定的示範：平均 improvement 1.8995 → 2.0623（**+8.6%**），每個種子 8–47 秒。

---

## 5. 模型效能指標與綜合效能分數

### 5.1 指標定義

| 指標 | 定義 | 方向 |
|---|---|---|
| **綜合效能分數**（score） | 效用模型 `norm_score`：`score = 0.5 × 覆蓋效益/覆蓋效益₀ − 0.5 × [0.4 × 移動成本/移動成本₀ + 0.6 × (充電時間/充電時間₀ + 等候時間/等候時間₀)] / 3`；下標 0 為既有充電站的值，因此既有規劃的分數為 **0.2333** | 越高越好 |
| **improvement** | score − 既有規劃 score（0.2333）；等於一個 episode 的總獎勵，是所有實驗的主要指標 | 越高越好 |
| 平均移動時間 | 開車到被指派充電站的時間（分鐘），依需求加權平均 | 越低越好 |
| 平均等待時間 | 在充電站排隊等候的時間（分鐘，M/D/1 等候模型），依到站車流加權平均 | 越低越好 |
| 最大等待時間 | 各充電站等候時間的最大值（分鐘） | 越低越好 |
| 充電站數、預算使用率 | 規劃的規模與成本 | 參考 |

覆蓋效益以調和數計算（同一路口被越多充電站的服務半徑涵蓋，效益遞減增加）；移動成本、充電時間、等候時間分別為所有
路口與充電站的加總（`README_for_doc.md` 5.4）。**服務目標**：平均移動時間與平均等待時間都低於 15 分鐘。

### 5.2 綜合效能分數一覽

| 模型 / 規劃 | 城市 | 綜合效能分數 | improvement | 充電站數 | 平均移動時間（分） | 平均等待時間（分） | 最大等待時間（分） |
|---|---|---|---|---|---|---|---|
| 既有充電站 | 台南 | 0.2333 | 0 | 17 | 10.806 | 7.457 | 56.842 |
| 原始 PCRL 模型 | 台南 | 8.1424 | 7.9091 | 755 | **1.771** | 0.365 | 8.456 |
| **來源模型** | 台南 | **8.7657** | **8.5323** | 405 | 2.178 | **0.098** | **0.457** |
| 既有充電站 | 台北 | 0.2333 | 0 | 29 | 3.861 | 3.030 | 26.692 |
| 來源模型 zero-shot（種子 1–5 平均） | 台北 | 2.5572 | 2.3239 | 301 | 1.199 | 0.110 | 0.498 |
| **來源模型 + beam search 微調**（種子 1–5 平均） | 台北 | **2.6345** | **2.4012** | 307 | **1.172** | 0.111 | **0.460** |

參考：隨機策略（同一環境，每步隨機選動作）的 improvement 台南平均 4.9757（種子 1–3）、台北平均 1.2563（種子 1–3）。

- 台南、台北所有模型規劃的平均移動時間與平均等待時間都**遠低於 15 分鐘的服務目標**。
- 台北的分數以台北自己的既有充電站正規化，與台南的分數不直接比較。

---

## 6. 各 AI 模型之訓練細節、測試細節與最終效能數據

### 6.1 訓練細節

| 項目 | 原始 PCRL 模型 | 來源模型 |
|---|---|---|
| 程式 | `reinforcement.py --city Tainan` | `python -m transfer.train --city Tainan --out runs/source_Tainan --steps 20000 --seed 1 --threads 8` |
| 環境 | `env_plus.StationPlacement` | `transfer.env.TransferStationPlacement`（動作、獎勵與原始環境完全相同，只替換觀測） |
| 演算法 | DQN（stable-baselines3 1.5.0），超參數見 3.2 | 同左 |
| 訓練資料 | 台南環境，20,000 步 | 台南環境，20,000 步（31 個完整 episode，平均 633 步） |
| 每個 episode | 從既有 17 站開始，每步一個佈建動作，預算用完或步數達 N/2 時結束；獎勵 = 分數創新高時的增量 | 同左；每個 episode 開始時清空快取並還原路口屬性 |
| 決定性 | seed 1；torch / numpy / random 種子＋`torch.use_deterministic_algorithms(True)` | 同左 |
| 訓練紀錄 | `tmp_Tainan/train.log` → `Results/Trackio/training/original_Tainan.csv` | `runs/source_Tainan/monitor.csv`、`Results/Transfer/source_selection.csv` → `Results/Trackio/training/source_Tainan.csv` |
| 訓練曲線圖 | `Images/Result_Plots/Training/original_Tainan_training.png` | `Images/Result_Plots/Training/source_Tainan_training.png` |
| 交付權重 | 11,200 步（訓練平均獎勵最高時存檔） | 17,000 步（貪婪評估 improvement 最高的 checkpoint） |
| 訓練環境 | Python 3.8、stable-baselines3 1.5.0、torch 1.12（CPU）（`docker/legacy-py38/`） | 同左 |

兩個模型都在舊環境（Python 3.8）訓練；目前環境（Python 3.12、`pcrl:2.0`）直接載入同一份權重，產生的規劃與舊環境逐站相同
（6.2）。

### 6.2 測試細節

| 項目 | 方法 |
|---|---|
| 推論方式 | 貪婪 episode：每一步取 Q 值最大的動作（`deterministic=True`），不探索（`transfer/evaluate.py` 以 torch 單執行緒執行） |
| 台南（訓練城市） | 兩個模型各跑一個貪婪 episode，產生規劃後以 `test_solution.py` / `analysis/plan_metrics.py` 計算服務指標 |
| 台北（未見過的城市） | 來源模型不經任何台北訓練直接跑貪婪 episode（zero-shot），episode 種子 1–5；隨機策略種子 1–3 作為參考 |
| 微調 | beam search B = 4、k = 2，同一組 episode 種子 1–5，與 zero-shot 逐一配對比較 |
| 正確性驗證 | B = k = 1 的搜尋在 5 個種子上與 zero-shot 結果逐位元相同；同一設定重跑結果逐位元相同 |
| 環境等價驗證 | 目前環境載入舊模型：台北 zero-shot（種子 1，403 站）、大安區 zero-shot 與 beam search、原始模型（中西區）的規劃都與舊環境逐站相同，分數差異 < 1e-12 |
| 映像驗證 | `pcrl:2.0-model` 內以內建權重重跑：原始 PCRL 模型的台南規劃（755 站，score 8.142435537145035）與 `Results/Tainan/` 逐站相同、`nodes_RL.txt` 逐位元相同；來源模型的台南規劃 improvement 8.532322275639508（舊環境 8.532322275639117）、405 站、581 步，逐站相同；大安區 zero-shot 1.8588917694329221、74 站 |
| 程式測試 | 140 個單元測試（新舊兩個環境皆通過）；SonarQube Quality Gate 通過，0 個 issue，評等 A / A / A |

### 6.3 最終效能數據

**台南（訓練城市）**

| 規劃 | improvement | 充電站數 | 平均移動時間（分） | 平均等待時間（分） | 最大等待時間（分） | 預算使用率 |
|---|---|---|---|---|---|---|
| 既有充電站 | 0 | 17 | 10.806 | 7.457 | 56.842 | 0% |
| 原始 PCRL 模型 | 7.9091 | 755（新建 738） | 1.771（−83.6%） | 0.365（−95.1%） | 8.456 | 99.994% |
| **來源模型** | **8.5323** | 405 | 2.178（−79.8%） | **0.098（−98.7%）** | **0.457** | — |

**台北（未用於訓練）：zero-shot 與 beam search 微調**

| episode 種子 | zero-shot improvement | 微調後 improvement | 提升 | 站數（前 → 後） | 平均移動時間（前 → 後，分） | 平均等待時間（前 → 後，分） |
|---|---|---|---|---|---|---|
| 1 | 2.4756 | **2.6428** | **+6.8%** | 403 → 373 | 1.018 → 1.043 | 0.090 → 0.101 |
| 2 | 2.0695 | 2.0954 | +1.3% | 231 → 231 | 1.360 → 1.338 | 0.123 → 0.125 |
| 3 | 2.3019 | 2.3021 | +0.01% | 292 → 293 | 1.166 → 1.164 | 0.113 → 0.114 |
| 4 | 2.3604 | **2.5539** | **+8.2%** | 285 → 342 | 1.219 → 1.082 | 0.113 → 0.104 |
| 5 | 2.4119 | 2.4119 | 0 | 294 → 294 | 1.230 → 1.230 | 0.111 → 0.111 |
| **平均** | **2.3239** | **2.4012** | **+3.3%** | 301 → 307 | 1.199 → 1.172 | 0.110 → 0.111 |

- zero-shot 比隨機策略（1.2563）高 85.0%，5 個種子全部高於隨機策略的最佳結果（1.8570）：台南學到的佈建策略可直接
  用在台北。
- 微調後平均移動時間再降 2.3%、最大等待時間降 7.7%（0.498 → 0.460 分）；與台北既有充電站相比，平均移動時間降低 69.6%、
  平均等待時間降低 96.3%。

**台北大安區示範（來源模型，B = 4、k = 2）**：zero-shot 平均 1.8995 → 微調後 2.0623（+8.6%），5 個種子全部提升。

---

## 7. 模型進行訓練與推論所需之參考時間

主機：Intel Core i7-14700F（28 執行緒）、31 GB 記憶體，只用 CPU。

**訓練**

| 模型 | 設定 | 時間 | 環境 |
|---|---|---|---|
| 原始 PCRL 模型 | 台南 20,000 步 | 約 **37.5 小時**（約 6.9 秒/步） | 舊環境；之後覆蓋計算已向量化，同樣的 episode 約快 2.2 倍 |
| 來源模型 | 台南 20,000 步、8 執行緒 | 約 **13.8 小時**（約 2.5 秒/步） | 舊環境 |
| 來源模型 checkpoint 選擇 | 20 個 checkpoint 各跑一個台南貪婪 episode（平行） | 每個 1,104–13,705 秒（依 episode 長度） | 舊環境 |
| 短訓練實測（目前環境，`pcrl:2.0-model`） | 台南 300 步 | 原始模型 381.7 秒、來源模型 300.0 秒（含載入路網；episode 前段站數少、每步較快，不能直接外推為完整訓練時間） | 目前環境 |

**推論**（一個完整的貪婪 episode = 產生一份完整規劃）

| 模型 / 情境 | 時間 | 環境 |
|---|---|---|
| 原始 PCRL 模型，台南（739 步） | **3,031 秒（約 51 分）**；舊環境紀錄 3,957 秒 | 目前環境（`pcrl:2.0-model`） |
| 來源模型，台南（581 步） | **945 秒（約 16 分）**；舊環境紀錄 1,020 秒 | 目前環境（`pcrl:2.0-model`） |
| 來源模型 zero-shot，台北（種子 1–5，516–688 步） | 363–833 秒；種子 1 在目前環境 631 秒 | 舊環境 / 目前環境 |
| 來源模型 zero-shot，大安區 | 約 4 秒 | |
| beam search 微調 B = 4、k = 2，台北 | 711–1,671 秒（11.8–27.8 分） | 舊環境 |
| beam search 微調 B = 4、k = 2，大安區 | 8–47 秒 | |

模型本身的前向計算很小（來源模型 15 萬個參數），時間主要花在每一步重新計算整份規劃的分數（路口指派、到站車流、
等候時間、覆蓋度）；目前環境比舊環境快約 24%（台北 zero-shot 種子 1：832.7 → 631.0 秒）。

---

## 8. 各 AI 模型於訓練與推論階段所需之最低軟硬體需求

記憶體為 `pcrl:2.0-model` 內的實測峰值（RSS，2026-11-20）；「建議最低」在實測值上保留餘裕，CPU、磁碟為建議配置：

| 模型 / 階段 | 實測記憶體 | 建議最低記憶體 | CPU | GPU | 磁碟 |
|---|---|---|---|---|---|
| 原始 PCRL 模型 訓練 | 前 300 步 1.86 GB；replay buffer 填滿再加 4.8 GB（10,000 筆 × 127,865 維 float16 × 觀測與下一觀測，依程式計算）；完整 episode 3.5 GB（推論實測） | 16 GB（原始環境的路徑快取在長時間訓練中持續增加，見 `README_for_doc.md` 11.2；本專案在 31 GB 主機完成訓練） | x86-64，4 核心以上 | 不需要 | 5 GB |
| 原始 PCRL 模型 推論 | 3.5 GB（台南完整 episode） | 6 GB | x86-64，2 核心以上 | 不需要 | 5 GB |
| 來源模型 訓練 | 前 300 步 1.36 GB；完整 episode 2.1 GB（推論實測）＋ replay buffer 填滿 0.12 GB；每個 episode 開始時清空快取 | 4 GB | x86-64，8 執行緒以上建議 | 不需要 | 5 GB |
| 來源模型 推論（zero-shot） | 2.1 GB（台南完整 episode） | 4 GB | x86-64，1 核心即可 | 不需要 | 5 GB |
| 微調（beam search B = 4、k = 2，台北） | 4.1–6.8 GB（協調行程＋所有候選行程，PSS） | 8 GB（`scripts/run_beam_search.sh` 預設每個 run 上限 12 GB） | x86-64，4 核心以上 | 不需要 | 5 GB |
| 微調（大安區示範） | 約 0.5 GB | 2 GB | x86-64 | 不需要 | 5 GB |

**軟體**

| 項目 | 需求 |
|---|---|
| 作業系統 | Linux x86-64（實測 Ubuntu 22.04.5，kernel 6.8）；beam search 使用 Linux 的 fork、`/proc` 與 Unix socket |
| 方式一：Docker | Docker 23 以上（預設啟用 BuildKit，`COPY --chmod` 需要；實測 29.9.2）；映像 `pcrl:2.0-model` 1.83 GB |
| 方式二：conda / pip | Python 3.12、`docker/requirements-pcrl.txt`（torch 2.15.0 CPU、stable-baselines3 2.9.2、gymnasium 1.4.0、osmnx 2.1.1、numpy 2.5.4、pandas 3.1.1 等，49 個套件 0 個已知弱點） |
| 選用 | systemd（`scripts/run_beam_search.sh` 以 `systemd-run --user --scope` 限制每個 run 的記憶體）；OCM API key（只在下載新城市的充電站時需要） |
| GPU | 不需要；網路很小，瓶頸在環境模擬，模型一律在 CPU 上載入 |

磁碟：映像 1.83 GB，加上輸出（每份台北規劃的 `plan_RL.pkl` + `nodes_RL.txt` 約 3.4 MB）；建議保留 5 GB 可用空間。

---

## 9. AI 模型使用手冊

### 9.1 執行環境

**方式一：Docker 映像（環境＋模型權重）**

```bash
# 建置（在 repo 根目錄；模型權重 *.zip 不在 git 中，需在含權重的工作目錄建置）
docker build -f docker/Dockerfile -t pcrl:2.0 .
docker build -f docker/Dockerfile.model -t pcrl:2.0-model .
docker run --rm pcrl:2.0-model          # 印出套件版本與兩個模型檔的 sha256

# 由檔案載入映像（交付檔）
docker load -i pcrl_2.0-model.tar.gz
```

映像內的工作目錄為 `/workspace/PCRL`，含程式碼、`Graph/` 輸入資料、模型權重與本文件；預設以非 root 使用者 `pcrl` 執行，
模型檔為唯讀。輸出可寫到容器內新資料夾，或以 `-v` 掛載主機資料夾。

**方式二：conda**

```bash
conda create -n PCRL-py312 python=3.12 && conda activate PCRL-py312
pip install -r docker/requirements-pcrl.txt
```

### 9.2 使用方式

以下指令在 repo 根目錄（或容器的 `/workspace/PCRL`）執行。

```bash
# 推論 1：原始 PCRL 模型產生台南規劃 → plan_RL.pkl、nodes_RL.txt
python model_eval.py --city Tainan --output Results/Tainan
python test_solution.py --city Tainan --plan-dir Results/Tainan          # 移動時間、等待時間、預算

# 推論 2：來源模型在任一城市產生規劃（zero-shot）
python -m transfer.evaluate --city Taipei --policy model --model runs/source_Tainan/best_source.zip \
    --label zeroshot --seed 0 --eval-seed 1 --csv Results/Transfer/zeroshot.csv --save-plan Results/Transfer/plans/zeroshot_e1

# 微調：模型引導 beam search
python -m transfer.beam_search --city Taipei --beam 4 --topk 2 --eval-seed 1 --label B4k2 \
    --csv Results/Transfer/beam.csv --save-plan Results/Transfer/plans/B4k2_e1

# 任一規劃的服務指標、地圖
python -m analysis.plan_metrics --city Taipei --plan Results/Transfer/plans/B4k2_e1 --label B4k2_e1 --csv metrics.csv
python Visualisierung/visualise.py --city Taipei --plan Results/Transfer/plans/B4k2_e1/plan_RL.pkl --output map.png

# 訓練
python reinforcement.py --city Tainan                                       # 原始 PCRL 模型 → tmp_Tainan/
python -m transfer.train --city Tainan --out runs/source_Tainan --steps 20000 --seed 1 --threads 8   # 來源模型
scripts/run_generalization.sh all       # 來源模型完整流程：訓練、checkpoint 選擇、台南指標、台北 zero-shot、統計
scripts/run_beam_search.sh all          # 微調完整流程：驗證、搜尋、分析

# 訓練與微調紀錄 → Results/Trackio/（第 10 節）
python -m analysis.trackio_logs
```

Docker 範例（不掛載，直接使用映像內的權重；輸出寫到主機的 `out/`）：

```bash
mkdir -p out && chmod 777 out
docker run --rm -v "$PWD/out":/out pcrl:2.0-model python -m transfer.evaluate --city Taipei_Daan --policy model \
    --model runs/source_Tainan/best_source.zip --label zeroshot --eval-seed 1 --csv /out/zeroshot.csv --save-plan /out/plan_e1
docker run --rm -v "$PWD/out":/out pcrl:2.0-model python model_eval.py --city Tainan --output /out/original_Tainan
```

**Python API**

```python
import os
import random
import numpy as np
import gym_compat as gc
from transfer import env as et

random.seed(1); np.random.seed(1)                              # episode 種子（同 --eval-seed 1）
env = et.make_env("Taipei", os.getcwd())                       # Graph/Taipei/ 的路網、路口與既有充電站
model = gc.load_dqn("runs/source_Tainan/best_source.zip", env)  # CPU 上載入
obs, done = gc.reset(env), False
existing_score = env.best_score                                # 既有規劃的分數（0.2333）
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = gc.step(env, int(action))
print("improvement", env.best_score - existing_score, "stations", len(env.best_plan))
# env.best_plan：規劃（格式同 plan_RL.pkl）；大安區（Taipei_Daan）約 4 秒，improvement 1.8588917694329221、74 站
```

主要參數：`--city`（`Graph/<city>/` 資料夾名稱）、`--model`、`--eval-seed`（episode 種子）、`--save-plan`（規劃輸出資料夾）、
`--beam` / `--topk`（微調）。完整參數見 `README_for_doc.md` 第 9 節，或各程式的 `--help`。

### 9.3 輸入格式

每個城市一個資料夾 `Graph/<city>/`：

| 檔案 | 格式 | 內容 |
|---|---|---|
| `<city>.graphml` | OSM 路網（osmnx GraphML，有向多重圖） | 節點 `x`（經度）、`y`（緯度）、`street_count`；道路 `length`（m）、`maxspeed`（km/h，可缺） |
| `node_list_<city>.txt` | Python 字面值：`[(osm_id, {屬性}), ...]` | `x`、`y`、`street_count`、`demand`（0.05–1）、`estate price`（0.1–1）、`private CS`（私人充電樁數）、`osmid` |
| `existingplan_<city>.pkl` | pickle：`[[(node_id, 屬性), numpy.array([n_7kW, n_22kW, n_50kW]), {}], ...]` | 既有充電站：所在節點與三種充電樁數量 |
| 模型 `.zip` | stable-baselines3 DQN 存檔 | 網路權重與設定；原始 PCRL 模型的輸入維度必須等於 8 × 路口數 + 1（台南），來源模型可用於任何城市 |

新城市的輸入檔可由 `Preprocessing/` 的四個腳本從 OSM 與 OCM 產生（`README.md` Preprocessing 一節）：路網 →
節點清單 → 下載 OCM 充電站 → 對應成既有規劃。至少需要一座既有充電站（分數以既有規劃正規化）。

### 9.4 輸出格式

| 檔案 | 格式 | 內容 |
|---|---|---|
| `plan_RL.pkl` | pickle：充電站清單 `[[(node_id, 屬性), numpy.array([n_7kW, n_22kW, n_50kW]), {站屬性}], ...]` | 規劃中的每座充電站（含既有站）；站屬性：`fee`（安裝費）、`capability`（kW）、`radius`（服務半徑 m）、`D_s`（到站車流）、`service rate`、`W_s`（等候時間，小時） |
| `nodes_RL.txt` | Python 字面值：`[(osm_id, {屬性}), ...]` | 每個路口被指派的充電站（`charging station`）、`distance`（m）、`travel time`（小時）、`covered`（覆蓋度） |
| 評估 CSV（`transfer/evaluate.py`） | CSV，每個 episode 一列 | `city, label, seed, policy, model, train_steps, existing_score, score, improvement, relative_pct, n_stations, episode_steps, seconds, eval_seed` |
| 微調 CSV（`transfer/beam_search.py`） | CSV，每個 run 一列 | `city, label, model, beam, topk, value_weight, eval_seed, existing_score, score, improvement, n_stations, episode_steps, expansions, seconds, action_counts` |
| 服務指標 CSV（`analysis/plan_metrics.py`） | CSV，每份規劃一列 | `city, label, plan, n_stations`、平均移動時間 `travel_avg_min`、平均等待時間 `waiting_avg_min`、最大等待時間 `waiting_max_min` 等 |
| 地圖（`Visualisierung/visualise.py`） | PNG | 紅點 = 充電站、灰點 = 一般路口 |
| 訓練／微調紀錄 | trackio CSV（第 10 節） | |

時間單位：CSV 中的指標為分鐘；`plan_RL.pkl`、`nodes_RL.txt` 內的時間為小時。

---

## 10. 訓練與微調紀錄（trackio 格式）

`python -m analysis.trackio_logs` 把訓練與微調的原始紀錄轉成 trackio（Hugging Face）`trackio.import_csv()` 可直接匯入的 CSV：
第一欄 `step`，有時間紀錄的檔案含 `timestamp`（ISO 8601，UTC），其餘欄位都是數值指標，該步沒有紀錄的欄位留空。
欄名都在 10 個英數字或 `/` 以內，匯入後名稱不變。已以 trackio 0.45.1 實際匯入全部 12 個檔案，逐列讀回確認筆數與數值一致。

| 檔案 | 列數 | 來源 | 欄位 |
|---|---|---|---|
| `Results/Trackio/training/original_Tainan.csv` | 49 | `tmp_Tainan/train.log`（每 400 步）、評估 episode | `reward/avg` 最近 3 個訓練 episode 平均獎勵、`reward/max` 至今最高平均、`saved` 該步是否存檔（1/0）、`epsilon` 探索率、`eval/imp` / `eval/stns` 交付模型（11,200 步）貪婪 episode 的 improvement 與站數 |
| `Results/Trackio/training/source_Tainan.csv` | 51 | `runs/source_Tainan/monitor.csv`（每個 episode）、`Results/Transfer/source_selection.csv`（每個 checkpoint） | `timestamp`、`hours` 訓練經過時間、`reward` episode 獎勵（含探索的 improvement）、`length` episode 步數、`epsilon`、`eval/imp` / `eval/stns` 各 checkpoint 貪婪 episode 的 improvement 與站數 |
| `Results/Trackio/finetuning/Taipei_B4k2_e<種子>.csv` | 22–29 | `logs/beam/B4k2_e<種子>.log`（每 25 步） | `score` 目前找到的最佳規劃分數、`improve` 其 improvement、`live/best` / `done/best` 進行中／已完成候選的最佳分數、`key` 搜尋排序分數、`beam` 候選數、`stations` / `budget` 領先候選的站數與剩餘預算、`seconds` 經過秒數 |
| `Results/Trackio/finetuning/Taipei_zeroshot_e<種子>.csv` | 22–29 | `logs/beam/B1k1_e<種子>.log` | 同上；B = k = 1 即 zero-shot 貪婪 episode，作為微調前的對照 |

`epsilon` 依訓練設定的線性排程計算（前 2,000 步由 1.0 降到 0.05）。新的 beam search 紀錄若含平均移動／等待時間，會另外輸出
`travel`、`waiting` 兩欄（分鐘）。

匯入 trackio（每個 CSV 匯入成一個 run；`import_csv` 只能匯入到尚未存在的 project）：

```bash
pip install trackio
python -c "import trackio; trackio.import_csv('Results/Trackio/training/source_Tainan.csv', project='pcrl-source-training')"
trackio show --project pcrl-source-training
```

---

## 11. 限制與注意事項

- **需求與地價為合成資料**：由路口連接數推估，未使用實際電動車登記、車流或地價資料；規劃結果反映這個假設。
- **原始 PCRL 模型只能用於台南**：輸入維度等於台南的路口數；其他城市請使用來源模型。
- **評估範圍**：目前在台南、台北（含大安區）評估；其他城市可依相同流程（前處理 → zero-shot → beam search）套用。
- **輸入資料提醒**：台南的 OCM 資料中有一座既有充電站有 18 支樁（超過每站 15 支的上限），評估時會出現
  `More chargers at the station than admitted` 提示，屬於輸入資料，不影響模型。
- **重新訓練**：兩個模型在舊環境（Python 3.8）訓練。目前環境可正常訓練，但 torch / stable-baselines3 版本不同，亂數序列
  不同，重新訓練出的權重不會與交付權重逐位元相同；要逐位元重現訓練請使用 `docker/legacy-py38/`。推論與微調結果兩個環境相同。
- **OSM 與 OCM 持續更新**：重新下載會得到不同的路網與充電站；要重現本文件結果，請使用 repo 中的 `Graph/<city>/`。

---

## 12. 檔案與版本

| 檔案 | 大小 | sha256 |
|---|---|---|
| `tmp_Tainan/best_model_Tainan_11200.zip` | 132,735,889 bytes | `7f48decb399b7e505ec374cb8c96e41ceb144d84504527fdc27fcceb8816d78e` |
| `runs/source_Tainan/best_source.zip` | 2,443,234 bytes | `3ef9936eb721f32ccd600cfcce01fea322a8b7841efae270065b032a404de913` |

| 映像 | 內容 | 大小 |
|---|---|---|
| `pcrl:2.0-model` | 環境＋程式碼＋`Graph/` 資料＋兩個模型權重＋本文件 | 1.83 GB |
| `pcrl:2.0` | 環境＋程式碼＋`Graph/` 資料 | 1.69 GB |
| `pcrl:1.0` | 舊環境（Python 3.8，訓練時使用） | 2.95 GB |

引用：原始方法 Leonie von Wahl, Nicolas Tempelmeier, Ashutosh Sao, Elena Demidova, "Reinforcement Learning-based Placement of
Charging Stations in Urban Road Networks"（`CITATION.cff`）。

#!/usr/bin/env bash
# Source model (Tainan) and generalisation to Taipei, see README_for_doc.md sections 7.4, 10.1 and 10.2
#
#   source     train the source model on Tainan (python -m transfer.train)
#   select     score every source checkpoint on Tainan, copy the best to runs/source_Tainan/best_source.zip
#   service    Tainan plan of the source model, service metrics (travel / waiting time) of the existing stations,
#              the original PCRL model (Results/Tainan) and the source model -> Results/Tainan/metrics.csv, figures
#   baselines  Taipei: zero-shot (source model, episode seeds 1-5), random policy (seeds 1-3)
#   analyze    python -m analysis.generalization: statistics + figure
#   all        every phase above, in order
#
# Usage: scripts/run_generalization.sh <source|select|service|baselines|analyze|all>
set -euo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"  # absolute, so the script can call itself after the cd
cd "$(dirname "$SELF")/.."  # repository root

# Python of the conda environment PCRL-py312 (README_for_doc.md section 3.1); the legacy environment that produced
# the reported results: PY=$HOME/anaconda3/envs/PCRL/bin/python
export PY=${PY:-$HOME/anaconda3/envs/PCRL-py312/bin/python}
SOURCE_STEPS=${SOURCE_STEPS:-20000}
EVAL_SEEDS=${EVAL_SEEDS:-"1 2 3 4 5"}
RANDOM_SEEDS=${RANDOM_SEEDS:-"1 2 3"}
JOBS=${JOBS:-8}  # parallel evaluations, each holds the road graph in memory (~1-1.5 GB)
CSV=Results/Transfer/generalization.csv
SOURCE=runs/source_Tainan/best_source.zip
mkdir -p runs logs/training logs/generalization Results/Transfer

# run each line "<log name> <args...>" as one transfer/evaluate.py call, JOBS at a time; fails if any call fails
parallel_eval() {
  xargs -P "$JOBS" -L 1 bash -c 'log=$0; "$PY" -m transfer.evaluate "$@" > "logs/generalization/$log.log" 2>&1 || { echo "failed: $log"; exit 255; }'
}

case "${1:-}" in
  source)
    $PY -m transfer.train --city Tainan --out runs/source_Tainan --steps "$SOURCE_STEPS" --seed 1 --threads 8 \
      > logs/training/source_Tainan.log 2>&1
    ;;
  select)
    rm -f Results/Transfer/source_selection.csv
    for model in runs/source_Tainan/checkpoints/*.zip; do
      echo "select_$(basename "$model" .zip) --city Tainan --policy model --model $model --label S_source" \
        "--csv Results/Transfer/source_selection.csv"
    done | parallel_eval
    # checkpoints up to learning_starts are still the random initial network
    best=$($PY -c "import json, pandas as pd; start=json.load(open('runs/source_Tainan/config.json'))['learning_starts']; \
d=pd.read_csv('Results/Transfer/source_selection.csv'); d=d[d.train_steps > start]; print(d.loc[d.improvement.idxmax(), 'model'])")
    cp "$best" "$SOURCE"
    echo "Selected source model: $best"
    ;;
  service)
    $PY -m transfer.evaluate --city Tainan --policy model --model $SOURCE --label source_Tainan --seed 1 \
      --csv Results/Tainan/source_model.csv --save-plan Results/Tainan/source_model > logs/training/source_model_Tainan.log 2>&1
    rm -f Results/Tainan/metrics.csv
    $PY -m analysis.plan_metrics --city Tainan --existing --csv Results/Tainan/metrics.csv
    $PY -m analysis.plan_metrics --city Tainan --plan Results/Tainan --label original_model --csv Results/Tainan/metrics.csv
    $PY -m analysis.plan_metrics --city Tainan --plan Results/Tainan/source_model --label source_model \
      --csv Results/Tainan/metrics.csv
    $PY -m analysis.training_plots
    ;;
  baselines)
    {
      for e in $EVAL_SEEDS; do
        echo "zeroshot_e$e --city Taipei --policy model --model $SOURCE --label zeroshot --seed 0 --eval-seed $e" \
          "--csv $CSV --save-plan Results/Transfer/plans/zeroshot_e$e"
      done
      for seed in $RANDOM_SEEDS; do
        echo "random_s$seed --city Taipei --policy random --label random --seed $seed --csv $CSV"
      done
    } | parallel_eval
    ;;
  analyze)
    $PY -m analysis.generalization
    ;;
  all)
    for phase in source select service baselines analyze; do
      echo "$(date '+%F %T') phase $phase"
      "$SELF" "$phase"
    done
    echo "$(date '+%F %T') done"
    ;;
  *)
    sed -n '2,12p' "$SELF"
    exit 1
    ;;
esac

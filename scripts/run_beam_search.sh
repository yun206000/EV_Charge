#!/usr/bin/env bash
# Beam search with the Tainan source model on Taipei (transfer/beam_search.py), see README_for_doc.md section 10.3
#
#   validate   B = k = 1 on episode seeds 1-5: must reproduce the zero-shot episodes of transfer/evaluate.py exactly
#   search     B = $BEAM, k = $TOPK on episode seeds 1-5, plans saved to Results/Transfer/plans/B<B>k<k>_e<seed>
#   analyze    python -m analysis.beam: comparison with zero-shot, service metrics, figures;
#              python -m analysis.plan_maps: maps of the Taipei stations before / after fine-tuning
#   all        every phase above, in order
#
# Memory: every run is started in its own cgroup (systemd-run --user --scope) with a hard limit of $MEM_MAX; if the
# limit is reached only that run is killed, never the whole machine. The search also stops itself before that
# (--max-mem-gb, --min-available-gb). At most $JOBS runs at a time.
#
# Usage: scripts/run_beam_search.sh <validate|search|analyze|all>
set -euo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"  # absolute, so the script can call itself after the cd
cd "$(dirname "$SELF")/.."  # repository root

export PY=${PY:-$HOME/anaconda3/envs/PCRL/bin/python}
BEAM=${BEAM:-4}
TOPK=${TOPK:-2}
EVAL_SEEDS=${EVAL_SEEDS:-"1 2 3 4 5"}
JOBS=${JOBS:-2}
MEM_MAX=${MEM_MAX:-12G}
GUARD_GB=${GUARD_GB:-11}
CSV=Results/Transfer/beam.csv
mkdir -p logs/beam Results/Transfer/plans

# run each line "<name> <beam> <topk> <seed>" as one transfer/beam_search.py call in its own memory-limited cgroup
run_beam() {
  xargs -P "$JOBS" -L 1 bash -c '
    name=$0; beam=$1; topk=$2; seed=$3
    systemd-run --user --scope -q -p MemoryMax="$MEM_MAX" -p MemorySwapMax=0 \
      "$PY" -m transfer.beam_search --city Taipei --beam "$beam" --topk "$topk" --eval-seed "$seed" --label "$name" \
      --max-mem-gb "$GUARD_GB" --csv "$CSV" --save-plan "Results/Transfer/plans/${name}_e$seed" \
      > "logs/beam/${name}_e$seed.log" 2>&1 || { echo "failed: ${name}_e$seed"; exit 255; }'
}
export -f run_beam
export CSV MEM_MAX GUARD_GB

case "${1:-}" in
  validate)
    for e in $EVAL_SEEDS; do echo "B1k1 1 1 $e"; done | run_beam
    ;;
  search)
    for e in $EVAL_SEEDS; do echo "B${BEAM}k${TOPK} $BEAM $TOPK $e"; done | run_beam
    ;;
  analyze)
    "$PY" -m analysis.beam
    "$PY" -m analysis.plan_maps --label "B${BEAM}k${TOPK}"
    ;;
  all)
    for phase in validate search analyze; do
      echo "$(date '+%F %T') phase $phase"
      "$SELF" "$phase"
    done
    echo "$(date '+%F %T') done"
    ;;
  *)
    sed -n '2,13p' "$SELF"
    exit 1
    ;;
esac

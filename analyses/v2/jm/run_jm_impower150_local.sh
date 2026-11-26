#!/bin/bash
# Joint model v2 on IMpower150 OS (train_jm_impower150.R), 100 splits, several splits in parallel (1 core each).
# Splits already finished (res_splitXXX.csv present) are skipped, so the script can be stopped and restarted.
# Usage: bash run_jm_impower150_local.sh <4tp|3tp|2tp|all> [n_jobs] [first] [last]
#   n_jobs default: performance cores (Mac) or all cores minus 1 (Linux)
# Inputs: results/jm_impower150/<tp>/IMP150_JM_long.csv + IMP150_JM_test_indices.csv (prepare_jm_impower150.py)
cd "$(dirname "$0")"
if [ "$(uname)" = "Darwin" ]; then
  DEF=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || sysctl -n hw.physicalcpu)
else
  DEF=$(( $(nproc) - 1 )); [ "$DEF" -lt 1 ] && DEF=1
fi
TPS=$1; [ "$TPS" = "all" ] && TPS="4tp 3tp 2tp"
N_JOBS=${2:-$DEF}; FIRST=${3:-0}; LAST=${4:-99}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
run_split() {   # $1 = tp, $2 = split
  local s; s=$(printf "%03d" "$2"); local out=results/jm_impower150/$1/jm_v2
  [ -f "$out/res_split$s.csv" ] && return 0
  if Rscript "$(dirname "$0")/train_jm_impower150.R" "$1" "$2" "$2" 10000 3000 3 1 > "$out/logs/split$s.log" 2>&1; then
    echo "$1 split $2 done"
  else
    echo "$1 split $2 FAILED (see $out/logs/split$s.log)"
  fi
}
for tp in $TPS; do
  mkdir -p results/jm_impower150/$tp/jm_v2/logs
  echo "$tp: splits $FIRST-$LAST, $N_JOBS in parallel"
  for i in $(seq "$FIRST" "$LAST"); do      # simple job pool in plain bash (macOS-safe)
    while [ "$(jobs -rp | wc -l)" -ge "$N_JOBS" ]; do sleep 5; done
    run_split "$tp" "$i" &
  done
  wait
  python jm/jm_metrics_impower150.py "$tp"
done

#!/bin/bash
# Run the 100 splits of the joint model v2 locally (Mac or Linux), several splits in parallel.
# Each split runs its 3 MCMC chains one after the other on 1 core; N_JOBS splits run at the same time.
# Splits already finished (res_splitXXX.csv present) are skipped, so the script can be stopped and restarted.
# Usage: bash run_jm_v2_local.sh [n_jobs] [first] [last]
# Faster variant (noise SD 0.2, 6000/2000 iterations; ~30 % faster, -0.017 C on split 0), kept in a separate folder:
#   JM_FAST=1 bash run_jm_v2_local.sh
#   default n_jobs = number of performance cores (Mac) or all cores minus 1 (Linux)
# Needs R with nlme, survival, dplyr, GLMMadaptive, JMbayes2 (install.packages("JMbayes2") works on a Mac).
set -e
cd "$(dirname "$0")"
if [ "$(uname)" = "Darwin" ]; then
  DEF=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || sysctl -n hw.physicalcpu)
else
  DEF=$(( $(nproc) - 1 )); [ "$DEF" -lt 1 ] && DEF=1
fi
N_JOBS=${1:-$DEF}; FIRST=${2:-0}; LAST=${3:-99}
if [ "${JM_FAST:-0}" = "1" ]; then
  export JM_SIGMA=0.2; N_ITER=6000; N_BURN=2000
  OUT=results/benchmark_response_longitudinal/jm_v2_fast; METRICS=JointModel_v2_fast_metrics_per_split.csv
else
  N_ITER=10000; N_BURN=3000
  OUT=results/benchmark_response_longitudinal/jm_v2; METRICS=JointModel_v2_metrics_per_split.csv
fi
export JM_OUT=$OUT N_ITER N_BURN
mkdir -p "$OUT/logs"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
echo "splits $FIRST-$LAST, $N_JOBS in parallel, $N_ITER/$N_BURN iterations, output $OUT"
run_split() {
  s=$(printf "%03d" "$1")
  [ -f "$JM_OUT/res_split$s.csv" ] && return 0
  if Rscript "$(dirname "$0")/train_jm_pada1_v2.R" "$1" "$1" "$N_ITER" "$N_BURN" 3 1 > "$JM_OUT/logs/split$s.log" 2>&1; then
    echo "split $1 done"
  else
    echo "split $1 FAILED (see $JM_OUT/logs/split$s.log)"
  fi
}
# simple job pool in plain bash (no xargs: macOS xargs fails with large environments, e.g. conda)
for i in $(seq "$FIRST" "$LAST"); do
  while [ "$(jobs -rp | wc -l)" -ge "$N_JOBS" ]; do sleep 5; done
  run_split "$i" &
done
wait
python jm/jm_metrics_pada1.py "$OUT" results/benchmark_response_longitudinal/$METRICS
python competitors_pada1_summary.py

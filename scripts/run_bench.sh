#!/bin/bash
# Retry wrapper for benchmark.py on a shared card: a row killed by another job's
# memory growth resumes from its checkpoint on the next attempt.
#   scripts/run_bench.sh <gpu> <benchmark.py args...>
cd "$(dirname "$0")/.." || exit 1
PY=${PYTHON:-~/anaconda3/envs/cfd_transformer/bin/python}
GPU=$1; shift
for attempt in $(seq 1 80); do
  echo "## attempt $attempt on gpu $GPU $(date +%m-%d" "%H:%M)"
  $PY scripts/benchmark.py --gpu "$GPU" --slots 2 --min-free-gb 40 "$@" && { echo "## done"; exit 0; }
  sleep 600
done

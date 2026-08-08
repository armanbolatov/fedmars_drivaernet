#!/bin/bash
# Serial queue for FULL-setting runs on ONE gpu.
#
# A full run needs ~60 GB (batch 32 x 50k points of activations), so only one fits
# per card. Jobs run strictly one at a time, after any benchmark already on that gpu
# finishes, and a job killed by another user's memory spike is retried -- completed
# runs are skipped on retry, so only what actually died is redone.
#
#   RETRIES=100 ./scripts/run_full_queue.sh <gpu> "<args1>" "<args2>" ...
set -u
cd ~/fedmuon-cfd
PY=~/anaconda3/envs/cfd_transformer/bin/python
GPU=$1; shift
RETRIES=${RETRIES:-3}

while pgrep -f "benchmark.py .*--gpu $GPU" | grep -qv "^$$\$"; do
  echo "## $(date +%H:%M) waiting: gpu $GPU still busy with a benchmark instance"
  sleep 300
done

for job in "$@"; do
  for attempt in $(seq 1 "$RETRIES"); do
    echo "## $(date +%H:%M) gpu$GPU attempt $attempt: $job"
    # shellcheck disable=SC2086
    $PY scripts/benchmark.py $job --setting full --gpu "$GPU" && break
    echo "## $(date +%H:%M) FAILED (likely OOM from other users); backing off 10 min"
    sleep 600
  done
done
echo "## $(date +%H:%M) queue done on gpu $GPU"

#!/bin/bash
# Edge probes: bracket every edge-flagged optimum one rung outward (certifies
# interiority or triggers a further extension). Usage: run_probes.sh GPU_A GPU_B
set -u
cd ~/fedmuon-cfd
PY=~/anaconda3/envs/cfd_transformer/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A=$1; B=$2
echo "## PROBES START $(date): GPU$A scaffold+centralized_muon | GPU$B fedavg"
(
  CUDA_VISIBLE_DEVICES=$A $PY scripts/tune.py --stage scaffold --alpha 0.5  --probe-edges > results/optuna/probe_scaffold_dir0.5.out  2>&1
  CUDA_VISIBLE_DEVICES=$A $PY scripts/tune.py --stage scaffold --alpha 10   --probe-edges > results/optuna/probe_scaffold_dir10.out   2>&1
  CUDA_VISIBLE_DEVICES=$A $PY scripts/tune.py --stage scaffold --alpha 1000 --probe-edges > results/optuna/probe_scaffold_dir1000.out 2>&1
  CUDA_VISIBLE_DEVICES=$A $PY scripts/tune.py --stage centralized_muon      --probe-edges > results/optuna/probe_centralized_muon.out 2>&1
) &
(
  CUDA_VISIBLE_DEVICES=$B $PY scripts/tune.py --stage fedavg --alpha 0.5    --probe-edges > results/optuna/probe_fedavg_dir0.5.out    2>&1
  CUDA_VISIBLE_DEVICES=$B $PY scripts/tune.py --stage fedavg --alpha 1000   --probe-edges > results/optuna/probe_fedavg_dir1000.out   2>&1
) &
wait
echo "## PROBES DONE $(date)"

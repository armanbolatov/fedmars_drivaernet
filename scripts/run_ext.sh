#!/bin/bash
# Ladder-extension stages (edge rule): measure the values one step beyond a ladder
# whose winner touched an endpoint. Certified against the UNION ladder in tune.py.
#   GPU A: scaffold_ext (4) -> gamma_ext (2)
#   GPU B: fedavg_ext (4)
set -u
cd ~/fedmuon-cfd
PY=~/anaconda3/envs/cfd_transformer/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p results/optuna
GPUS=($(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
  | awk -F, '{gsub(/ /,"",$1);gsub(/ /,"",$2); if($2+0>=20000) print $2","$1}' \
  | sort -nr | cut -d, -f2))
[ ${#GPUS[@]} -lt 2 ] && { echo "need 2 free GPUs, have ${#GPUS[@]}"; exit 1; }
A=${GPUS[0]}; B=${GPUS[1]}
echo "## EXT START $(date): A=GPU$A (scaffold_ext->gamma_ext)  B=GPU$B (fedavg_ext)"
(
  CUDA_VISIBLE_DEVICES=$A $PY scripts/tune.py --stage scaffold_ext --n-trials 4 > results/optuna/scaffold_ext.out 2>&1
  CUDA_VISIBLE_DEVICES=$A $PY scripts/tune.py --stage gamma_ext    --n-trials 2 > results/optuna/gamma_ext.out    2>&1
) &
(
  CUDA_VISIBLE_DEVICES=$B $PY scripts/tune.py --stage fedavg_ext   --n-trials 4 > results/optuna/fedavg_ext.out   2>&1
) &
wait
echo "## EXT DONE $(date)"

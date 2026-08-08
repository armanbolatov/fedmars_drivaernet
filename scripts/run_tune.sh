#!/bin/bash
# Reference-split tuning pipeline (TPE + Hyperband; see docs/tuning.md):
#   GPU A: fedmuon (24) -> gamma grid (5) -> fedavg worker (12)
#   GPU B: scaffold (24) -> fedavg worker (12)
set -u
cd ~/fedmuon-cfd
PY=~/anaconda3/envs/cfd_transformer/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p results/optuna

# create the DB studies serially first (two workers racing to init a fresh sqlite
# schema makes one die on a UNIQUE constraint)
$PY - <<'EOF'
import optuna
from optuna.storages import RDBStorage
st = RDBStorage("sqlite:///results/optuna/tune.db",
                engine_kwargs={"connect_args": {"timeout": 60}})
for name in ("fedmuon_b32", "scaffold_b32", "fedavg_b32", "gamma_b32"):
    optuna.create_study(study_name=name, direction="minimize", storage=st, load_if_exists=True)
print("db ready")
EOF

GPUS=($(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
  | awk -F, '{gsub(/ /,"",$1);gsub(/ /,"",$2); if($2+0>=20000) print $2","$1}' \
  | sort -nr | cut -d, -f2))
[ ${#GPUS[@]} -lt 2 ] && { echo "need 2 free GPUs, have ${#GPUS[@]}"; exit 1; }
A=${GPUS[0]}; B=${GPUS[1]}
echo "## TUNE START $(date): A=GPU$A (fedmuon->gamma->fedavg)  B=GPU$B (scaffold->fedavg)"
(
  CUDA_VISIBLE_DEVICES=$A $PY scripts/tune.py --stage fedmuon  --n-trials 24 > results/optuna/fedmuon.out  2>&1
  CUDA_VISIBLE_DEVICES=$A $PY scripts/tune.py --stage gamma    --n-trials 5  > results/optuna/gamma.out    2>&1
  CUDA_VISIBLE_DEVICES=$A $PY scripts/tune.py --stage fedavg   --n-trials 12 > results/optuna/fedavg_a.out 2>&1
) &
(
  CUDA_VISIBLE_DEVICES=$B $PY scripts/tune.py --stage scaffold --n-trials 24 > results/optuna/scaffold.out 2>&1
  CUDA_VISIBLE_DEVICES=$B $PY scripts/tune.py --stage fedavg   --n-trials 12 > results/optuna/fedavg_b.out 2>&1
) &
wait
echo "## TUNE DONE $(date)"

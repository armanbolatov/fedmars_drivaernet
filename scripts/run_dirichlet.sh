#!/bin/bash
# Dirichlet heterogeneity tuning: fully independent tuning per distribution, same
# Bayesian protocol as the reference split. Per alpha chain: FedMuon (TPE 24, LRs x beta)
# -> gamma grid (5, on that alpha's FedMuon best) -> SCAFFOLD (TPE 24) -> FedAvg (TPE 24). Equal budget for every method.
# The 2D baseline spaces have 20 combos, so 24 trials + the dedup cache = exhaustive (+4 free cache hits).
# The reference optimum is enqueued as trial 0 (a seed, not a constraint).
# One chain per GPU; 3 GPUs runs all alphas in parallel, 2 GPUs chains a=1000 after a=0.5.
set -u
cd ~/fedmuon-cfd
PY=~/anaconda3/envs/cfd_transformer/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p results/optuna

# pre-create every study serially: two workers racing to init a fresh sqlite schema
# makes one die on the alembic UNIQUE constraint
$PY - <<'EOF'
import optuna
from optuna.storages import RDBStorage
st = RDBStorage("sqlite:///results/optuna/tune.db",
                engine_kwargs={"connect_args": {"timeout": 60}})
for stage in ("fedmuon", "gamma", "scaffold", "fedavg"):
    for a in ("0.5", "10", "1000"):
        optuna.create_study(study_name=f"{stage}_dir{a}", direction="minimize",
                            storage=st, load_if_exists=True)
print("db ready")
EOF

GPUS=($(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
  | awk -F, '{gsub(/ /,"",$1);gsub(/ /,"",$2); if($2+0>=20000) print $2","$1}' \
  | sort -nr | cut -d, -f2))
[ ${#GPUS[@]} -lt 2 ] && { echo "need >=2 free GPUs, have ${#GPUS[@]}"; exit 1; }

chain () {  # $1 = gpu, $2 = alpha
  for spec in "fedmuon 24" "gamma 5" "scaffold 24" "fedavg 24"; do
    st=${spec% *}; n=${spec#* }
    CUDA_VISIBLE_DEVICES=$1 $PY scripts/tune.py --stage $st --alpha $2 --n-trials $n \
      > results/optuna/${st}_dir$2.out 2>&1
  done
}

if [ ${#GPUS[@]} -ge 3 ]; then
  echo "## DIRICHLET START $(date): GPU${GPUS[0]} a=0.5 | GPU${GPUS[1]} a=10 | GPU${GPUS[2]} a=1000"
  chain ${GPUS[0]} 0.5 &
  chain ${GPUS[1]} 10 &
  chain ${GPUS[2]} 1000 &
else
  echo "## DIRICHLET START $(date): GPU${GPUS[0]} a=0.5->1000 | GPU${GPUS[1]} a=10"
  ( chain ${GPUS[0]} 0.5; chain ${GPUS[0]} 1000 ) &
  chain ${GPUS[1]} 10 &
fi
wait
echo "## DIRICHLET DONE $(date)"

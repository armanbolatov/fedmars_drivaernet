HYDRA_FULL_ERROR=1 nohup python3 scripts/jet_benchmark.py \
    --methods fedmars \
    --splits dirichlet_0.5 dirichlet_10 dirichlet_1000 \
    --local-epochs 1 2 4 8 \
    --seeds 42 123 456 \
    --muon-lrs 0.005 \
    --scalar-lrs 0.0001 \
    --muon-wds 0.01 \
    --fed-betas 0.9 \
    --setting proxy \
    --model omniarch \
    --run-folder tuned_fedmars \
    --gpu 2 \
    > jet_fedmars1.txt 2>&1 &

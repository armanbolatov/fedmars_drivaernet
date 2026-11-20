HYDRA_FULL_ERROR=1 nohup python3 scripts/jet_benchmark.py \
    --methods fedlion \
    --splits single_class \
    --local-epochs 2 4 \
    --seeds 42 123 \
    --lion-lrs 0.0001 0.0002 0.0003 0.0004 0.0005 \
    --lion-betas 0.8 0.9 \
    --lion-beta2s 0.95 \
    --lion-wds 0.005 0.01 0.05 \
    --setting proxy \
    --model omniarch \
    --run-folder fedlion_search \
    --gpu 3 \
    > jet_fedlion_search.txt 2>&1 &

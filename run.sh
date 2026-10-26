python scripts/benchmark.py \
  --methods fedavg scaffold fedlion fedmuon fedmars fedolion \
  --splits single_class dirichlet_0.5 dirichlet_10 dirichlet_1000 \
  --local-epochs 1 2 4 8 16 \
  --setting full30 --model omniarch --gpu 0

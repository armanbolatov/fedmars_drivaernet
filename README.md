# FedMuon / FedMARS-M on DrivAerNet++

Federated learning for CFD surrogate modelling: predict aerodynamic force coefficients
(drag **c_d**, lift **c_l**, side **c_s**) and surface fields (pressure, wall shear stress)
of DrivAerNet++ car geometries, with clients split by car body type (natural non-IID) or by
Dirichlet allocation.

Methods: **FedAvg**, **SCAFFOLD-Adam**, **FedMuon** (Muon + control variates),
**FedMARS-M** (FedMuon + MARS variance reduction), and a **centralized** upper bound.
FedMuon ≡ FedMARS with γ = 0 — one implementation, one flag.

## Layout

```
configs/            protocol (base.yaml), per-method and per-split configs
src/train.py        the trainer (all methods; one file)
src/eval_ext.py     evaluation: c_d/c_l/c_s + relative-L2 field errors
src/forces.py       lift & side coefficients (directional force integration)
scripts/tune.py     hyperparameter tuning (Optuna TPE + Hyperband pruning)
scripts/benchmark.py  benchmark matrix: methods x splits x seeds x local-epochs
scripts/report.py   aggregate results/runs -> mean±std tables
results/            small CSVs, optuna exports, figures (no raw data)
docs/algorithms.tex algorithm statements
```

## Setup

1. Conda env `cfd_transformer` with `requirements.txt` installed.
2. The [cfd_transformer] repo (model, datamodule, compiled force integrator) at
   `~/cfd_transformer`, or set `CFD_ROOT`.
3. DrivAerNet++ `.npy` dataset at `$CFD_ROOT/dataset/DrivAerNetPP` with
   `split/{train,test}_split.json`.

Nothing heavy lives in this repo: no dataset, no checkpoints, no run logs.

## Metrics

- **Integral, per car** (surface integration of predicted fields, frontal-area reference):
  drag `c_d` = pressure part `c_p` + friction part `c_wss`; lift `c_l`; side `c_s`.
  Cars are symmetric at zero yaw, so ground-truth `c_s ≈ 0` — it is a sanity metric.
  Expect lift R² well below drag R²: lift is a small residual of cancelling integrals.
- **Field**: relative L2 of pressure and WSS (plus MSE, kept for continuity).

## Tuning protocol

`scripts/tune.py` — TPE (seeded) + Hyperband pruning (rounds 10→20, η=3), one trial =
one 20-round proxy run, **objective = mean drag MAE over the last 5 evaluated rounds**
(single final rounds are noise). Lift/rel-L2 recorded per trial as an audit: the winner
must not be an outlier on them. Baselines tune a single faithful LR (+β); FedMuon tunes
its two structural LRs (+β); server step fixed at 1.0 for everyone (redundant with client
LRs). Ladders are 1e-x/3e-x and every reported optimum is **interior** — edge hits
auto-flag and extend. The 150-sample proxy acts as validation; benchmarks report the
full test set.

```
python scripts/tune.py --stage fedmuon  --n-trials 24     # also: scaffold, fedavg
python scripts/tune.py --stage gamma    --n-trials 5      # MARS gamma at FedMuon's best
python scripts/tune.py --stage fedmuon --alpha 0.5 --n-trials 3   # per-split warm-start
```

## Benchmarks

```
python scripts/benchmark.py --suite headline   --gpu 0            # 5 methods, 3 seeds, full task
python scripts/benchmark.py --suite splits     --setting proxy --gpu 0   # Dirichlet sweep
python scripts/benchmark.py --suite localsteps --setting proxy --gpu 0   # communication ablation
python scripts/benchmark.py --suite gamma      --gpu 0            # gamma ablation, error bars
python scripts/report.py                                          # tables
```

Runs are resumable (existing `summary.csv` ⇒ skipped); run one `benchmark.py` per GPU.

## Tuned configurations (batch 32, proxy objective; all interior)

| method | config | mean c_d (rounds 16–20) |
|---|---|---|
| FedMARS (γ=0.01) | muon-lr 3e-3, scalar-lr 1e-5, β 0.8 | 6.10e-4 |
| FedMuon | muon-lr 3e-3, scalar-lr 1e-5, β 0.8 | 6.46e-4 |
| SCAFFOLD | lr 1e-4, β 0.8 | 1.26e-3 |
| FedAvg | lr 1e-3, β 0.9 | 1.62e-3 |

γ's effect is smaller than single-run noise; the benchmark carries γ ∈ {0, 0.01, 0.04}
with error bars rather than claiming a tuned value.

[cfd_transformer]: ../cfd_transformer

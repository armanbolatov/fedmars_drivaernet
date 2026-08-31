# FedMuon / FedMARS-M on DrivAerNet++

Federated learning for CFD surrogate modelling: predict aerodynamic force coefficients
(drag **c_d**, lift **c_l**, side **c_s**) and surface fields (pressure, wall shear stress)
of DrivAerNet++ car geometries, with clients split by car body type (natural non-IID) or by
Dirichlet allocation.

Six federated methods and two centralized upper bounds:

| method | client update | server |
|---|---|---|
| FedAvg | AdamW | weight averaging |
| SCAFFOLD-Adam | Adam-normalized corrected momentum | + control variates |
| FedLion | Lion (sign of interpolated momentum) | averages weights **and** momentum |
| FedMuon | Newton–Schulz orthogonalized momentum on matrix parameters | + control variates |
| FedMARS-M | FedMuon + MARS variance-reduced gradient | + control variates |
| FedOLion | FedMuon with the orthogonalized direction passed through a sign | + control variates |

FedMuon ≡ FedMARS-M with γ = 0 — one implementation, one flag. FedOLion is our federated
adaptation of OLion; the centralized algorithm is due to its authors, the federated
wrapping is ours.

## Layout

```
configs/               protocol (base.yaml), per-method and per-split configs
src/train.py           the trainer (all methods; one file)
src/eval_ext.py        evaluation: c_d/c_l/c_s + relative-L2 field errors
src/forces.py          lift & side coefficients (directional force integration)
scripts/tune.py        stage 1: Bayesian search at reduced fidelity
scripts/tune_full.py   stages 2-3: bracketed ladders at evaluation scale
scripts/benchmark.py   benchmark matrix: methods x splits x seeds x local epochs
scripts/report.py      aggregate results/runs -> mean±std tables
scripts/plot.py        figures
scripts/eval_round0.py round-0 point of the convergence curves
scripts/validate.py    integrity check over every completed run
scripts/predict.py     evaluate or predict fields from a saved checkpoint
scripts/status.py      read-only progress monitor
results/               small CSVs, tuning exports, figures (no raw data)
```

Nothing heavy lives in this repo: no dataset, no checkpoints, no run logs.

## Setup

1. Conda env `cfd_transformer` with `requirements.txt` installed.
2. The [cfd_transformer] repo (model, datamodule, compiled force integrator) at
   `~/cfd_transformer`, or set `CFD_ROOT`.
3. DrivAerNet++ `.npy` dataset at `$CFD_ROOT/dataset/DrivAerNetPP` with
   `split/{train,test}_split.json`.

## Metrics

- **Integral, per car** (surface integration of predicted fields, frontal-area reference):
  drag `c_d` = pressure part `c_p` + friction part `c_wss`; lift `c_l`; side `c_s`.
  Cars are symmetric at zero yaw, so ground-truth `c_s ≈ 0` — it is a sanity metric.
  Expect lift R² well below drag R²: lift is a small residual of cancelling integrals.
- **Field**: relative L2 of pressure and wall shear stress.

## Tuning and benchmarking

**[`docs/protocol.md`](docs/protocol.md) is the procedure of record** — three tuning
stages, the benchmark suites, and the choices behind them. In short: every method gets
the same budget of 24 distinct configurations from the same sampler at reduced fidelity,
then bracketed ladder walks at evaluation scale select the learning rate per distribution
and the local-epoch budget on the reference split. Selection is on drag; lift is reported
alongside and never optimised against.

```
# tuning
python scripts/tune.py --stage fedmuon                             # stage 1
python scripts/tune.py --stage fedmuon --probe-edges               #   certify a ladder end
python scripts/tune_full.py --methods fedmuon --gpu 0              # stage 2: learning rate
python scripts/tune_full.py --methods fedmuon --epochs --gpu 0     # stage 3: local epochs
python scripts/tune_full.py --methods fedmuon --rebracket --gpu 0  #   rate at the chosen E

# benchmark, tables and figures
python scripts/benchmark.py --suite headline --gpu 0     # reference split, tuned E
python scripts/benchmark.py --suite splits --gpu 0       # Dirichlet sweep, E = 1
python scripts/benchmark.py --suite gamma --gpu 0        # MARS gamma ablation
python scripts/report.py && python scripts/plot.py       # tables and figures
python scripts/validate.py                               # integrity check
```

Selections land in `results/tune_full_<method>.json`, `tune_epochs_<method>.json` and
`tune_rebracket_<method>.json`, each recording every rung evaluated. `benchmark.py` reads
them directly, so those files are the tuning of record.

Runs are resumable (an existing `summary.csv` is skipped) and one `benchmark.py` drives
one GPU. `scripts/status.py` prints live runs, ladder state and benchmark coverage.

[cfd_transformer]: ../cfd_transformer

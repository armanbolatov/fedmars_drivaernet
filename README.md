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
configs/               protocol (base.yaml), per-method, per-split and per-model configs
src/train.py           the trainer (all methods; one file)
src/models/            model registry: configs/model/<name>.yaml -> network
src/eval_ext.py        evaluation: c_d/c_l/c_s + relative-L2 field errors
src/forces.py          lift & side coefficients (directional force integration)
scripts/tune.py        stage 1: Bayesian search at reduced fidelity
scripts/tune_full.py   stages 2-3: bracketed ladders at evaluation scale
scripts/benchmark.py   benchmark suites: methods x splits x seeds x local epochs
scripts/run_bench.sh   retry wrapper for benchmark.py on a shared card
scripts/report.py      aggregate results/runs -> mean±std tables
scripts/plot.py        figures
scripts/validate.py    integrity check over every completed run
scripts/status.py      read-only progress monitor
scripts/eval_round0.py round-0 point of the convergence curves
scripts/predict.py     evaluate or predict fields from a saved checkpoint
results/               tuning records, tables, figures (no raw runs)
CLAUDE.md              working conventions: what must not change, server etiquette
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

## Protocol

Every method gets the same budget of 24 distinct configurations from the same sampler at
reduced fidelity (stage 1), then bracketed ladder walks at evaluation scale select the
learning rate per distribution (stage 2) and the local-epoch budget E on the reference
split, with a one-rung learning-rate re-check at that E (stage 3). A tuning rung scores
as the mean drag MAE at rounds 20, 25 and 30. Selection is on drag; lift is reported
alongside and never optimised against.

The benchmark runs 100 rounds at full resolution on the full test set, seed 42, and
reports the mean over rounds 96–100:

| suite | rows |
|---|---|
| `headline` | reference split (single-class), every method at its tuned E, plus both centralized ceilings |
| `splits` | four client distributions, every method at E = 1 |
| `participation` | K ∈ {1, 2, 3} of 6 clients per round, tuned E (K = 6 is the headline row) |
| `gamma` | FedMARS-M at γ ∈ {0, 0.01, 0.04} |
| `epochs` | local-epoch ladder on the Dirichlet splits (30-round setting) |

The Muon family anneals its matrix learning rate with a cosine schedule; the AdamW and
Lion baselines were tuned at a constant rate. A schedule check (FedAvg with the same
cosine decay, `--lr-decay cosine`) quantifies the difference; see Results.

```
# tuning
python scripts/tune.py --stage fedmuon                             # stage 1
python scripts/tune.py --stage fedmuon --probe-edges               #   certify a ladder end
python scripts/tune_full.py --methods fedmuon --gpu 0              # stage 2: learning rate
python scripts/tune_full.py --methods fedmuon --epochs --gpu 0     # stage 3: local epochs
python scripts/tune_full.py --methods fedmuon --rebracket --gpu 0  #   rate at the chosen E

# benchmark, tables and figures
scripts/run_bench.sh 0 --suite headline                  # retried until every row is done
scripts/run_bench.sh 0 --suite splits
scripts/run_bench.sh 0 --suite participation
scripts/run_bench.sh 0 --suite epochs --setting full30
python scripts/report.py && python scripts/plot.py       # tables and figures
python scripts/validate.py                               # integrity check
```

Selections land in `results/tune_full_<method>.json`, `tune_epochs_<method>.json` and
`tune_rebracket_<method>.json`, each recording every rung evaluated. `benchmark.py` reads
them directly, so those files are the tuning of record.

A run checkpoints its full state after every round and continues from it if restarted,
reproducing the uninterrupted run. The test set is evaluated every fifth round and at
every round of the reporting window; a skipped evaluation still makes the random draws
an evaluation makes, so the cadence never changes a trajectory. Runners on several GPUs
share one work list: finished rows are skipped, rows in progress are locked, and the
longest rows start first.

## Another model

The trainer is architecture-agnostic: a model is any `nn.Module` whose forward takes
the batch's input dict (`centroid` and `normal`, both (B, N, 3)) and returns
`{"pressure": (B, N, 1), "wss": (B, N, 3)}`. Add `configs/model/<name>.yaml` (a hydra
`model:` node plus `muon_exclude`, the embedding-like layers kept on Adam), make the
class importable, and pass `--model <name>` to every script:

```
python scripts/tune.py --stage fedmuon --model omniarch
python scripts/tune_full.py --methods fedmuon --gpu 0 --model omniarch
scripts/run_bench.sh 0 --suite headline --model omniarch
python scripts/report.py --setting full-omniarch
```

Runs, tuning records and Optuna exports of another model live in their own namespace
(`results/runs/full-omniarch/`, `results/tune_full_fedmuon-omniarch.json`, ...), so the
Transolver campaign is untouched. Hyperparameters do not transfer: run all three
tuning stages before benchmarking. `configs/model/omniarch.yaml` (the OmniArch
implementation shipped with cfd_transformer) is a ready second model.

## Results so far

Seed 42, full resolution, 100 rounds, mean over rounds 96–100. Values are ×10⁻³, given
as drag MAE / lift MAE. 34 of 53 rows are finished (29 September 2026); the AdamW and
Lion baseline rows marked pending wait on the schedule decision below.

**Headline** — single-class split, every method at its tuned local epochs E:

| method | E | drag / lift |
|---|---|---|
| FedMARS-M | 2 | **6.73 / 26.0** |
| Centralized Muon | – | 6.82 / 26.1 |
| FedMuon | 4 | 6.93 / 26.4 |
| FedOLion | 8 | 8.45 / 27.7 |
| FedAvg | 16 | 10.3 / 43.2 |
| SCAFFOLD-Adam | 8 | pending |
| FedLion | 16 | pending |
| Centralized AdamW | – | pending |

FedOLion's 8-epoch budget, chosen on 30 rounds, did not transfer to 100 rounds: its
1-epoch row reaches 7.46 / 28.1.

**Heterogeneity** — one local epoch, per-split learning rate:

| method | single-class | α = 0.5 | α = 10 | α = 1000 |
|---|---|---|---|---|
| FedMARS-M | 6.93 / 26.9 | 6.78 / 26.6 | 6.84 / 26.8 | 6.84 / 26.8 |
| FedMuon | 7.07 / 27.3 | 6.94 / 27.1 | 6.80 / 27.0 | 6.88 / 26.9 |
| FedOLion | 7.46 / 28.1 | 7.23 / 27.4 | 6.93 / 26.9 | 7.03 / 27.4 |
| SCAFFOLD-Adam | 16.9 / 52.0 | pending | 12.6 / 45.4 | pending |
| FedAvg | 16.3 / 60.7 | 16.8 / 53.1 | pending | pending |
| FedLion | pending | pending | pending | pending |

**Partial participation** — single-class, tuned E, drag MAE by clients per round K:

| method | K = 1 | K = 2 | K = 3 | K = 6 |
|---|---|---|---|---|
| FedMARS-M | 7.21 | 6.65 | 6.69 | 6.73 |
| FedMuon | 7.30 | 7.07 | 6.85 | 6.93 |
| FedOLion | 8.59 | 7.47 | 7.02 | 8.45 |
| FedAvg | pending | 12.1 | pending | 10.3 |

**MARS correction** — FedMARS-M, single-class, one local epoch: γ = 0 gives 7.43 / 27.3,
γ = 0.01 gives 6.93 / 26.9, γ = 0.04 gives 6.94 / 27.2.

**Schedule check** — single-class, 30 rounds, FedAvg at 16 local epochs, mean drag MAE
at rounds 20, 25, 30: constant rate 3e-4 gives 1.074e-2, the same rate with cosine
decay gives 1.019e-2; FedMARS-M at its tuned budget gives 9.26e-3 on the same setting.
Annealing closes about a third of the gap at this budget. A second row (peak rate
1e-3 with cosine decay) is running.

Figures: `results/figures/` (heterogeneity, local epochs, convergence, participation,
learning-rate ladders); tables: `results/tables/`.

[cfd_transformer]: ../cfd_transformer

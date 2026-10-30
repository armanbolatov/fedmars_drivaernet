# FedMuon / FedMARS-M on DrivAerNet++

Federated learning for CFD surrogate modelling: predict aerodynamic force coefficients
(drag **c_d**, lift **c_l**, side **c_s**) and surface fields (pressure, wall shear stress)
of DrivAerNet++ car geometries, with clients split by car body type (natural non-IID) or by
Dirichlet allocation.

Six federated methods and two centralized upper bounds. The spectral methods update each
weight matrix along an orthogonalized direction; the non-spectral methods update every
parameter element-wise.

| method | type | client update | server |
|---|---|---|---|
| FedMuon | spectral | Newton–Schulz orthogonalized momentum on matrix parameters | + control variates |
| FedMARS-M | spectral | FedMuon + MARS variance-reduced gradient | + control variates |
| FedOLion | spectral | FedMuon with the orthogonalized direction passed through a sign | + control variates |
| FedAvg | non-spectral | AdamW | weight averaging |
| SCAFFOLD-Adam | non-spectral | Adam-normalized corrected momentum | + control variates |
| FedLion | non-spectral | Lion (sign of interpolated momentum) | averages weights **and** momentum |

FedMuon ≡ FedMARS-M with γ = 0 — one implementation, one flag. FedOLion is a federated
adaptation of OLion, whose centralized algorithm is due to its authors.

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
scripts/grad_spectrum.py  gradient spectra per weight matrix and client (stable rank, gain, sign-step size)
scripts/plot_spectrum.py  figure of the above
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
reduced fidelity (stage 1), then a bracketed ladder walk at evaluation scale selects the
learning rate per distribution (stage 2), the whole ladder E ∈ {1, 2, 4, 8, 16} is
evaluated on the reference split, and the rate is re-checked one rung either way at the
selected E (stage 3). A tuning rung scores
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
| `epochs` | local-epoch ladder E ∈ {2, 4, 8, 16} on the Dirichlet splits for every method (30-round setting; E = 1 is the stage-2 row) |

Every method anneals its learning rate to zero with a cosine schedule over the run
(`--muon-lr-decay cosine` for the spectral methods' matrix rate, `--lr-decay cosine` for
the AdamW and Lion rates). The non-spectral methods were first tuned at a constant rate,
as they are usually run in federated learning; that was a mistake, and re-tuned from
stage 1 under the schedule they improve by 13 to 26%. The re-tuned `fedavg_cosine`,
`scaffold_cosine`, `fedlion_cosine2` and `centralized_cosine` configs are the
configs of record (`fedlion_cosine` is FedLion's stage-1 winner, which did not
transfer to full resolution). The constant-rate selections and rows stay under the
original names as the schedule ablation; the figures and tables leave them out (their
ladder is kept in `results/superseded/schedule_ablation/`).

The non-spectral methods were tuned on RTX A6000 cards and the spectral methods on
A100s. All 100-round rows run on A100s (the MBZUAI DANA cluster reproduces the A100
server's digits exactly); a trajectory is bit-reproducible on one card type but not
across the two.

```
# tuning
python scripts/tune.py --stage fedmuon                             # stage 1
python scripts/tune.py --stage fedmuon --probe-edges               #   certify a ladder end
python scripts/tune_full.py --methods fedmuon --gpu 0              # stage 2: learning rate
python scripts/tune_full.py --methods fedmuon --epochs --gpu 0     # stage 3: every E
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
as drag MAE / lift MAE. The spectral methods' rows are final. The non-spectral methods'
rows under the cosine schedule are running on the DANA cluster (October 2026) and are
marked pending; their constant-rate rows of September are the schedule ablation and
are not shown.

**Headline** — single-class split, every method at its tuned local epochs E:

| method | E | drag / lift |
|---|---|---|
| FedMARS-M | 2 | **6.73 / 26.0** |
| Centralized Muon | – | 6.82 / 26.1 |
| FedMuon | 4 | 6.93 / 26.4 |
| FedOLion | 8 | 8.45 / 27.7 |
| SCAFFOLD-Adam | 16 | 7.94 / 27.0 |
| FedLion | 16 | 8.94 / 33.3 |
| FedAvg | 16 | 9.26 / 37.1 |
| Centralized AdamW | – | 7.54 / 27.9 |

FedOLion's 8-epoch budget, chosen on 30 rounds, did not transfer to 100 rounds: its
1-epoch row reaches 7.46 / 28.1.

**Heterogeneity** — one local epoch, per-split learning rate:

| method | single-class | α = 0.5 | α = 10 | α = 1000 |
|---|---|---|---|---|
| FedMARS-M | 6.93 / 26.9 | 6.78 / 26.6 | 6.84 / 26.8 | 6.84 / 26.8 |
| FedMuon | 7.07 / 27.3 | 6.94 / 27.1 | 6.80 / 27.0 | 6.88 / 26.9 |
| FedOLion | 7.46 / 28.1 | 7.23 / 27.4 | 6.93 / 26.9 | 7.03 / 27.4 |
| SCAFFOLD-Adam | 13.7 / 51.1 | 14.2 / 45.9 | 14.4 / 45.3 | 14.3 / 45.5 |
| FedLion | 16.4 / 51.2 | 11.2 / 41.0 | 12.8 / 43.7 | 13.2 / 45.2 |
| FedAvg | 17.0 / 57.8 | 17.5 / 46.8 | 32.6 / 54.6 | 16.7 / 50.8 |

FedAvg at α = 10 runs at its stage-2 rate 3e-3, which won the 30-round ladder narrowly
but trains to a higher loss over 100 rounds; its drag error settles at 32.6 while the
field error keeps falling.

**Partial participation** — single-class, tuned E, drag MAE by clients per round K:

| method | K = 1 | K = 2 | K = 3 | K = 6 |
|---|---|---|---|---|
| FedMARS-M | 7.21 | 6.65 | 6.69 | 6.73 |
| FedMuon | 7.30 | 7.07 | 6.85 | 6.93 |
| FedOLion | 8.59 | 7.47 | 7.02 | 8.45 |
| SCAFFOLD-Adam (E = 16) | pending | pending | 8.01 | 7.94 |
| FedLion (E = 16) | pending | pending | pending | 8.94 |
| FedAvg (E = 16) | pending | pending | 9.80 | 9.26 |

**MARS correction** — FedMARS-M, single-class, one local epoch: γ = 0 gives 7.43 / 27.3,
γ = 0.01 gives 6.93 / 26.9, γ = 0.04 gives 6.94 / 27.2.

**Local epochs** — the stage-3 ladder, single-class, 30 rounds, drag MAE at the rate
selected at one epoch (bold: the selected budget):

| method | E = 1 | E = 2 | E = 4 | E = 8 | E = 16 |
|---|---|---|---|---|---|
| FedMARS-M | 10.8 | **9.26** | 9.29 | 16.2 | 12.0 |
| FedMuon | 11.6 | 10.3 | **9.63** | 14.0 | 20.2 |
| FedOLion | 12.7 | 9.13 | 8.90 | **8.66** | 13.2 |
| SCAFFOLD-Adam | 17.4 | 19.0 | 13.9 | 11.7 | **10.7** |
| FedAvg | 18.7 | 19.1 | 18.3 | 14.4 | **11.6** |
| FedLion | 25.9 | 22.4 | 16.3 | 13.7 | **10.3** |

The spectral methods are best at two to eight epochs; the non-spectral methods keep
improving to the end of the ladder. With the rate re-checked at sixteen epochs FedAvg reaches 9.19 (3e-4)
and SCAFFOLD-Adam 9.58 (1e-4), level with FedMARS-M at two epochs on drag.

**Local epochs on the Dirichlet splits** — 30 rounds, mean at rounds 20, 25, 30, at the
split's one-epoch learning rate (drag / lift):

| method | split | E = 1 | E = 2 | E = 4 | E = 8 | E = 16 |
|---|---|---|---|---|---|---|
| FedMARS-M | α = 0.5 | 9.25 / 36.0 | 7.80 / 30.2 | **6.89** / 27.3 | 7.50 / **26.9** | 55.2 / 45.4 (unstable) |
| FedMARS-M | α = 10 | 8.86 / 36.1 | 7.71 / 29.1 | 6.98 / 27.0 | **6.83 / 26.5** | 6.99 / 27.7 |
| FedMARS-M | α = 1000 | 9.12 / 35.8 | 7.66 / 29.6 | 6.96 / 27.0 | **6.78 / 26.2** | 7.05 / 27.1 |
| FedMuon | α = 0.5 | 9.68 / 37.5 | 8.34 / 32.0 | 7.19 / 27.3 | **6.89 / 26.5** | 27.6 / 126 (unstable) |
| FedMuon | α = 10 | 9.57 / 37.7 | 7.91 / 30.2 | 7.03 / 27.4 | 6.87 / **26.8** | **6.84** / 27.8 |
| FedMuon | α = 1000 | 9.91 / 36.5 | 7.95 / 30.4 | **6.89** / 27.4 | 6.91 / **26.6** | 6.99 / 26.9 |

Error falls steadily up to four local epochs on every split (about 25% from one to
four) and is flat from four to eight. Sixteen adds nothing on the mild splits and is
unstable at α = 0.5.

**Schedule ablation** — the same ladder with the non-spectral methods at a constant rate, drag MAE
at one and at sixteen local epochs: FedAvg 21.5 and 13.6 (cosine: 18.7 and 11.6),
SCAFFOLD-Adam 22.8 and 14.4 (17.4 and 10.7), FedLion 31.6 and 12.2 (25.9 and 10.3).

Figures: `results/figures/` (convergence, heterogeneity, local epochs, participation,
learning-rate ladders); tables: `results/tables/`.

[cfd_transformer]: ../cfd_transformer

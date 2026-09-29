#!/usr/bin/env python
"""Evaluate the untrained model -- the round-0 point of every convergence curve.

The benchmark runs with --skip-round0-eval (a full-resolution pass over 1154 cars
is expensive and the value is identical for every method), so round 0 is missing
from history.csv. It only depends on the seed: `seed_everything` runs before
`make_model`, so all methods sharing a seed start from the same weights. One
evaluation per seed therefore serves every curve.

  python scripts/eval_round0.py --seeds 42 43 44 --gpu 0

Writes results/round0.csv (seed, metric columns), which plot.py prepends at round 0.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import train as trainer          # noqa: F401  (sets up CFD_ROOT on sys.path)
import ppbc_drivaernetpp_transformer as base
import eval_ext
import models

METRICS = ("c_d_mae", "c_d_r2", "c_l_mae", "c_l_r2", "c_s_mae",
           "rel_l2_p", "rel_l2_wss")


def round0_row(cfg_path, seed, gpu):
    args = argparse.Namespace(**json.loads(Path(cfg_path).read_text()))
    args.seed = seed
    args.eval_batch_size = 1
    args.gpu = gpu

    repo_cfg = base.load_repo_cfg(args)
    fabric = base.make_fabric(args)
    fabric.launch()
    base.seed_everything(repo_cfg, fabric, workers=True)   # same order as training
    state = base.clone_state_cpu(models.make_model(args).state_dict())

    records = base.read_records(Path(args.data_root) / "split" / "test_split.json",
                                Path(args.data_root), args.max_test_samples,
                                args.seed + 1)
    dataset = base.make_dataset(records, args.test_downsample_size)
    rows = eval_ext.evaluate_state(state, dataset, records, args, repo_cfg, fabric,
                                   base.PointCloudDataModule(base.MEAN_STD_DICT),
                                   "init", 0.0, 0)
    overall = rows[0]
    return {"seed": seed, **{m: overall.get(m) for m in METRICS}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="results/runs/full/single_class/"
                                        "fedmuon_E1_s42/config.json",
                    help="any completed full run: supplies the eval protocol")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--out", default="results/round0.csv")
    a = ap.parse_args()

    rows = [round0_row(ROOT / a.config, s, a.gpu) for s in a.seeds]
    out = ROOT / a.out
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", *METRICS])
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"seed {r['seed']}: c_d_mae={r['c_d_mae']:.4e} c_l_mae={r['c_l_mae']:.4e}")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

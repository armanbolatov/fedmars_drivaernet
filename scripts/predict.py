#!/usr/bin/env python
"""Load a saved checkpoint and evaluate it (or predict fields with it).

The checkpoint stores the full training config, so the evaluation protocol is
reconstructed from it and nothing has to be re-specified by hand.

  # full metrics on the test set (drag/lift/side + field errors)
  python scripts/predict.py --checkpoint results/checkpoints/centralized_muon_full.pt

  # quick check on 20 cars
  python scripts/predict.py --checkpoint model.pt --max-samples 20

  # per-car predictions to CSV
  python scripts/predict.py --checkpoint model.pt --out-csv cars.csv

Requires the cfd_transformer repo (CFD_ROOT env var, default ~/cfd_transformer)
and its conda environment -- the same ones used for training.
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch

import train as trainer          # sets up CFD_ROOT on sys.path as a side effect
import ppbc_drivaernetpp_transformer as base
import eval_ext
import models


def load_model(checkpoint, device="cuda"):
    """Checkpoint -> ready-to-use model. Use this for plain field prediction:

        model = load_model("model.pt")
        outputs = model(inputs)      # dict with 'pressure' and 'wss' predictions
    """
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = models.make_model(argparse.Namespace(**ck["args"]))
    model.load_state_dict(ck["state"])
    return model.to(device).eval(), ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", help="default: the path stored in the checkpoint")
    ap.add_argument("--max-samples", type=int, default=0, help="0 = whole test set")
    ap.add_argument("--downsample-size", type=int,
                    help="eval points per car; default: as trained (-1 = full resolution)")
    ap.add_argument("--out-csv", help="write per-car predicted vs GT coefficients")
    a = ap.parse_args()

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    args = argparse.Namespace(**ck["args"])         # exact training configuration
    args.eval_batch_size = 1                        # repository metric semantics
    args.max_test_samples = a.max_samples
    if a.data_root:
        args.data_root = a.data_root
    if a.downsample_size is not None:
        args.test_downsample_size = a.downsample_size
    print(f"checkpoint: {a.checkpoint}\nmethod: {ck['method']}\n"
          f"eval points/car: {args.test_downsample_size} (-1 = full resolution)")

    repo_cfg = base.load_repo_cfg(args)
    fabric = base.make_fabric(args)
    fabric.launch()
    records = base.read_records(Path(args.data_root) / "split" / "test_split.json",
                                Path(args.data_root), args.max_test_samples, args.seed + 1)
    dataset = base.make_dataset(records, args.test_downsample_size)
    datamodule = base.PointCloudDataModule(base.MEAN_STD_DICT)

    per_car, original = [], eval_ext.case_result_rows

    def capture(*call):                       # collect per-car rows during the one pass
        rows = original(*call)
        per_car.extend(rows)
        return rows

    eval_ext.case_result_rows = capture
    try:
        rows = eval_ext.evaluate_state(ck["state"], dataset, records, args, repo_cfg,
                                       fabric, datamodule, ck["method"], 0.0, 0)
    finally:
        eval_ext.case_result_rows = original

    overall = rows[0]
    print(f"\n{overall['n']} cars")
    for m in eval_ext.PRINT_METRICS:
        if m in overall:
            print(f"  {m:<12} {base.format_metric(overall[m])}")
    print("\nper body type:")
    for row in rows[1:]:
        if row["n"]:
            print(f"  {row['body']:<16} n={row['n']:<5} "
                  f"c_d_mae={base.format_metric(row['c_d_mae'])}  "
                  f"c_l_mae={base.format_metric(row['c_l_mae'])}")

    if a.out_csv:
        with open(a.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_car[0]))
            w.writeheader()
            w.writerows(per_car)
        print(f"\nwrote {len(per_car)} per-car rows to {a.out_csv}")


if __name__ == "__main__":
    main()

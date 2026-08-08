"""Evaluation with the full force set: drag (c_d = c_p + c_wss), lift (c_l), side (c_s),
plus relative-L2 field errors for pressure and wall shear stress.

Vendored from ppbc_drivaernetpp_transformer.evaluate_state/case_result_rows/metric_row/
print_round (only these four change); all other helpers are reused from that module.
Assumes CFD_ROOT is already on sys.path (train.py does this before importing us).
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

import ppbc_drivaernetpp_transformer as base
from util.cfd_utils import calc_force_coeff_wrapper

import forces

COEFFS = ("c_d", "c_p", "c_wss", "c_l", "c_s")
CASE_METRICS = ("mse_p", "mse_wss", "rae_c_p", "rae_c_wss", "rae_c_d",
                "rel_l2_p", "rel_l2_wss", "rel_l1_p", "rel_l1_wss")


def _rel_l2(pred, gt):
    denom = torch.linalg.vector_norm(gt)
    return float(torch.linalg.vector_norm(pred - gt) / denom) if denom > 0 else float("nan")


def _rel_l1(pred, gt):
    denom = gt.abs().sum()
    return float((pred - gt).abs().sum() / denom) if denom > 0 else float("nan")


def evaluate_state(state, dataset, records, args: argparse.Namespace, repo_cfg,
                   fabric, datamodule, method: str, alpha: float, round_idx: int,
                   train_loss: float | None = None):
    model = base.make_model()
    model.load_state_dict(state)
    model = fabric.setup_module(model)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=fabric.device.type == "cuda",
        drop_last=False,
    )
    loader = fabric.setup_dataloaders(loader)
    metric_wrapper = base.make_metric_wrapper(repo_cfg)
    case_rows = []

    with torch.no_grad():
        for inputs, targets, others in loader:
            file_paths = others["file_path"]
            outputs = model(inputs)
            datamodule._denormalize(outputs, targets, others)

            calc_force_coeff_wrapper(inputs, outputs, targets, others)
            forces.add_lift_side(inputs, outputs, targets, others)
            metric_dict = metric_wrapper(outputs, targets)
            # eval_batch_size is 1, so per-batch == per-case
            if "pressure_denorm" in outputs:
                metric_dict["rel_l2_p"] = torch.tensor(
                    _rel_l2(outputs["pressure_denorm"], targets["pressure_denorm"]))
                metric_dict["rel_l1_p"] = torch.tensor(
                    _rel_l1(outputs["pressure_denorm"], targets["pressure_denorm"]))
            if "wss_denorm" in outputs:
                metric_dict["rel_l2_wss"] = torch.tensor(
                    _rel_l2(outputs["wss_denorm"], targets["wss_denorm"]))
                metric_dict["rel_l1_wss"] = torch.tensor(
                    _rel_l1(outputs["wss_denorm"], targets["wss_denorm"]))
            case_rows.extend(case_result_rows(file_paths, outputs, targets, metric_dict))

    del model
    if fabric.device.type == "cuda":
        torch.cuda.empty_cache()

    base_row = {
        "method": method,
        "alpha": alpha,
        "round": round_idx,
        "train_loss": float("nan") if train_loss is None else train_loss,
    }
    rows = [metric_row(base_row, "all", case_rows)]
    for _, (body_name, _) in base.BODY_TO_CLASS.items():
        body_rows = [row for row in case_rows if row["body"] == body_name]
        rows.append(metric_row(base_row, body_name, body_rows))
    return rows


def case_result_rows(file_paths, outputs, targets, metric_dict):
    rows = []
    for idx, file_path in enumerate(file_paths):
        prefix = base.Path(file_path).stem.split("_", maxsplit=1)[0]
        row = {
            "file_name": base.Path(file_path).stem,
            "body": base.BODY_TO_CLASS[prefix][0],
        }
        for c in COEFFS:
            row[f"{c}_pred"] = float(outputs[c][idx].detach().cpu())
            row[f"{c}_gt"] = float(targets[c][idx].detach().cpu())
        for metric_name, metric in metric_dict.items():
            row[metric_name] = float(metric.detach().cpu())
        rows.append(row)
    return rows


def metric_row(base_row, body, case_rows):
    row = dict(base_row)
    row["body"] = body
    row["n"] = len(case_rows)
    if not case_rows:
        for c in COEFFS:
            row[f"{c}_mae"] = row[f"{c}_rmse"] = float("nan")
        for c in ("c_d", "c_l"):
            row[f"{c}_r2"] = float("nan")
        for m in CASE_METRICS:
            row[m] = float("nan")
        return row
    for c in COEFFS:
        pred = np.array([case[f"{c}_pred"] for case in case_rows], dtype=np.float64)
        gt = np.array([case[f"{c}_gt"] for case in case_rows], dtype=np.float64)
        row[f"{c}_mae"] = base.mae(pred, gt)
        row[f"{c}_rmse"] = base.rmse(pred, gt)
        if c in ("c_d", "c_l"):
            row[f"{c}_r2"] = base.r2(pred, gt)
    for m in CASE_METRICS:
        row[m] = base.mean_case_metric(case_rows, m)
    return row


PRINT_METRICS = ("c_d_mae", "c_d_r2", "c_l_mae", "c_l_r2", "c_s_mae",
                 "c_p_mae", "c_wss_mae", "rel_l2_p", "rel_l2_wss", "mse_p", "mse_wss")


def print_round(row, total_rounds):
    metrics = ", ".join(
        f"{name}: {base.format_metric(row[name])}" for name in PRINT_METRICS if name in row
    )
    print(
        f"[{str(row['method']).upper()}] alpha={float(row['alpha']):g}, "
        f"round={row['round']}/{total_rounds}, body={row['body']}, n={row['n']}, "
        f"train_loss: {base.format_metric(row['train_loss'])}, <Metrics> {metrics}",
        flush=True,
    )

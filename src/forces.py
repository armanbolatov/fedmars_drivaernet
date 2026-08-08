"""Lift and side force coefficients.

`calc_force_coeff` (cfd_transformer, compiled) integrates a per-point force field
projected onto an arbitrary unit direction and normalises by 0.5*rho*U^2*A_frontal
(verified empirically: +x force field -> full sum on +x, exactly 0 on +y/+z).
The stock wrapper only evaluates it along the flow direction (drag). Axes, from the
car geometry: x streamwise (drag), y lateral (side), z vertical (lift).

Reference area is the frontal area for all three coefficients (automotive convention).
Cars are symmetric at zero yaw, so ground-truth c_s ~ 0; it serves as a sanity metric.
Call AFTER base's calc_force_coeff_wrapper so denormalised fields/corrections exist.
"""
import torch
from util.cfd_utils_impl import calc_force_coeff  # CFD_ROOT put on sys.path by train.py

LIFT = (0.0, 0.0, 1.0)
SIDE = (0.0, 1.0, 0.0)


def _coeff(direction, inputs, d, others):
    dir_t = torch.zeros_like(others["flow_dir"])
    dir_t[:] = torch.tensor(direction, dtype=dir_t.dtype, device=dir_t.device)
    args = (others["air_density"], dir_t, others["flow_speed"],
            others["frontal_area"], inputs["area"])
    out = 0.0
    if "pressure_denorm" in d:
        out = out + calc_force_coeff(*args, d["pressure_denorm"] * inputs["normal"])
    if "wss_denorm" in d:
        out = out + calc_force_coeff(*args, d["wss_denorm"])
    return out


def add_lift_side(inputs, outputs, targets, others):
    for d in (outputs, targets):
        d["c_l"] = _coeff(LIFT, inputs, d, others)
        d["c_s"] = _coeff(SIDE, inputs, d, others)

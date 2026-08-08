"""Federated SCAFFOLD / FedMuon on DrivAerNet++ with the repository Transformer.

Reuses the data pipeline, client partitioning, and evaluation of
``ppbc_drivaernetpp_transformer.py`` from the cfd_transformer repo (imported as a
library, nothing there is modified). Adds two methods on top of its FedAvg:

- ``scaffold``: momentum-SGD local steps corrected by control variates,
  following the FedMuon paper's SCAFFOLD baseline: the local update is
  ``p -= lr * (m_i - c_i + c)`` and after each round ``c_i <- mean of m_i over
  the round's steps``, ``c <- weighted mean of c_i``.
- ``scaffold_adam``: same control variates, but the corrected momentum is
  normalized Adam-style: ``v <- beta2*v + (1-beta2)*(m_i - c_i + c)^2`` and
  ``p -= lr * (m_i - c_i + c) / (sqrt(v) + eps)``. This mirrors the intended
  semantics of the supplementary ``scaffold_adam_optimizer.py`` (whose ``v``
  buffer accidentally never persists) and is the fair baseline against the
  AdamW-based FedAvg.
- ``fedmuon``: bias-corrected LMO updates (FedMuon, arXiv:2509.26337). Matrix
  parameters take ``p -= muon_lr * NS(m_i - c_i + c) * sqrt(max(dims))`` with
  Newton-Schulz orthogonalization; non-matrix parameters (biases, norms) take
  the scaffold_adam-style Adam update with ``scalar_lr``. After each round
  ``c_i <- m_i``, ``c <- weighted mean of c_i``. Momentum ``m_i`` and second
  moments ``v_i`` persist across rounds per client.

Both methods use full client participation each round, matching the existing
FedAvg loop. Set ``CFD_ROOT`` env var if cfd_transformer is not in ``~``.
"""

import argparse
import json
import math
import time
import os
import sys
from pathlib import Path

CFD_ROOT = Path(os.environ.get("CFD_ROOT", Path.home() / "cfd_transformer")).resolve()
if not (CFD_ROOT / "ppbc_drivaernetpp_transformer.py").exists():
    raise FileNotFoundError(
        f"cfd_transformer repo not found at {CFD_ROOT}; set the CFD_ROOT env var."
    )
sys.path.insert(0, str(CFD_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import ppbc_drivaernetpp_transformer as base

import eval_ext


# ---------------------------------------------------------------------------
# Muon helpers (ported from the FedMuon supplementary code, with the shadowed
# double parameter loop of the original optimizer classes removed).
# ---------------------------------------------------------------------------

MUON_EXCLUDE_KEYWORDS = ("embed",)
MUON_EXTRA_EXCLUDES: list[str] = []
ADAM_EPS = 1e-8


def is_muon_param(name: str, param: torch.Tensor) -> bool:
    keywords = MUON_EXCLUDE_KEYWORDS + tuple(MUON_EXTRA_EXCLUDES)
    return param.ndim >= 2 and not any(k in name for k in keywords)


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Newton-Schulz iteration approximating the orthogonalization of G.

    With steps=0 this returns G normalized by its Frobenius norm, matching the
    T=0 case of the paper's inexact LMO.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.transpose(-2, -1)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.transpose(-2, -1)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.transpose(-2, -1)
    return X


def muon_update(
    corrected_momentum: torch.Tensor,
    ns_steps: int,
    scale_mode: str = "shape",
    scale_const: float = 1.0,
) -> torch.Tensor:
    """Spectral update of the corrected momentum.

    scale_mode:
      - "shape" (default, standard Muon/FedMuon): sqrt(max(d1,d2)) * NS(M). The
        orthogonalized direction has ~unit singular values, so the per-weight RMS
        step is ~constant regardless of gradient magnitude — a pure LMO that does
        not shrink near a stationary point (hence the eval oscillation).
      - "rms" (improved spectral-scaffold): additionally multiply by rms(M), the
        root-mean-square of the corrected momentum. This keeps the spectral
        *conditioning* (well-scaled direction across singular subspaces) but makes
        the step *magnitude* track the gradient, so it shrinks toward convergence
        like a proper preconditioned descent step. Converges with constant LR.
    """
    M = corrected_momentum
    if M.ndim > 2:  # conv-style filters: flatten trailing dims
        M = M.view(len(M), -1)
    update = zeropower_via_newtonschulz5(M, steps=ns_steps)
    # scale_const=0.2 gives the Moonlight RMS-match (update RMS ~0.2, like AdamW);
    # scale_const=1.0 is the original FedMuon/Keller-Jordan scale (RMS ~1).
    update = update * (scale_const * max(update.size(-2), update.size(-1)) ** 0.5)
    if scale_mode == "rms":
        rms = M.norm() / (M.numel() ** 0.5 + 1e-12)
        update = update * rms
    return update


# ---------------------------------------------------------------------------
# Local training with control-variate correction (shared by both methods).
# ---------------------------------------------------------------------------


def train_client_corrected(
    method: str,
    base_state: dict[str, torch.Tensor],
    dataset,
    indices: list[int],
    args: argparse.Namespace,
    repo_cfg,
    fabric,
    momentum_cpu: dict[str, torch.Tensor],
    v_cpu: dict[str, torch.Tensor],
    last_grad_cpu: dict[str, torch.Tensor],
    local_c_cpu: dict[str, torch.Tensor],
    global_c_cpu: dict[str, torch.Tensor],
):
    """One client's local pass. Returns (pseudo_grad, loss, momentum, v, last_grad, new c_i)."""
    model = base.make_model()
    model.load_state_dict(base_state)
    # Capture parameter names before Fabric wraps the module (the wrapper
    # prefixes names with "_forward_module."); setup_module moves these same
    # Parameter objects to the device in place.
    params = dict(model.named_parameters())
    model = fabric.setup_module(model)

    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=fabric.device.type == "cuda",
        drop_last=False,
    )
    loader = fabric.setup_dataloaders(loader)
    loss_wrapper = base.make_loss_wrapper(args, repo_cfg)
    model.train()

    device = fabric.device
    momentum = {k: momentum_cpu[k].to(device) for k in params}
    v_buf = {k: v_cpu[k].to(device) for k in params}
    last_grad = {k: last_grad_cpu[k].to(device) for k in params}
    local_c = {k: local_c_cpu[k].to(device) for k in params}
    global_c = {k: global_c_cpu[k].to(device) for k in params}
    use_mean_cv = method in ("scaffold", "scaffold_adam") or (
        method == "fedmuon" and args.cv_rule == "mean"
    )
    grad_accum = (
        {k: torch.zeros_like(t) for k, t in momentum.items()} if use_mean_cv else None
    )

    beta = args.fed_beta
    mars_factor = args.mars_gamma * beta / (1.0 - beta)
    total_loss = 0.0
    total_batches = 0
    num_steps = 0
    for _ in range(args.local_epochs):
        for batch_idx, data in enumerate(loader):
            if args.max_client_batches and batch_idx >= args.max_client_batches:
                break
            inputs, targets, others = data
            for p in params.values():
                p.grad = None
            outputs = model(inputs)
            loss, _ = loss_wrapper(inputs, outputs, targets, others)
            fabric.backward(loss)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    params.values(), max_norm=args.grad_clip, norm_type=2
                )
            with torch.no_grad():
                for name, p in params.items():
                    grad = p.grad
                    if grad is None:
                        continue
                    use_muon = (
                        method == "fedmuon"
                        and args.muon_mode != "off"
                        and is_muon_param(name, p)
                    )
                    g_in = grad
                    if use_muon and mars_factor > 0:
                        lg = last_grad[name]
                        c_t = grad + mars_factor * (grad - lg)
                        if args.mars_clip > 0:
                            cn = c_t.norm()
                            if cn > args.mars_clip:
                                c_t = c_t * (args.mars_clip / cn)
                        lg.copy_(grad)
                        g_in = c_t
                    m = momentum[name]
                    m.mul_(beta).add_(g_in, alpha=1.0 - beta)
                    corrected = m - local_c[name] + global_c[name]
                    if use_muon:
                        if args.muon_mode == "identity":
                            # LMO removed: raw corrected-momentum step
                            # (SGD-scaffold semantics for equivalence tests)
                            p.add_(corrected, alpha=-args.muon_lr_now)
                        else:
                            # decoupled (AdamW-style) weight decay, Moonlight-style:
                            # uses the base scheduled lr, anchors weight norm.
                            if args.muon_wd > 0:
                                p.mul_(1.0 - args.muon_lr_now * args.muon_wd)
                            update = muon_update(
                                corrected,
                                args.ns_steps,
                                args.muon_scale,
                                args.muon_scale_const,
                            )
                            p.add_(update.reshape(p.shape), alpha=-args.muon_lr_now)
                    elif method == "scaffold":
                        p.add_(corrected, alpha=-args.scaffold_lr)
                    else:
                        # Adam-style normalization of the corrected momentum
                        # (scaffold_adam for all params; fedmuon for non-matrix)
                        vb = v_buf[name]
                        vb.mul_(args.fed_beta2).addcmul_(
                            corrected, corrected, value=1.0 - args.fed_beta2
                        )
                        if method == "fedmuon":
                            lr = args.scalar_lr
                        else:  # scaffold_adam: matrix params use scaffold_lr, rest scalar_lr
                            lr = args.scaffold_lr if is_muon_param(name, p) else args.scalar_lr
                        if args.adam_wd > 0:
                            p.mul_(1.0 - lr * args.adam_wd)
                        p.addcdiv_(corrected, vb.sqrt().add_(ADAM_EPS), value=-lr)
                    if grad_accum is not None:
                        grad_accum[name].add_(m)
            num_steps += 1
            total_loss += float(loss.detach().cpu())
            total_batches += 1

    if use_mean_cv:  # c_i <- mean momentum over the round (SCAFFOLD rule)
        new_local_c = {
            k: (t / max(num_steps, 1)).detach().cpu() for k, t in grad_accum.items()
        }
    else:  # c_i <- final local momentum (FedMuon paper rule)
        new_local_c = {k: t.detach().cpu().clone() for k, t in momentum.items()}
    new_momentum = {k: t.detach().cpu() for k, t in momentum.items()}
    new_v = {k: t.detach().cpu() for k, t in v_buf.items()}
    new_last_grad = {k: t.detach().cpu() for k, t in last_grad.items()}

    local_state = model.state_dict()
    pseudo_grad = {}
    for key, base_value in base_state.items():
        if torch.is_floating_point(base_value):
            pseudo_grad[key] = local_state[key].detach().cpu() - base_value
    del model
    if fabric.device.type == "cuda":
        torch.cuda.empty_cache()
    return (
        pseudo_grad,
        total_loss / max(total_batches, 1),
        new_momentum,
        new_v,
        new_last_grad,
        new_local_c,
    )


STAT_COLS = ("method", "alpha", "round", "train_loss", "update_norm", "client_norm_mean",
             "client_drift", "drift_ratio", "muon_lr", "seconds", "peak_gib")


def state_sq_norm(s) -> float:
    """Squared L2 norm of a state dict, over floating-point entries only."""
    total = None
    for v in s.values():
        if torch.is_floating_point(v):
            n = torch.linalg.vector_norm(v.detach().float()) ** 2
            total = n if total is None else total + n
    return float(total) if total is not None else 0.0


def log_round_stats(args, method, alpha, round_idx, train_loss, agg_update,
                    client_sq_sum, client_norm_mean, seconds):
    """One row per communication round, independent of the eval cadence.

    client_drift = sqrt(sum_i w_i ||d_i||^2 - ||d_bar||^2), the weighted-variance
    identity, so it costs two scalars rather than a copy of every client update.
    drift_ratio normalizes it by the aggregate step.
    """
    agg_norm = math.sqrt(max(state_sq_norm(agg_update), 0.0))
    drift = math.sqrt(max(client_sq_sum - agg_norm ** 2, 0.0))
    row = (method, f"{alpha:g}", round_idx, f"{train_loss:.6e}", f"{agg_norm:.6e}",
           f"{client_norm_mean:.6e}", f"{drift:.6e}",
           f"{(drift / agg_norm if agg_norm > 0 else float('nan')):.6e}",
           f"{getattr(args, 'muon_lr_now', float('nan')):.6e}", f"{seconds:.1f}",
           f"{torch.cuda.max_memory_reserved() / 2**30:.1f}")
    path = Path(args.output_dir) / "round_stats.csv"
    # truncate at round 1: a relaunch after an OOM must not keep the dead attempt's rows
    with path.open("w" if round_idx <= 1 else "a") as f:
        if f.tell() == 0:
            f.write(",".join(STAT_COLS) + "\n")
        f.write(",".join(str(x) for x in row) + "\n")


def run_corrected(
    method: str,
    initial_state: dict[str, torch.Tensor],
    train_dataset,
    test_dataset,
    test_records,
    clients: list[list[int]],
    args: argparse.Namespace,
    repo_cfg,
    fabric,
    datamodule,
    alpha: float,
) -> list[dict[str, float | int | str]]:
    state = base.clone_state_cpu(initial_state)
    active_clients = [client_id for client_id, indices in enumerate(clients) if indices]
    total_samples = sum(len(clients[client_id]) for client_id in active_clients)

    template = base.make_model()
    zero_params = {
        name: torch.zeros_like(p, device="cpu")
        for name, p in template.named_parameters()
    }
    if method == "fedmuon":
        muon_named = [
            (n, p) for n, p in template.named_parameters() if is_muon_param(n, p)
        ]
        other_numel = sum(
            p.numel() for n, p in template.named_parameters() if not is_muon_param(n, p)
        )
        print(
            f"[FEDMUON] mode={args.muon_mode}, scale={args.muon_scale}x{args.muon_scale_const:g}, "
            f"muon_wd={args.muon_wd:g}, mars_gamma={args.mars_gamma:g}, "
            f"mars_clip={args.mars_clip:g}, "
            f"ema={args.ema_decay:g}, cv_rule={args.cv_rule}, lr_decay={args.muon_lr_decay}, "
            f"excludes={MUON_EXCLUDE_KEYWORDS + tuple(MUON_EXTRA_EXCLUDES)}: "
            f"{len(muon_named)} matrix params "
            f"({sum(p.numel() for _, p in muon_named)} weights) via Muon "
            f"(ns_steps={args.ns_steps}, muon_lr={args.muon_lr:g}); "
            f"{other_numel} weights via Adam (scalar_lr={args.scalar_lr:g})",
            flush=True,
        )
    del template

    momentum = {
        cid: {k: t.clone() for k, t in zero_params.items()} for cid in active_clients
    }
    v_moments = {
        cid: {k: t.clone() for k, t in zero_params.items()} for cid in active_clients
    }
    last_grads = {
        cid: {k: t.clone() for k, t in zero_params.items()} for cid in active_clients
    }
    local_c = {
        cid: {k: t.clone() for k, t in zero_params.items()} for cid in active_clients
    }
    global_c = {k: t.clone() for k, t in zero_params.items()}
    ema_state = base.clone_state_cpu(state) if args.ema_decay > 0 else None

    if args.skip_round0_eval:
        history = []
    else:
        history = eval_ext.evaluate_state(
            state,
            test_dataset,
            test_records,
            args,
            repo_cfg,
            fabric,
            datamodule,
            method,
            alpha,
            0,
        )

    for round_idx in range(1, args.rounds + 1):
        if args.muon_lr_decay == "cosine":
            args.muon_lr_now = args.muon_lr * 0.5 * (
                1.0 + math.cos(math.pi * (round_idx - 1) / max(args.rounds, 1))
            )
        else:
            args.muon_lr_now = args.muon_lr
        aggregate_update = base.zero_state_like(state)
        losses = []
        new_local_cs = {}
        round_t0 = time.time()
        client_sq_sum, client_norms = 0.0, []
        for client_id in active_clients:
            pseudo_grad, loss, new_m, new_v, new_lg, new_ci = train_client_corrected(
                method,
                state,
                train_dataset,
                clients[client_id],
                args,
                repo_cfg,
                fabric,
                momentum[client_id],
                v_moments[client_id],
                last_grads[client_id],
                local_c[client_id],
                global_c,
            )
            weight = base.client_weight(
                client_id, active_clients, clients, total_samples, args.aggregation
            )
            base.add_scaled_state(aggregate_update, pseudo_grad, weight)
            sq = state_sq_norm(pseudo_grad)
            client_sq_sum += weight * sq
            client_norms.append(math.sqrt(max(sq, 0.0)))
            momentum[client_id] = new_m
            v_moments[client_id] = new_v
            last_grads[client_id] = new_lg
            new_local_cs[client_id] = new_ci
            losses.append(loss)

        state = base.state_plus_update(state, aggregate_update, scale=args.server_lr)
        for client_id in active_clients:
            local_c[client_id] = new_local_cs[client_id]
        weights = {
            cid: base.client_weight(
                cid, active_clients, clients, total_samples, args.aggregation
            )
            for cid in active_clients
        }
        for key in global_c:
            acc = torch.zeros_like(global_c[key])
            for cid in active_clients:
                acc.add_(local_c[cid][key], alpha=weights[cid])
            global_c[key] = acc

        if ema_state is not None:
            d = args.ema_decay
            for k in ema_state:
                if torch.is_floating_point(ema_state[k]):
                    ema_state[k].mul_(d).add_(state[k], alpha=1.0 - d)
                else:
                    ema_state[k] = state[k].clone()

        train_loss = float(np.mean(losses)) if losses else float("nan")
        log_round_stats(args, method, alpha, round_idx, train_loss, aggregate_update,
                        client_sq_sum, float(np.mean(client_norms)) if client_norms
                        else float("nan"), time.time() - round_t0)
        if base.should_evaluate(round_idx, args):
            rows = eval_ext.evaluate_state(
                ema_state if ema_state is not None else state,
                test_dataset,
                test_records,
                args,
                repo_cfg,
                fabric,
                datamodule,
                method,
                alpha,
                round_idx,
                train_loss,
            )
            history.extend(rows)
            eval_ext.print_round(rows[0], args.rounds)
    torch.save({"state": ema_state if ema_state is not None else state,
                "method": method, "args": vars(args)},
               Path(args.output_dir) / f"model_{method}.pt")
    return history


# ---------------------------------------------------------------------------
# CLI / main (mirrors the base script, with the new methods).
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "FedAvg/SCAFFOLD/FedMuon on DrivAerNet++ with the cfd_transformer "
            "Transformer. Vehicle body types define client heterogeneity."
        )
    )
    parser.add_argument(
        "--data-root", default=str(CFD_ROOT / "dataset" / "DrivAerNetPP")
    )
    parser.add_argument("--output-dir", default="log/fedmuon_scaffold_drivaernetpp")
    parser.add_argument(
        "--methods", nargs="+", default=["fedavg", "scaffold_adam", "fedmuon"]
    )
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.1])
    parser.add_argument("--num-clients", type=int, default=6)
    parser.add_argument(
        "--client-split-mode",
        default="single_class",
        choices=["dirichlet", "single_class"],
    )
    parser.add_argument(
        "--single-class-allocation",
        default="proportional",
        choices=["equal_classes", "proportional"],
    )
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=16)
    # FedAvg client optimizer (AdamW, as in the base script)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    # SCAFFOLD / FedMuon
    parser.add_argument("--fed-beta", type=float, default=0.9,
                        help="Momentum coefficient for scaffold/scaffold_adam/fedmuon local steps.")
    parser.add_argument("--fed-beta2", type=float, default=0.95,
                        help="Second-moment coefficient for the Adam-style updates.")
    parser.add_argument("--scaffold-lr", type=float, default=3.0e-4,
                        help="LR for scaffold (SGD) and scaffold_adam local steps.")
    parser.add_argument("--muon-lr", type=float, default=0.003,
                        help="LR for matrix params under the Muon (LMO) update.")
    parser.add_argument("--scalar-lr", type=float, default=3.0e-4,
                        help="LR for non-matrix params in fedmuon (Adam-style update).")
    parser.add_argument("--ns-steps", type=int, default=5,
                        help="Newton-Schulz iterations (T in the paper). 0 = Frobenius-normalized momentum.")
    parser.add_argument("--muon-mode", default="lmo", choices=["lmo", "identity", "off"],
                        help="lmo: Newton-Schulz LMO (default). identity: raw corrected momentum "
                             "(SGD-scaffold semantics, equivalence test). off: matrix params use "
                             "the Adam branch (scaffold_adam semantics, equivalence test).")
    parser.add_argument("--muon-lr-decay", default="none", choices=["none", "cosine"],
                        help="Per-round decay of muon-lr. LMO updates have constant norm, so "
                             "constant LR cannot converge to a point.")
    parser.add_argument("--muon-scale", default="shape", choices=["shape", "rms"],
                        help="shape: standard Muon sqrt(max(d))*NS(M) (constant-norm LMO). "
                             "rms: additionally scale by rms(M) so the step tracks gradient "
                             "magnitude and converges with constant LR (improved spectral-scaffold).")
    parser.add_argument("--muon-scale-const", type=float, default=1.0,
                        help="Constant multiplying sqrt(max(d)) in the Muon update. Moonlight uses "
                             "0.2 to match update RMS to AdamW, making muon-lr transferable.")
    parser.add_argument("--muon-wd", type=float, default=0.0,
                        help="Decoupled weight decay on Muon-updated params (Moonlight uses 0.1). "
                             "Anchors weight norm; prevents the output-RMS drift behind c_d oscillation.")
    parser.add_argument("--ema-decay", type=float, default=0.0,
                        help="EMA of the global model across rounds (0=off). Evaluates the EMA "
                             "model; averages out round-to-round c_d oscillation.")
    parser.add_argument("--mars-gamma", type=float, default=0.0,
                        help="MARS-M variance reduction on the Muon gradient (0=off, paper 0.025). "
                             "c_t = g + gamma*beta/(1-beta)*(g-g_prev), clipped to unit norm.")
    parser.add_argument("--server-lr", type=float, default=1.0,
                        help="Server (outer) LR: scales the aggregated client update. 1.0 = plain averaging.")
    parser.add_argument("--adam-wd", type=float, default=0.0,
                        help="Decoupled weight decay for the Adam branch (scaffold_adam / fedmuon scalars).")
    parser.add_argument("--mars-clip", type=float, default=1.0,
                        help="Clip the MARS corrected gradient c_t to this norm (0 = no clip). "
                             "MARS-M paper clips to 1; set 0 to isolate variance reduction.")
    parser.add_argument("--cv-rule", default="final", choices=["final", "mean"],
                        help="fedmuon control variate: 'final' = c_i<-last momentum (FedMuon paper), "
                             "'mean' = c_i<-mean momentum over round (SCAFFOLD rule, lower variance).")
    parser.add_argument("--muon-exclude", default="",
                        help="Comma-separated extra name substrings excluded from the Muon group.")
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--downsample-size", type=int, default=50000)
    parser.add_argument("--test-downsample-size", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--precision", default="bf16", choices=["fp32", "bf16"])
    parser.add_argument("--aggregation", default="equal", choices=["equal", "sample"])
    parser.add_argument("--pressure-weight", type=float, default=1.0)
    parser.add_argument("--wss-weight", type=float, default=1.0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--max-client-batches", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--skip-round0-eval", action="store_true",
                        help="Skip the (unused) full test eval of the untrained model at "
                             "round 0. Saves minutes per run during hyperparameter tuning.")
    parser.add_argument("--cache-data", action="store_true",
                        help="Pre-materialize the downsampled dataset into RAM once (single "
                             "process), then serve it without per-access downsampling. Freezes "
                             "the random downsample (identical every epoch). Unused by default.")
    return parser.parse_args()


class _ListDataset(torch.utils.data.Dataset):
    """Serves pre-materialized items from RAM (no disk, no downsampling at access time)."""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def train_client_fedavg(base_state, dataset, indices, args, repo_cfg, fabric):
    """base.train_client, but with a two-group AdamW: matrix params (same is_muon_param
    split as the corrected methods) at --lr, the rest at --scalar-lr, beta1=--fed-beta.
    Keeps the tuning protocol identical across methods: matrix-LR x scalar-LR x beta."""
    model = base.make_model()
    model.load_state_dict(base_state)
    matrix = [p for n, p in model.named_parameters() if is_muon_param(n, p)]
    rest = [p for n, p in model.named_parameters() if not is_muon_param(n, p)]
    optimizer = torch.optim.AdamW(
        [{"params": matrix, "lr": args.lr},
         {"params": rest, "lr": args.scalar_lr}],
        betas=(args.fed_beta, 0.999),
        weight_decay=args.weight_decay,
    )
    model, optimizer = fabric.setup(model, optimizer)
    loader = DataLoader(
        torch.utils.data.Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=fabric.device.type == "cuda",
        drop_last=False,
    )
    loader = fabric.setup_dataloaders(loader)
    loss_wrapper = base.make_loss_wrapper(args, repo_cfg)
    model.train()
    total_loss, total_batches = 0.0, 0
    for _ in range(args.local_epochs):
        for batch_idx, data in enumerate(loader):
            if args.max_client_batches and batch_idx >= args.max_client_batches:
                break
            inputs, targets, others = data
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss, _ = loss_wrapper(inputs, outputs, targets, others)
            fabric.backward(loss)
            if args.grad_clip > 0:
                fabric.clip_gradients(model, optimizer, max_norm=args.grad_clip, norm_type=2)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            total_batches += 1
    local_state = model.state_dict()
    grad = {}
    for key, base_value in base_state.items():
        if torch.is_floating_point(base_value):
            grad[key] = local_state[key].detach().cpu() - base_value
    del model
    if fabric.device.type == "cuda":
        torch.cuda.empty_cache()
    return grad, total_loss / max(total_batches, 1)


def run_fedavg_server_lr(initial_state, train_dataset, test_dataset, test_records,
                         clients, args, repo_cfg, fabric, datamodule, alpha):
    """FedAvg, but with the server (outer) learning rate applied.

    base.run_fedavg hard-codes the aggregation scale to 1.0, so the baseline could never
    use the server-lr that the corrected methods tune (1.5 beat 1.0 by ~6%). That would
    credit our method for a property of the aggregation step rather than of Muon. The
    client update is untouched (AdamW); only the aggregation scale is exposed."""
    state = base.clone_state_cpu(initial_state)
    active = [cid for cid, idx in enumerate(clients) if idx]
    total = sum(len(clients[cid]) for cid in active)
    if args.skip_round0_eval:
        history = []
    else:
        history = eval_ext.evaluate_state(state, test_dataset, test_records, args, repo_cfg,
                                      fabric, datamodule, "fedavg", alpha, 0)
    for round_idx in range(1, args.rounds + 1):
        aggregate_update = base.zero_state_like(state)
        losses = []
        round_t0 = time.time()
        client_sq_sum, client_norms = 0.0, []
        for cid in active:
            grad, loss = train_client_fedavg(state, train_dataset, clients[cid], args,
                                             repo_cfg, fabric)
            weight = base.client_weight(cid, active, clients, total, args.aggregation)
            base.add_scaled_state(aggregate_update, grad, weight)
            sq = state_sq_norm(grad)
            client_sq_sum += weight * sq
            client_norms.append(math.sqrt(max(sq, 0.0)))
            losses.append(loss)
        state = base.state_plus_update(state, aggregate_update, scale=args.server_lr)
        train_loss = float(np.mean(losses)) if losses else float("nan")
        log_round_stats(args, "fedavg", alpha, round_idx, train_loss, aggregate_update,
                        client_sq_sum, float(np.mean(client_norms)) if client_norms
                        else float("nan"), time.time() - round_t0)
        if base.should_evaluate(round_idx, args):
            rows = eval_ext.evaluate_state(state, test_dataset, test_records, args, repo_cfg,
                                       fabric, datamodule, "fedavg", alpha, round_idx,
                                       train_loss)
            history.extend(rows)
            eval_ext.print_round(rows[0], args.rounds)
    torch.save({"state": state, "method": "fedavg", "args": vars(args)},
               Path(args.output_dir) / "model_fedavg.pt")
    return history


def _precache_dataset(dataset):
    """Materialize every (already downsampled + normalized) item into RAM once, in a
    single process. The real per-access cost is the point downsample (random 6k-of-500k)
    + normalization -- CPU, not disk. Doing it once up front then serving from RAM makes
    extra local epochs and later rounds essentially free. Deliberately NO DataLoader
    workers: forking after CUDA init deadlocks the run. The random downsample is frozen
    at this pass, so it is applied identically every epoch (fair across methods)."""
    n = len(dataset)
    items = [None] * n
    for i in range(n):
        items[i] = dataset[i]
        if (i + 1) % 2000 == 0:
            print(f"  pre-cached {i + 1}/{n}", flush=True)
    return _ListDataset(items)


def main() -> None:
    args = parse_args()
    valid_methods = {"fedavg", "scaffold", "scaffold_adam", "fedmuon"}
    if not set(args.methods).issubset(valid_methods):
        raise ValueError(f"--methods must be a subset of {sorted(valid_methods)}")
    if args.eval_batch_size != 1:
        raise ValueError("--eval-batch-size must be 1 (repository metric semantics).")
    if args.muon_exclude:
        MUON_EXTRA_EXCLUDES.extend(s for s in args.muon_exclude.split(",") if s)
    args.muon_lr_now = args.muon_lr

    output_dir = Path(args.output_dir)
    log_path = base.setup_run_log(output_dir)
    print(f"Logging to {log_path}", flush=True)
    print(f"cfd_transformer root: {CFD_ROOT}", flush=True)
    base.check_device(args)
    repo_cfg = base.load_repo_cfg(args)
    fabric = base.make_fabric(args)
    fabric.launch()
    base.seed_everything(repo_cfg, fabric, workers=True)
    base.set_seed(args.seed)

    data_root = Path(args.data_root)
    split_dir = data_root / "split"
    train_records = base.read_records(
        split_dir / "train_split.json", data_root, args.max_train_samples, args.seed
    )
    test_records = base.read_records(
        split_dir / "test_split.json", data_root, args.max_test_samples, args.seed + 1
    )
    train_classes = np.array(
        [record.class_id for record in train_records], dtype=np.int64
    )
    train_dataset = base.make_dataset(train_records, args.downsample_size)
    test_dataset = base.make_dataset(test_records, args.test_downsample_size)
    if args.cache_data:
        fabric.print("Pre-caching datasets into RAM (single process; downsample frozen; "
                     "extra local epochs become ~free)...")
        train_dataset = _precache_dataset(train_dataset)
        test_dataset = _precache_dataset(test_dataset)
        # keep --num-workers: the cache removes the per-sample downsample, but collation
        # still costs, and serving it single-process just moves the bottleneck. Workers
        # fork-share the cached tensors (~1GB), so the copies are cheap.
        fabric.print("Pre-cache done.")
    datamodule = base.PointCloudDataModule(base.MEAN_STD_DICT)

    fabric.print(
        f"Loaded DrivAerNet++: {len(train_records)} train, "
        f"{len(test_records)} test. Methods: {args.methods}.",
    )

    initial_model = base.make_model()
    initial_state = base.clone_state_cpu(initial_model.state_dict())
    del initial_model
    output_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["cfd_root"] = str(CFD_ROOT)
    config["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    config["train_cases"] = len(train_records)
    config["test_cases"] = len(test_records)
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    all_history = []
    distribution_rows = []
    for alpha in args.alphas:
        clients = base.build_clients(args, train_classes, alpha)
        base.validate_client_purity(clients, train_classes, args.client_split_mode)
        counts = base.client_distribution(clients, train_classes)
        stats = base.distribution_stats(counts)
        stats.update(
            {
                "alpha": alpha,
                "client_split_mode": args.client_split_mode,
                "single_class_allocation": args.single_class_allocation,
            }
        )
        distribution_rows.append(stats)
        np.savetxt(
            output_dir / f"client_class_counts_alpha_{alpha:g}.csv",
            counts,
            delimiter=",",
            fmt="%d",
            header="estateback,notchback,fastback",
            comments="",
        )
        print(f"alpha={alpha:g} distribution: {stats}", flush=True)

        for method in args.methods:
            base.set_seed(args.seed)
            if method == "fedavg":
                rows = run_fedavg_server_lr(
                    initial_state,
                    train_dataset,
                    test_dataset,
                    test_records,
                    clients,
                    args,
                    repo_cfg,
                    fabric,
                    datamodule,
                    alpha,
                )
            else:
                rows = run_corrected(
                    method,
                    initial_state,
                    train_dataset,
                    test_dataset,
                    test_records,
                    clients,
                    args,
                    repo_cfg,
                    fabric,
                    datamodule,
                    alpha,
                )
            all_history.extend(rows)
            base.write_csv(
                output_dir / f"history_{method}_alpha{alpha:g}.csv", rows
            )

    base.write_csv(output_dir / "history.csv", all_history)
    base.write_csv(output_dir / "summary.csv", base.final_summary(all_history, args.rounds))
    base.write_csv(output_dir / "distribution_summary.csv", distribution_rows)
    print(f"Wrote results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()

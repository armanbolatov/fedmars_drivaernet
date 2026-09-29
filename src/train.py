"""Federated training on DrivAerNet++ with the cfd_transformer model.

The data pipeline, client partitioning, model and metrics come from
``ppbc_drivaernetpp_transformer.py`` in the cfd_transformer repo (imported, never
modified; set ``CFD_ROOT`` if it is not in ``~``). One trainer covers every method:

- ``fedavg``        AdamW clients, weight averaging.
- ``scaffold_adam`` control variates (SCAFFOLD) on the momentum, Adam-normalised step.
- ``fedmuon``       Newton-Schulz orthogonalised corrected momentum on matrix params
                    (FedMuon); ``--mars-gamma > 0`` adds the MARS correction (FedMARS-M).
- ``fedolion``      FedMuon with the orthogonalised direction passed through a sign.
- ``fedlion``       Lion clients; the server averages weights and momentum.
- ``scaffold``      plain SGD with control variates (equivalence tests only).

Per round: sample clients (``--clients-per-round``, default all), train each from the
global state, aggregate, evaluate on the cadence (``--eval-every``, ``--eval-tail``),
checkpoint. A run restarted into the same directory resumes from its checkpoint and
reproduces the uninterrupted run; skipped evaluations reproduce the RNG draws an
evaluation makes, so the cadence never changes the trajectory.
"""

import argparse
import json
import math
import random
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
import models


# --- Muon update -------------------------------------------------------------

MUON_EXCLUDE_KEYWORDS = ("embed",)
MUON_EXTRA_EXCLUDES: list[str] = []      # filled from --muon-exclude
ADAM_EPS = 1e-8


def is_muon_param(name: str, param: torch.Tensor) -> bool:
    """Matrix parameters get the Muon update; excluded names and vectors get Adam."""
    keywords = MUON_EXCLUDE_KEYWORDS + tuple(MUON_EXTRA_EXCLUDES)
    return param.ndim >= 2 and not any(k in name for k in keywords)


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Newton-Schulz approximation of the orthogonalisation of G (steps=0: G divided
    by its Frobenius norm)."""
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
    """scale_const * sqrt(max(d1, d2)) * NS(M): the standard Muon step ("shape"; 0.2
    matches AdamW's update RMS as in Moonlight, 1.0 is the original scale). "rms"
    additionally multiplies by rms(M) so the step shrinks as the momentum does."""
    M = corrected_momentum
    if M.ndim > 2:  # conv-style filters: flatten trailing dims
        M = M.view(len(M), -1)
    update = zeropower_via_newtonschulz5(M, steps=ns_steps)
    update = update * (scale_const * max(update.size(-2), update.size(-1)) ** 0.5)
    if scale_mode == "rms":
        rms = M.norm() / (M.numel() ** 0.5 + 1e-12)
        update = update * rms
    return update


# --- Local training ------------------------------------------------------------


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
    """One client's local pass for every method except fedavg.

    Returns (pseudo_grad, mean loss, momentum, v, last_grad, new control variate),
    all on the CPU."""
    model = models.make_model(args)
    model.load_state_dict(base_state)
    params = dict(model.named_parameters())   # names before Fabric prefixes them
    model = fabric.setup_module(model)

    micro = max(args.micro_batches, 1)        # gradient accumulation: same batch,
    loader = DataLoader(                      # peak memory divided by `micro`
        Subset(dataset, indices),
        batch_size=args.batch_size // micro,
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
        method in ("fedmuon", "fedolion") and args.cv_rule == "mean"
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
            if batch_idx % micro == 0:
                for p in params.values():
                    p.grad = None
            outputs = model(inputs)
            loss, _ = loss_wrapper(inputs, outputs, targets, others)
            total_loss = total_loss + loss.detach().double()   # no per-step GPU sync
            total_batches += 1
            fabric.backward(loss / micro if micro > 1 else loss)
            if (batch_idx + 1) % micro != 0:
                continue          # keep accumulating; step on the group boundary
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    params.values(), max_norm=args.grad_clip, norm_type=2
                )
            with torch.no_grad():
                for name, p in params.items():
                    grad = p.grad
                    if grad is None:
                        continue
                    if method == "fedlion":   # sign of the interpolated momentum,
                        m = momentum[name]    # no control variates
                        upd = torch.sign(beta * m + (1.0 - beta) * grad)
                        m.mul_(args.lion_beta2).add_(grad, alpha=1.0 - args.lion_beta2)
                        if args.weight_decay > 0:
                            p.mul_(1.0 - args.lion_lr_now * args.weight_decay)
                        p.add_(upd, alpha=-args.lion_lr_now)
                        continue
                    use_muon = (
                        method in ("fedmuon", "fedolion")
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
                        if args.muon_mode == "identity":   # raw corrected momentum
                            p.add_(corrected, alpha=-args.muon_lr_now)
                        else:
                            if args.muon_wd > 0:           # decoupled weight decay
                                p.mul_(1.0 - args.muon_lr_now * args.muon_wd)
                            update = muon_update(
                                corrected,
                                args.ns_steps,
                                args.muon_scale,
                                args.muon_scale_const,
                            )
                            if method == "fedolion":       # sign of the direction
                                update = torch.sign(update) * args.muon_scale_const
                            p.add_(update.reshape(p.shape), alpha=-args.muon_lr_now)
                    elif method == "scaffold":
                        p.add_(corrected, alpha=-args.scaffold_lr_now)
                    else:   # Adam-normalised corrected momentum (scaffold_adam;
                        vb = v_buf[name]                   # non-matrix params of Muon)
                        vb.mul_(args.fed_beta2).addcmul_(
                            corrected, corrected, value=1.0 - args.fed_beta2
                        )
                        if method in ("fedmuon", "fedolion"):
                            lr = args.scalar_lr_now
                        else:  # scaffold_adam: matrix params use scaffold_lr, rest scalar_lr
                            lr = (args.scaffold_lr_now if is_muon_param(name, p)
                                  else args.scalar_lr_now)
                        if args.adam_wd > 0:
                            p.mul_(1.0 - lr * args.adam_wd)
                        p.addcdiv_(corrected, vb.sqrt().add_(ADAM_EPS), value=-lr)
                    if grad_accum is not None:
                        grad_accum[name].add_(m)
            num_steps += 1

    if use_mean_cv:   # c_i <- mean momentum over the round (SCAFFOLD rule)
        new_local_c = {
            k: (t / max(num_steps, 1)).detach().cpu() for k, t in grad_accum.items()
        }
    else:             # c_i <- final local momentum (FedMuon paper rule)
        new_local_c = {k: t.detach().cpu().clone() for k, t in momentum.items()}
    new_momentum = {k: t.detach().cpu() for k, t in momentum.items()}
    new_v = {k: t.detach().cpu() for k, t in v_buf.items()}
    new_last_grad = {k: t.detach().cpu() for k, t in last_grad.items()}

    local_state = model.state_dict()
    pseudo_grad = {}
    for key, base_value in base_state.items():
        if torch.is_floating_point(base_value):
            pseudo_grad[key] = local_state[key].detach().cpu() - base_value
    del model   # cached GPU memory is kept: released memory on a shared card is
    return (    # taken by other jobs, which then starves the next client
        pseudo_grad,
        float(total_loss) / max(total_batches, 1),
        new_momentum,
        new_v,
        new_last_grad,
        new_local_c,
    )


# --- Round bookkeeping: statistics, checkpoints, schedules ----------------------

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
    """One row per communication round, independent of the evaluation cadence.
    client_drift = sqrt(sum_i w_i ||d_i||^2 - ||mean d||^2) (the weighted-variance
    identity, so no client update has to be kept); drift_ratio divides it by the
    aggregate step."""
    agg_norm = math.sqrt(max(state_sq_norm(agg_update), 0.0))
    drift = math.sqrt(max(client_sq_sum - agg_norm ** 2, 0.0))
    row = (method, f"{alpha:g}", round_idx, f"{train_loss:.6e}", f"{agg_norm:.6e}",
           f"{client_norm_mean:.6e}", f"{drift:.6e}",
           f"{(drift / agg_norm if agg_norm > 0 else float('nan')):.6e}",
           f"{getattr(args, 'muon_lr_now', float('nan')):.6e}", f"{seconds:.1f}",
           f"{torch.cuda.max_memory_reserved() / 2**30:.1f}")
    path = Path(args.output_dir) / "round_stats.csv"
    with path.open("w" if round_idx <= 1 else "a") as f:   # round 1 starts the file
        if f.tell() == 0:
            f.write(",".join(STAT_COLS) + "\n")
        f.write(",".join(str(x) for x in row) + "\n")


def checkpoint_path(args, method, alpha):
    return Path(args.output_dir) / f"checkpoint_{method}_alpha{alpha:g}.pt"


def save_checkpoint(path, round_idx, history, **buffers):
    """Everything a resumed run needs to continue exactly as an uninterrupted one."""
    rng = {"python": random.getstate(), "numpy": np.random.get_state(),
           "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}
    tmp = path.with_suffix(".tmp")
    torch.save({"round": round_idx, "history": history, "rng": rng, **buffers}, tmp)
    tmp.replace(path)


def load_checkpoint(path, args):
    """Restore the run's checkpoint if there is one; None means a fresh start."""
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    random.setstate(ckpt["rng"]["python"])
    np.random.set_state(ckpt["rng"]["numpy"])
    torch.set_rng_state(ckpt["rng"]["torch"])
    torch.cuda.set_rng_state_all(ckpt["rng"]["cuda"])
    stats = Path(args.output_dir) / "round_stats.csv"
    if stats.exists():        # drop rounds logged after the checkpoint was written
        lines = stats.read_text().splitlines(keepends=True)
        kept = [l for l in lines[1:] if int(l.split(",")[2]) <= ckpt["round"]]
        stats.write_text("".join(lines[:1] + kept))
    print(f"[RESUME] {path.name}: continuing after round {ckpt['round']}", flush=True)
    return ckpt


def cosine(lr, round_idx, args):
    """Cosine decay from lr at round 1 to zero at the last round."""
    return lr * 0.5 * (1.0 + math.cos(math.pi * (round_idx - 1) / max(args.rounds, 1)))


def set_round_lrs(args, round_idx):
    """This round's learning rates: muon-lr under --muon-lr-decay, the AdamW and Lion
    rates under --lr-decay, constant otherwise."""
    args.muon_lr_now = (cosine(args.muon_lr, round_idx, args)
                        if args.muon_lr_decay == "cosine" else args.muon_lr)
    for name in ("lr", "scalar_lr", "scaffold_lr", "lion_lr"):
        value = getattr(args, name)
        setattr(args, f"{name}_now",
                cosine(value, round_idx, args) if args.lr_decay == "cosine" else value)


def should_evaluate(round_idx, args):
    """The configured cadence, plus every round of the reporting tail."""
    return base.should_evaluate(round_idx, args) or round_idx > args.rounds - args.eval_tail


def skip_evaluation(args):
    """Make the RNG draws an evaluation would make (model construction, loader seed),
    so a sparser evaluation cadence leaves the training trajectory unchanged."""
    models.make_model(args)
    torch.empty((), dtype=torch.int64).random_()


# --- Federated rounds --------------------------------------------------------------


def round_participants(active, args, round_idx):
    """The clients that train this round: all of them, or a uniform sample of
    --clients-per-round without replacement, deterministic in seed and round."""
    k = args.clients_per_round
    if k <= 0 or k >= len(active):
        return active
    chosen = sorted(random.Random(args.seed * 1_000_003 + round_idx).sample(active, k))
    print(f"[ROUND {round_idx}] clients {chosen}", flush=True)
    return chosen


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
    """The federated loop for every method except fedavg: per-client momentum,
    second moments, last gradients and control variates persist across rounds."""
    state = base.clone_state_cpu(initial_state)
    active_clients = [client_id for client_id, indices in enumerate(clients) if indices]
    total_samples = sum(len(clients[client_id]) for client_id in active_clients)

    template = models.make_model(args)
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
            f"[FEDMUON] mode={args.muon_mode}, "
            f"scale={args.muon_scale}x{args.muon_scale_const:g}, "
            f"muon_wd={args.muon_wd:g}, mars_gamma={args.mars_gamma:g}, "
            f"mars_clip={args.mars_clip:g}, ema={args.ema_decay:g}, "
            f"cv_rule={args.cv_rule}, lr_decay={args.muon_lr_decay}, "
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

    ckpt_path = checkpoint_path(args, method, alpha)
    ckpt = load_checkpoint(ckpt_path, args)
    start, history = 1, []
    if ckpt:
        state, momentum, v_moments, last_grads, local_c, global_c, ema_state = (
            ckpt[k] for k in ("state", "momentum", "v_moments", "last_grads",
                              "local_c", "global_c", "ema_state"))
        start, history = ckpt["round"] + 1, ckpt["history"]
    elif not args.skip_round0_eval:
        history = eval_ext.evaluate_state(
            state, test_dataset, test_records, args, repo_cfg, fabric, datamodule,
            method, alpha, 0)

    for round_idx in range(start, args.rounds + 1):
        set_round_lrs(args, round_idx)
        aggregate_update = base.zero_state_like(state)
        losses = []
        new_local_cs = {}
        round_t0 = time.time()
        client_sq_sum, client_norms = 0.0, []
        participants = round_participants(active_clients, args, round_idx)
        round_samples = sum(len(clients[cid]) for cid in participants)
        for client_id in participants:
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
                client_id, participants, clients, round_samples, args.aggregation
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
        if method == "fedlion":   # the server also averages momentum, and every
            mean_m = {k: torch.zeros_like(t) for k, t in momentum[active_clients[0]].items()}
            for cid in participants:                     # client receives the mean
                w = base.client_weight(cid, participants, clients, round_samples,
                                       args.aggregation)
                for k in mean_m:
                    mean_m[k].add_(momentum[cid][k], alpha=w)
            for cid in active_clients:
                momentum[cid] = {k: t.clone() for k, t in mean_m.items()}
        for client_id in participants:   # absent clients keep their control variate
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
        if should_evaluate(round_idx, args):
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
        else:
            skip_evaluation(args)
        save_checkpoint(ckpt_path, round_idx, history, state=state, momentum=momentum,
                        v_moments=v_moments, last_grads=last_grads, local_c=local_c,
                        global_c=global_c, ema_state=ema_state)
    torch.save({"state": ema_state if ema_state is not None else state,
                "method": method, "args": vars(args)},
               Path(args.output_dir) / f"model_{method}.pt")
    ckpt_path.unlink(missing_ok=True)
    return history


def train_client_fedavg(base_state, dataset, indices, args, repo_cfg, fabric):
    """One FedAvg client: AdamW with the same matrix/non-matrix parameter groups as
    the other methods (--lr and --scalar-lr), beta1 = --fed-beta."""
    model = models.make_model(args)
    model.load_state_dict(base_state)
    matrix = [p for n, p in model.named_parameters() if is_muon_param(n, p)]
    rest = [p for n, p in model.named_parameters() if not is_muon_param(n, p)]
    optimizer = torch.optim.AdamW(
        [{"params": matrix, "lr": args.lr_now},
         {"params": rest, "lr": args.scalar_lr_now}],
        betas=(args.fed_beta, 0.999),
        weight_decay=args.weight_decay,
    )
    model, optimizer = fabric.setup(model, optimizer)
    micro = max(args.micro_batches, 1)
    loader = DataLoader(
        torch.utils.data.Subset(dataset, indices),
        batch_size=args.batch_size // micro,
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
            if batch_idx % micro == 0:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss, _ = loss_wrapper(inputs, outputs, targets, others)
            total_loss = total_loss + loss.detach().double()   # no per-step GPU sync
            total_batches += 1
            fabric.backward(loss / micro if micro > 1 else loss)
            if (batch_idx + 1) % micro != 0:
                continue
            if args.grad_clip > 0:
                fabric.clip_gradients(model, optimizer, max_norm=args.grad_clip, norm_type=2)
            optimizer.step()
    local_state = model.state_dict()
    grad = {}
    for key, base_value in base_state.items():
        if torch.is_floating_point(base_value):
            grad[key] = local_state[key].detach().cpu() - base_value
    del model
    return grad, float(total_loss) / max(total_batches, 1)


def run_fedavg_server_lr(initial_state, train_dataset, test_dataset, test_records,
                         clients, args, repo_cfg, fabric, datamodule, alpha):
    """The FedAvg loop, with the same server learning rate, participation sampling,
    checkpointing and evaluation cadence as run_corrected."""
    state = base.clone_state_cpu(initial_state)
    active = [cid for cid, idx in enumerate(clients) if idx]
    ckpt_path = checkpoint_path(args, "fedavg", alpha)
    ckpt = load_checkpoint(ckpt_path, args)
    start, history = 1, []
    if ckpt:
        state, start, history = ckpt["state"], ckpt["round"] + 1, ckpt["history"]
    elif not args.skip_round0_eval:
        history = eval_ext.evaluate_state(state, test_dataset, test_records, args,
                                          repo_cfg, fabric, datamodule, "fedavg",
                                          alpha, 0)
    for round_idx in range(start, args.rounds + 1):
        set_round_lrs(args, round_idx)
        aggregate_update = base.zero_state_like(state)
        losses = []
        round_t0 = time.time()
        client_sq_sum, client_norms = 0.0, []
        participants = round_participants(active, args, round_idx)
        round_total = sum(len(clients[cid]) for cid in participants)
        for cid in participants:
            grad, loss = train_client_fedavg(state, train_dataset, clients[cid], args,
                                             repo_cfg, fabric)
            weight = base.client_weight(cid, participants, clients, round_total,
                                        args.aggregation)
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
        if should_evaluate(round_idx, args):
            rows = eval_ext.evaluate_state(state, test_dataset, test_records, args, repo_cfg,
                                       fabric, datamodule, "fedavg", alpha, round_idx,
                                       train_loss)
            history.extend(rows)
            eval_ext.print_round(rows[0], args.rounds)
        else:
            skip_evaluation(args)
        save_checkpoint(ckpt_path, round_idx, history, state=state)
    torch.save({"state": state, "method": "fedavg", "args": vars(args)},
               Path(args.output_dir) / "model_fedavg.pt")
    ckpt_path.unlink(missing_ok=True)
    return history


# --- Entry point -------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--data-root", default=str(CFD_ROOT / "dataset" / "DrivAerNetPP"))
    p.add_argument("--output-dir", default="log/fedmuon_scaffold_drivaernetpp")
    p.add_argument("--model", default=models.DEFAULT, help="configs/model/<name>.yaml")
    p.add_argument("--methods", nargs="+", default=["fedavg", "scaffold_adam", "fedmuon"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--precision", default="bf16", choices=["fp32", "bf16"])
    p.add_argument("--num-workers", type=int, default=16)
    # clients
    p.add_argument("--num-clients", type=int, default=6)
    p.add_argument("--client-split-mode", default="single_class",
                   choices=["dirichlet", "single_class"])
    p.add_argument("--alphas", nargs="+", type=float, default=[0.1],
                   help="Dirichlet concentration(s); one run per value")
    p.add_argument("--single-class-allocation", default="proportional",
                   choices=["equal_classes", "proportional"])
    p.add_argument("--clients-per-round", type=int, default=0,
                   help="clients sampled per round without replacement; 0 = all")
    p.add_argument("--aggregation", default="equal", choices=["equal", "sample"])
    p.add_argument("--server-lr", type=float, default=1.0,
                   help="scales the aggregated update; 1.0 = plain averaging")
    # rounds and batches
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--micro-batches", type=int, default=1,
                   help="gradient-accumulation steps per batch (memory / N)")
    p.add_argument("--grad-clip", type=float, default=0.0)
    p.add_argument("--max-client-batches", type=int, default=0, help="0 = all")
    # learning rates and schedules
    p.add_argument("--lr", type=float, default=1.0e-4, help="fedavg: matrix params")
    p.add_argument("--scalar-lr", type=float, default=3.0e-4,
                   help="non-matrix params (fedavg, scaffold_adam, Muon family)")
    p.add_argument("--scaffold-lr", type=float, default=3.0e-4,
                   help="scaffold / scaffold_adam: matrix params")
    p.add_argument("--muon-lr", type=float, default=0.003, help="Muon family: matrix params")
    p.add_argument("--lion-lr", type=float, default=1e-4)
    p.add_argument("--lr-decay", default="none", choices=["none", "cosine"],
                   help="per-round decay of lr, scalar-lr, scaffold-lr and lion-lr")
    p.add_argument("--muon-lr-decay", default="none", choices=["none", "cosine"],
                   help="per-round decay of muon-lr (constant-norm updates need it)")
    p.add_argument("--weight-decay", type=float, default=0.01, help="fedavg and fedlion")
    p.add_argument("--adam-wd", type=float, default=0.0, help="decoupled, Adam branch")
    p.add_argument("--muon-wd", type=float, default=0.0, help="decoupled, Muon branch")
    # momentum
    p.add_argument("--fed-beta", type=float, default=0.9, help="momentum / Adam beta1")
    p.add_argument("--fed-beta2", type=float, default=0.95, help="Adam-branch beta2")
    p.add_argument("--lion-beta2", type=float, default=0.99)
    p.add_argument("--cv-rule", default="final", choices=["final", "mean"],
                   help="Muon family control variate: last momentum, or the round mean")
    p.add_argument("--ema-decay", type=float, default=0.0,
                   help="evaluate an EMA of the global model instead (0 = off)")
    # Muon
    p.add_argument("--ns-steps", type=int, default=5, help="Newton-Schulz iterations")
    p.add_argument("--muon-mode", default="lmo", choices=["lmo", "identity", "off"],
                   help="identity / off: equivalence tests without the orthogonalisation")
    p.add_argument("--muon-scale", default="shape", choices=["shape", "rms"])
    p.add_argument("--muon-scale-const", type=float, default=1.0,
                   help="0.2 matches AdamW's update RMS (Moonlight)")
    p.add_argument("--muon-exclude", default="",
                   help="comma-separated name substrings kept out of the Muon group "
                        "(default: the model yaml's list)")
    p.add_argument("--mars-gamma", type=float, default=0.0,
                   help="MARS correction g + gamma*beta/(1-beta)*(g - g_prev); 0 = FedMuon")
    p.add_argument("--mars-clip", type=float, default=1.0,
                   help="norm the corrected gradient is clipped to (0 = no clip)")
    # data and evaluation
    p.add_argument("--downsample-size", type=int, default=50000, help="points per car")
    p.add_argument("--test-downsample-size", type=int, default=-1, help="-1 = all")
    p.add_argument("--max-train-samples", type=int, default=0, help="0 = all")
    p.add_argument("--max-test-samples", type=int, default=0, help="0 = all")
    p.add_argument("--pressure-weight", type=float, default=1.0)
    p.add_argument("--wss-weight", type=float, default=1.0)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--eval-tail", type=int, default=0,
                   help="also evaluate every one of the last N rounds")
    p.add_argument("--skip-round0-eval", action="store_true",
                   help="skip evaluating the untrained model")
    p.add_argument("--cache-data", action="store_true",
                   help="materialise the downsampled dataset in RAM once (freezes the "
                        "random downsample; not used by the protocol)")
    return p.parse_args()


class _ListDataset(torch.utils.data.Dataset):
    """Items materialised in RAM by --cache-data."""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def _precache_dataset(dataset):
    """Load every item once, single-process (workers forked after CUDA init deadlock)."""
    n = len(dataset)
    items = [None] * n
    for i in range(n):
        items[i] = dataset[i]
        if (i + 1) % 2000 == 0:
            print(f"  pre-cached {i + 1}/{n}", flush=True)
    return _ListDataset(items)


def main() -> None:
    args = parse_args()
    valid_methods = {"fedavg", "scaffold", "scaffold_adam", "fedmuon",
                     "fedlion", "fedolion"}
    if not set(args.methods).issubset(valid_methods):
        raise ValueError(f"--methods must be a subset of {sorted(valid_methods)}")
    if args.eval_batch_size != 1:
        raise ValueError("--eval-batch-size must be 1 (repository metric semantics).")
    MUON_EXTRA_EXCLUDES.extend(models.muon_excludes(args))
    set_round_lrs(args, 1)

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
        fabric.print("Pre-cache done.")
    datamodule = base.PointCloudDataModule(base.MEAN_STD_DICT)

    fabric.print(
        f"Loaded DrivAerNet++: {len(train_records)} train, "
        f"{len(test_records)} test. Methods: {args.methods}.",
    )

    initial_model = models.make_model(args)
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

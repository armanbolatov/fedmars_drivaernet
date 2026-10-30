"""Gradient spectra of the Transolver's weight matrices, per client.

For each model (the seed-42 initialisation, or a saved model or checkpoint) and each of
the six single-class clients, the gradient of every Muon-updated matrix over the
client's first cars. Per matrix and model:
  stable rank     ||G||_F^2 / s_max^2 of the mean gradient, and its median over clients
  gain            ||G||_* s_max / ||G||_F^2: first-order work of an orthogonalised step
                  over a gradient step of the same spectral size
  sign size       ||sign G||_2 sqrt(min(m,n) / (mn)): spectral size of a sign step over
                  an orthogonalised step of the same Frobenius size
  client share    mean_i ||G_i - G||_F / ||G||_F, G the mean over clients

  python scripts/grad_spectrum.py --models init path/to/model_fedavg.pt ...

Writes results/tables/grad_spectrum.csv. A diagnostic for the paper's section on local
steps, not part of the protocol.
"""
import argparse
import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", default=["init"])
ap.add_argument("--cars", type=int, default=16, help="cars per client")
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--points", type=int, default=50000, help="points per car")
ap.add_argument("--out", default=str(ROOT / "results" / "tables" / "grad_spectrum.csv"))
a = ap.parse_args()
sys.argv = ["grad_spectrum", "--output-dir", str(ROOT / "log" / "gradspec"),
            "--downsample-size", str(a.points), "--methods", "fedmuon", "--rounds", "1",
            "--seed", "42", "--skip-round0-eval", "--client-split-mode", "single_class"]
import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402
from torch.utils.data import DataLoader, Subset                 # noqa: E402
import train, models                                            # noqa: E402

base = train.base
args = train.parse_args()
train.MUON_EXTRA_EXCLUDES.extend(models.muon_excludes(args))
repo_cfg = base.load_repo_cfg(args)
fabric = base.make_fabric(args)
fabric.launch()
base.seed_everything(repo_cfg, fabric, workers=True)
base.set_seed(args.seed)
init_state = base.clone_state_cpu(models.make_model(args).state_dict())  # as in the runs
root = Path(args.data_root)
records = base.read_records(root / "split" / "train_split.json", root, 0, args.seed)
classes = np.array([r.class_id for r in records], dtype=np.int64)
clients = [idx for idx in base.build_clients(args, classes, args.alphas[0]) if len(idx)]
dataset = base.make_dataset(records, args.downsample_size)
loss_wrapper = base.make_loss_wrapper(args, repo_cfg)


def load(path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return next(v for v in blob.values() if isinstance(v, dict) and set(init_state) <= set(v))


def gradients(state, idx):
    """Mean gradient of every Muon-updated matrix over the client's cars."""
    model = models.make_model(args)
    model.load_state_dict(state)
    params = dict(model.named_parameters())
    model = fabric.setup_module(model)
    model.train()
    loader = fabric.setup_dataloaders(DataLoader(Subset(dataset, list(idx)),
                                                 batch_size=a.batch, num_workers=4))
    for inputs, targets, others in loader:
        loss, _ = loss_wrapper(inputs, model(inputs), targets, others)
        fabric.backward(loss / len(loader))
    return {n: p.grad.detach().float().reshape(p.shape[0], -1).cpu()
            for n, p in params.items() if p.grad is not None and train.is_muon_param(n, p)}


def spectrum(G):
    s = torch.linalg.svdvals(G)
    fro2, top = (s ** 2).sum(), s[0]
    m, n = G.shape
    sign = torch.linalg.matrix_norm(torch.sign(G), ord=2) * (min(m, n) / (m * n)) ** 0.5
    return (fro2 / top ** 2).item(), (s.sum() * top / fro2).item(), sign.item()


header = ["model", "matrix", "rows", "cols", "stable_rank", "stable_rank_client_median",
          "gain", "sign_size", "client_share"]
rows = []
for spec in a.models:
    state = init_state if spec == "init" else load(spec)
    per_client = [gradients(state, idx[:a.cars]) for idx in clients]
    label = spec if spec == "init" else Path(spec).parent.name
    for name in per_client[0]:
        Gs = [g[name] for g in per_client]
        G = sum(Gs) / len(Gs)
        sr, gain, sign = spectrum(G)
        sr_client = statistics.median(spectrum(Gi)[0] for Gi in Gs)
        share = statistics.mean((Gi - G).norm().item() for Gi in Gs) / G.norm().item()
        rows.append([label, name, *G.shape, sr, sr_client, gain, sign, share])
    mine = [r for r in rows if r[0] == label]
    print(f"{label}: {len(mine)} matrices, median stable rank "
          f"{statistics.median(r[4] for r in mine):.2f} (clients "
          f"{statistics.median(r[5] for r in mine):.2f}), gain "
          f"{statistics.median(r[6] for r in mine):.2f}, sign size "
          f"{statistics.median(r[7] for r in mine):.2f}, client share "
          f"{statistics.median(r[8] for r in mine):.2f}", flush=True)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
with open(a.out, "w", newline="") as f:
    csv.writer(f).writerows([header] + rows)
print("wrote", a.out)

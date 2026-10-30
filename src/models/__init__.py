"""Model registry: `configs/model/<name>.yaml` says how to build the network.

A model yaml has two keys:

  model:          a hydra node (`_target_` plus constructor arguments) for any
                  nn.Module whose forward takes the batch's input dict and returns
                  {"pressure": (B, N, 1), "wss": (B, N, 3)} in normalised units
  muon_exclude:   name substrings of layers the spectral methods keep on Adam
                  (embedding-like layers; `embed` is always excluded)

The trainer and the evaluation build the network through make_model, so
a new architecture is one yaml (and a class, if it does not exist yet) plus
`--model <name>`; nothing else in the pipeline knows which model runs.
"""
from pathlib import Path

import hydra
from omegaconf import OmegaConf

DEFAULT = "transolver"
CONFIGS = Path(__file__).resolve().parent.parent.parent / "configs" / "model"


def spec(args):
    return OmegaConf.load(CONFIGS / f"{getattr(args, 'model', DEFAULT)}.yaml")


def make_model(args):
    return hydra.utils.instantiate(spec(args).model)


def muon_excludes(args):
    """--muon-exclude if given, else the model yaml's list."""
    if getattr(args, "muon_exclude", ""):
        return [s for s in args.muon_exclude.split(",") if s]
    return list(spec(args).get("muon_exclude", []))

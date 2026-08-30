#!/usr/bin/env python
"""Count parameters of the TRM-diffusion model components for an experiment.

Instantiates each Hydra config group (backbone / thinker / translator /
condition_encoder) exactly as training does and reports total + trainable
parameter counts. No checkpoints are loaded -- this measures architecture size.

Run from examples/trm_diffusion/:
  python scripts/count_params.py experiment=amaze_thinker_v2_controlnet
  python scripts/count_params.py experiment=amaze_dit_maze
"""
from __future__ import annotations

import argparse
import os
import sys
import types

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
_CONFIGS = os.path.join(_ROOT, "configs")
sys.path.insert(0, _ROOT)  # make models/, configs/ importable for instantiate

GROUPS = ["backbone", "thinker", "translator", "condition_encoder"]

# Optional compiled backends only needed at train/forward time (not for building
# a module and counting its parameters). On a dev box without them, inject a
# permissive stub so instantiation still works. No-op where they are installed.
_OPTIONAL_BACKENDS = ("adam_atan2_backend", "flash_attn", "flash_attn_interface")


def _install_optional_stubs() -> None:
    class _Stub(types.ModuleType):
        def __getattr__(self, name):  # noqa: D401 -- attribute-less stub
            return None

    for name in _OPTIONAL_BACKENDS:
        try:
            __import__(name)
        except Exception:  # noqa: BLE001
            sys.modules[name] = _Stub(name)


def _counts(module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config-name", default="config")
    ap.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides, e.g. experiment=amaze_thinker_v2_controlnet",
    )
    args = ap.parse_args()

    _install_optional_stubs()

    with initialize_config_dir(version_base=None, config_dir=_CONFIGS):
        cfg = compose(config_name=args.config_name, overrides=list(args.overrides))

    print(f"{'component':<20}{'total':>14}{'trainable':>14}")
    print("-" * 48)
    grand_total = 0
    for group in GROUPS:
        node = cfg.get(group)
        if node is None or "_target_" not in node:
            continue
        try:
            module = instantiate(node)
        except Exception as exc:  # noqa: BLE001 -- report and keep going
            print(f"{group:<20}{'FAILED':>14}   {type(exc).__name__}: {exc}")
            continue
        total, trainable = _counts(module)
        grand_total += total
        print(f"{group:<20}{total:>14,}{trainable:>14,}")
    print("-" * 48)
    print(f"{'sum(above)':<20}{grand_total:>14,}")


if __name__ == "__main__":
    main()

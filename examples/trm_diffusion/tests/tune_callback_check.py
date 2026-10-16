"""Runs SteeringDiagnosticsCallback on a tiny-thinker MiniT2I model over a
built Std-120K fixture (argv[1]) and writes {"sharded", "single", "zero_init"}
metrics to argv[2]: sharded across all processes, on the main process alone,
and at the translator's zero initialization. Launched by
tests/test_tune_callbacks.py with `accelerate launch --num_processes 2`.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from accelerate import Accelerator
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from factory import build_model  # noqa: E402

TINY_THINKER = [
    "thinker.hidden_size=64", "thinker.n_heads=2", "thinker.vocab_size=16",
    "thinker.n_sup=2", "thinker.H_cycles=2", "thinker.L_cycles=1", "thinker.L_layers=1",
]


def main(data_root: str, out: str) -> None:
    acc = Accelerator()
    overrides = ["experiment=std120k_thinker_minit2i", f"data.root={data_root}", "~eval_callbacks",
                 "tune_callbacks=steering", "tune_callbacks.0.max_batches=4", "tune_callbacks.0.out_path=null",
                 "data.val_dataset.max_samples=8", *TINY_THINKER]
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    torch.manual_seed(0)  # the same thinker init on every rank (train_trm.py gets this from DDP's broadcast)
    model = build_model(cfg, instantiate(cfg.diffusion)).to(acc.device).eval()
    model.painter.painter_dtype = None  # fp32: sharded and single-process results comparable
    ds = instantiate(cfg.data.val_dataset)
    dl = DataLoader(ds, batch_size=2, shuffle=False, drop_last=True, collate_fn=getattr(type(ds), "collate_fn", None))
    [callback] = model.eval_callbacks
    single = SimpleNamespace(num_processes=1, process_index=0, is_main_process=True, device=acc.device,
                             gather=lambda t: t)

    results = {"zero_init": callback(model, dl, acc, step=0)}
    torch.manual_seed(0)
    with torch.no_grad():
        for proj in model.thinker_painter_translator.projs:
            proj.weight.normal_(std=1e-2)
    results["sharded"] = callback(model, dl, acc, step=1)
    if acc.is_main_process:
        results["single"] = callback(model, dl, single, step=1)
        Path(out).write_text(json.dumps(results, indent=2))
    acc.wait_for_everyone()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

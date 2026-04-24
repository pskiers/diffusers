"""
eval_painter.py — Standalone evaluation script for trained painter models.

Loads a checkpoint, runs sampling eval and realsolution eval (where applicable),
and prints all metrics. Optionally logs to wandb.

Usage:
    python eval_painter.py \\
        --checkpoint runs/standalone_painter/checkpoint_final.pt \\
        --mode standalone_painter \\
        --num_samples 1000 \\
        --batch_size 64 \\
        --sampler ddim --num_steps 20

    python eval_painter.py \\
        --checkpoint runs/trm_v0tok/checkpoint_final.pt \\
        --mode painter --painter_variant v0tok \\
        --num_samples 1000 --sampler ddpm --num_steps 100 \\
        --cfg_scale 1.5

    # Sweep CFG values:
    python eval_painter.py \\
        --checkpoint runs/standalone_painter/checkpoint_final.pt \\
        --mode standalone_painter \\
        --cfg_scale 1.0 2.0 4.0 --num_samples 512

Checkpoint format: {"model_state": ..., "ema_state": ...} as saved by train_trm.py.
EMA weights are used automatically when present (pass --no_ema to skip).
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# ── Local imports (same cwd as train_trm.py) ─────────────────────────────────
from mnist_sudoku_dataset import MNISTSudokuDataset
from mnist_eval import evaluate_grids, load_or_train_classifier, sample_grids, make_panel_image
from trm_wrappers import (
    OriginalTRMRatatouilleV0Tok,
    OriginalTRMRatatouilleV0,
    OriginalTRMRatatouilleV1,
    OriginalTRMRatatouilleV2,
    OriginalTRMRatatouilleV3,
    OriginalTRMRatatouilleV4,
    StandalonePainter,
    ThinkerWithFrozenPainter,
)
from models.ema import EMAHelper


# ── Condition helpers (mirrors train_trm.py) ──────────────────────────────────

def _solution_tokens(solution: torch.Tensor) -> torch.Tensor:
    """(B, 81) int [0-8] → (B, 81) long [2-10] token IDs."""
    return (solution + 2).long()


def _get_condition(mb: dict, model, device) -> torch.Tensor:
    if isinstance(model, StandalonePainter):
        return _solution_tokens(mb["solution"].to(device))
    if model.token_input:
        return mb["puzzle_tokens"].to(device)
    return mb["conditions"].to(device)


def _get_full_solution_condition(mb: dict, model, device="cpu") -> torch.Tensor:
    if isinstance(model, StandalonePainter) or model.token_input:
        return _solution_tokens(mb["solution"].to(device))
    return mb["images"].to(device)


# ── Model building ────────────────────────────────────────────────────────────

def build_model(args) -> torch.nn.Module:
    """Instantiate the model skeleton from CLI args (no weights loaded yet)."""
    t = args  # shorthand — we use the same field names as the thinker config

    thinker_kwargs = dict(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        n_heads=args.n_heads,
        L_layers=args.L_layers,
        L_cycles=args.L_cycles,
        H_cycles=args.H_cycles,
        n_sup=args.n_sup,
        expansion=args.expansion,
        forward_dtype=args.forward_dtype,
        mlp_t=args.mlp_t,
        pos_encodings=args.pos_encodings,
        puzzle_emb_ndim=args.puzzle_emb_ndim,
        puzzle_emb_len=args.puzzle_emb_len,
        num_puzzle_identifiers=args.num_puzzle_identifiers,
        halt_exploration_prob=0.0,
        batch_size=args.batch_size,
        freeze_weights=False,
    )

    painter_size = 9 * args.cell_size

    if args.mode == "standalone_painter":
        model = StandalonePainter(
            painter_size=painter_size,
            cell_size=args.cell_size,
            vocab_size=args.vocab_size,
            bridge_channels=args.bridge_channels,
            painter_channels=tuple(args.painter_channels),
            painter_layers_per_block=args.painter_layers_per_block,
            cfg_prob=0.0,
            cfg_scale=1.0,
            painter_dtype=args.painter_dtype,
        )
    elif args.painter_variant == "v0tok":
        model = OriginalTRMRatatouilleV0Tok(
            painter_size=painter_size,
            cell_size=args.cell_size,
            bridge_channels=args.bridge_channels,
            painter_channels=tuple(args.painter_channels),
            painter_layers_per_block=args.painter_layers_per_block,
            diff_thinker_weight=1.0,
            thinker_bridge_mode=args.thinker_bridge_mode,
            painter_dtype=args.painter_dtype,
            **thinker_kwargs,
        )
    else:
        img_thinker_kwargs = {k: v for k, v in thinker_kwargs.items()
                              if k not in ("vocab_size", "puzzle_emb_ndim", "puzzle_emb_len",
                                           "num_puzzle_identifiers", "halt_exploration_prob",
                                           "batch_size", "freeze_weights")}
        img_painter_kwargs = dict(
            painter_size=painter_size,
            cell_size=args.cell_size,
            enc_channels=args.enc_channels,
            bridge_channels=args.bridge_channels,
            painter_channels=tuple(args.painter_channels),
            painter_layers_per_block=args.painter_layers_per_block,
            diff_thinker_weight=1.0,
            painter_dtype=args.painter_dtype,
        )
        _VARIANT_CLS = {
            "v0": OriginalTRMRatatouilleV0,
            "v1": OriginalTRMRatatouilleV1,
            "v2": OriginalTRMRatatouilleV2,
            "v3": OriginalTRMRatatouilleV3,
            "v4": OriginalTRMRatatouilleV4,
        }
        cls = _VARIANT_CLS.get(args.painter_variant)
        if cls is None:
            raise ValueError(f"Unknown painter_variant: {args.painter_variant!r}")
        if args.painter_variant in ("v2", "v3", "v4"):
            extra = dict(thinker_out_channels=args.thinker_out_channels or 16)
            if args.painter_variant == "v4":
                extra["compression_factor"] = args.cell_size
                extra["bridge_num_heads"]   = 4
                img_thinker_kwargs = {k: v for k, v in img_thinker_kwargs.items() if k != "seq_len"}
            model = cls(**extra, **img_painter_kwargs, **img_thinker_kwargs)
        else:
            model = cls(
                num_classes=args.num_classes,
                thinker_out_channels=args.thinker_out_channels,
                enc_timestep_cond=False,
                thinker_timestep_cond=False,
                **img_painter_kwargs, **img_thinker_kwargs,
            )

    return model


def load_checkpoint(model: torch.nn.Module, path: str, use_ema: bool, device: torch.device):
    ckpt = torch.load(path, map_location="cpu")
    if use_ema and ckpt.get("ema_state") is not None:
        # EMAHelper.state_dict() returns self.shadow: {param_name: tensor}
        # EMAHelper.ema(model) copies shadow → model params.
        ema = EMAHelper(mu=0.999)
        ema.load_state_dict(ckpt["ema_state"])
        ema.ema(model)
        print(f"Loaded EMA weights from {path} (step={ckpt.get('step', '?')})")
    else:
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded model weights from {path} (step={ckpt.get('step', '?')})")
    model.to(device)
    return model


# ── Eval loop ─────────────────────────────────────────────────────────────────

def run_eval(model, eval_ds, args, device, cfg_scale: float):
    """Run one full eval pass with given cfg_scale. Returns metrics dict."""
    painter_size = 9 * args.cell_size
    n_total  = args.num_samples
    n_batch  = args.batch_size
    n_steps  = args.num_steps
    use_ddpm = args.sampler == "ddpm"

    # For DDPM sampling, num_steps equals num_train_timesteps (full chain).
    sample_steps = args.num_train_timesteps if use_ddpm else n_steps

    classifier = load_or_train_classifier(
        args.classifier_path,
        cell_size=args.cell_size,
        device=device,
    )

    # Apply cfg_scale to model
    if hasattr(model, "cfg_scale"):
        model.cfg_scale = cfg_scale

    model.eval()
    loader = DataLoader(eval_ds, batch_size=n_batch, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    all_cell_acc:   list[float] = []
    all_puzzle_acc: list[float] = []
    all_thinker_cell_best:   list[float] = []
    all_thinker_cell_mean:   list[float] = []
    all_thinker_puzzle_best: list[float] = []
    all_thinker_puzzle_mean: list[float] = []
    all_thinker_deviation:   list[float] = []
    all_painter_dev_best:    list[float] = []
    all_painter_dev_mean:    list[float] = []

    all_real_cell_acc:   list[float] = []
    all_real_puzzle_acc: list[float] = []
    do_real_eval = getattr(model, "has_realsolution_eval", False)

    n_done = 0
    token_offset = getattr(model, "token_offset", 0)

    for eb in tqdm(loader, desc=f"Sampling (cfg={cfg_scale})",
                   total=(n_total + n_batch - 1) // n_batch):
        if n_done >= n_total:
            break
        B_cur       = eb["solution"].shape[0]
        sols        = eb["solution"]
        pids        = eb.get("puzzle_id", None)
        given_masks = eb.get("given_mask", None)
        cond        = _get_condition(eb, model, device="cpu")

        with torch.no_grad():
            sr = sample_grids(
                model, cond,
                num_train_timesteps=args.num_train_timesteps,
                beta_schedule=args.beta_schedule,
                prediction_type=args.prediction_type,
                num_steps=sample_steps,
                device=device,
                puzzle_ids=pids,
                solutions=sols,
                painter_size=painter_size,
                given_masks=given_masks,
            )

        acc = evaluate_grids(sr["generated"], sols, classifier, args.cell_size,
                             given_masks=given_masks)
        all_cell_acc.append(acc["cell_acc"])
        all_puzzle_acc.append(acc["puzzle_acc"])

        for key, lst in [
            ("thinker_cell_acc_best",       all_thinker_cell_best),
            ("thinker_cell_acc_mean",       all_thinker_cell_mean),
            ("thinker_puzzle_acc_best",     all_thinker_puzzle_best),
            ("thinker_puzzle_acc_mean",     all_thinker_puzzle_mean),
            ("thinker_deviation_from_best", all_thinker_deviation),
        ]:
            if key in sr:
                lst.append(sr[key])

        painter_preds = acc["preds"]
        _gm = given_masks[:B_cur] if given_masks is not None else None
        for tp_raw, dev_lst in [
            (sr.get("best_thinker_preds"), all_painter_dev_best),
            (sr.get("mean_thinker_preds"), all_painter_dev_mean),
        ]:
            if tp_raw is not None:
                tp   = tp_raw - token_offset
                N    = tp.shape[1]
                diff = painter_preds[:, :N] != tp
                if _gm is not None:
                    blank = ~_gm[:, :N]
                    n_b   = blank.sum()
                    dev   = diff[blank].float().mean().item() if n_b > 0 else diff.float().mean().item()
                else:
                    dev = diff.float().mean().item()
                dev_lst.append(dev)

        # Realsolution eval for this batch
        if do_real_eval:
            full_cond = _get_full_solution_condition(eb, model, device="cpu")
            with torch.no_grad():
                sr_r = sample_grids(
                    model, full_cond,
                    num_train_timesteps=args.num_train_timesteps,
                    beta_schedule=args.beta_schedule,
                    prediction_type=args.prediction_type,
                    num_steps=sample_steps,
                    device=device,
                    puzzle_ids=pids,
                    solutions=sols,
                    painter_size=painter_size,
                    given_masks=given_masks,
                )
            acc_r = evaluate_grids(sr_r["generated"], sols, classifier, args.cell_size,
                                   given_masks=given_masks)
            all_real_cell_acc.append(acc_r["cell_acc"])
            all_real_puzzle_acc.append(acc_r["puzzle_acc"])

        n_done += B_cur

    metrics = {
        "cfg_scale":  cfg_scale,
        "sampler":    args.sampler,
        "num_steps":  sample_steps,
        "n_samples":  n_done,
        "cell_acc":   float(np.mean(all_cell_acc)),
        "puzzle_acc": float(np.mean(all_puzzle_acc)),
    }
    if all_real_cell_acc:
        metrics["realsolution_cell_acc"]   = float(np.mean(all_real_cell_acc))
        metrics["realsolution_puzzle_acc"] = float(np.mean(all_real_puzzle_acc))
    if all_thinker_cell_best:
        metrics["thinker_cell_acc_best"]       = float(np.mean(all_thinker_cell_best))
        metrics["thinker_cell_acc_mean"]       = float(np.mean(all_thinker_cell_mean))
        metrics["thinker_puzzle_acc_best"]     = float(np.mean(all_thinker_puzzle_best))
        metrics["thinker_puzzle_acc_mean"]     = float(np.mean(all_thinker_puzzle_mean))
        metrics["thinker_deviation_from_best"] = float(np.mean(all_thinker_deviation))
    if all_painter_dev_best:
        metrics["painter_dev_from_best_thinker"] = float(np.mean(all_painter_dev_best))
        metrics["painter_dev_from_mean_thinker"] = float(np.mean(all_painter_dev_mean))
    return metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained painter model.")

    # Required
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    p.add_argument("--classifier_path", default="runs/mnist_classifier_cell16.pt",
                   help="Path to (or where to save) the MNIST cell classifier")

    # Mode / variant
    p.add_argument("--mode", default="standalone_painter",
                   choices=["standalone_painter", "painter", "thinker_frozen_painter"],
                   help="Training mode of the checkpoint")
    p.add_argument("--painter_variant", default="v0tok",
                   choices=["v0tok", "v0", "v1", "v2", "v3", "v4"],
                   help="Painter variant (used when mode=painter)")

    # Data
    p.add_argument("--sudoku_dir",  default="data/sudoku-extreme-1k-aug-1000")
    p.add_argument("--mnist_root",  default="data/mnist")
    p.add_argument("--cell_size",   type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)

    # Diffusion scheduler
    p.add_argument("--num_train_timesteps", type=int, default=100)
    p.add_argument("--beta_schedule",       default="squaredcos_cap_v2")
    p.add_argument("--prediction_type",     default="sample",
                   choices=["sample", "epsilon"])

    # Sampling
    p.add_argument("--sampler",    default="ddim", choices=["ddim", "ddpm"],
                   help="ddim = DDIM with --num_steps; ddpm = full DDPM chain")
    p.add_argument("--num_steps",  type=int, default=20,
                   help="DDIM denoising steps (ignored for ddpm)")
    p.add_argument("--num_samples", type=int, default=512,
                   help="Total samples for eval")
    p.add_argument("--batch_size",  type=int, default=64)

    # CFG sweep
    p.add_argument("--cfg_scale", type=float, nargs="+", default=[1.0],
                   help="One or more CFG scale values to evaluate")

    # EMA
    p.add_argument("--no_ema", action="store_true",
                   help="Load raw model weights instead of EMA")

    # Device
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # ── Thinker / painter architecture (must match checkpoint) ───────────────
    p.add_argument("--vocab_size",  type=int, default=11)
    p.add_argument("--seq_len",     type=int, default=81)
    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--n_heads",     type=int, default=8)
    p.add_argument("--L_layers",    type=int, default=2)
    p.add_argument("--L_cycles",    type=int, default=6)
    p.add_argument("--H_cycles",    type=int, default=3)
    p.add_argument("--n_sup",       type=int, default=16)
    p.add_argument("--expansion",   type=float, default=4.0)
    p.add_argument("--forward_dtype", default="bfloat16")
    p.add_argument("--mlp_t",       action="store_true")
    p.add_argument("--pos_encodings", default="rope")
    p.add_argument("--puzzle_emb_ndim",         type=int, default=0)
    p.add_argument("--puzzle_emb_len",          type=int, default=16)
    p.add_argument("--num_puzzle_identifiers",  type=int, default=1000)
    p.add_argument("--num_classes",             type=int, default=9)
    p.add_argument("--thinker_out_channels",    type=int, default=None)
    p.add_argument("--enc_channels",            type=int, default=32)
    p.add_argument("--bridge_channels",         type=int, default=16)
    p.add_argument("--painter_channels",        type=int, nargs="+", default=[32, 64, 64])
    p.add_argument("--painter_layers_per_block", type=int, default=2)
    p.add_argument("--thinker_bridge_mode",     default="logits",
                   choices=["logits", "onehot", "softmax"])
    p.add_argument("--painter_dtype",           default="bfloat16",
                   choices=["bfloat16", "float16", "none"])

    # Wandb
    p.add_argument("--wandb_project", default=None,
                   help="Set to enable wandb logging")
    p.add_argument("--wandb_run_name", default=None)

    return p.parse_args()


def main():
    args = parse_args()

    if args.painter_dtype == "none":
        args.painter_dtype = None

    device = torch.device(args.device)

    # ── Dataset ───────────────────────────────────────────────────────────────
    test_dir = os.path.join(args.sudoku_dir, "test")
    eval_dir = test_dir if os.path.isdir(test_dir) else os.path.join(args.sudoku_dir, "train")
    eval_ds = MNISTSudokuDataset(
        sudoku_dir=eval_dir,
        mnist_root=args.mnist_root,
        cell_size=args.cell_size,
        mnist_split="test",
        mask_given=True,
    )
    print(f"Eval dataset: {len(eval_ds)} samples from {eval_dir}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(args)
    model = load_checkpoint(model, args.checkpoint, use_ema=not args.no_ema, device=device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {type(model).__name__}  total={n_params:,}  trainable={n_trainable:,}")

    # ── Wandb ─────────────────────────────────────────────────────────────────
    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
            )
        except ImportError:
            print("wandb not installed; skipping wandb logging")

    # ── CFG sweep ─────────────────────────────────────────────────────────────
    all_results = []
    for cfg_scale in args.cfg_scale:
        print(f"\n{'='*60}")
        print(f"cfg_scale={cfg_scale}  sampler={args.sampler}  steps={args.num_steps}")
        print(f"{'='*60}")
        with torch.no_grad():
            metrics = run_eval(model, eval_ds, args, device, cfg_scale=cfg_scale)
        all_results.append(metrics)

        # Print results
        print(f"\n── Results (cfg={cfg_scale}) ──")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k:45s} {v:.4f}")
            else:
                print(f"  {k:45s} {v}")

        if wandb_run is not None:
            wandb_run.log({f"eval/{k}": v for k, v in metrics.items()})

    # ── Summary table for sweeps ───────────────────────────────────────────────
    if len(args.cfg_scale) > 1:
        print(f"\n{'='*60}")
        print("CFG sweep summary:")
        header_keys = ["cfg_scale", "cell_acc", "puzzle_acc",
                       "realsolution_cell_acc", "realsolution_puzzle_acc"]
        cols = [k for k in header_keys if any(k in r for r in all_results)]
        print("  " + "  ".join(f"{c:>28}" for c in cols))
        for r in all_results:
            print("  " + "  ".join(
                f"{r[c]:>28.4f}" if isinstance(r.get(c), float) else f"{r.get(c, ''):>28}"
                for c in cols
            ))

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()

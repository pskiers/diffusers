"""
Sampling script for Bit-Diffusion Sokoban (standard and TRM).

Loads the best checkpoint, generates boards via lit_model.evaluate(),
prints metrics, and saves rendered images.

Usage (standard diffusion):
  python sample.py --config-name=standard_diffusion run_name=my_run checkpoint_path=best

Usage (TRM diffusion):
  python sample.py --config-name=trm_diffusion run_name=my_run checkpoint_path=best
"""
import json
import sys
from pathlib import Path
from typing import Optional

import hydra
import lightning as L
import numpy as np
import torch
import wandb
from omegaconf import DictConfig
from PIL import Image
from tqdm.auto import tqdm

from diffusers import DDPMScheduler, Transformer2DModel
from sokoban.bit_diffusion.train_std import SokobanBitDataModule, SokobanBitDiffusion
from sokoban.bit_diffusion.train_trm import SokobanTRMBitDiffusion, TRMDiT
from sokoban.bit_diffusion.train_trm_embedded import EmbeddedTRMDiffusion, SokobanEmbeddedTRMDiffusion


def find_best_checkpoint(output_dir: str, run_name: str, is_std: bool) -> str:
    """Find the latest best-*.ckpt. std saves to output_dir/run_name/std_checkpoints,
    trm and embedded save to output_dir/checkpoints."""
    base_dir = Path(output_dir)
    ckpt_dir = base_dir / run_name / "std_checkpoints" if is_std else base_dir / "checkpoints"

    candidates = list(ckpt_dir.glob("best-*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No best-*.ckpt found in {ckpt_dir}")

    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def _resolve_wandb_run_id(cfg: DictConfig, ckpt_path: str) -> Optional[str]:
    """Resolve the training run's W&B id so sampling logs onto the same experiment.

    Priority: explicit cfg.wandb_run_id, else wandb_run_id.txt saved next to the checkpoints during training.
    """
    run_id = cfg.get("wandb_run_id", None)
    if run_id:
        return str(run_id)
    id_file = Path(ckpt_path).parent / "wandb_run_id.txt"
    if id_file.exists():
        return id_file.read_text().strip() or None
    return None


@hydra.main(version_base=None, config_path="config", config_name="trm_diffusion")
def main(cfg: DictConfig):
    seed = cfg.get("seed", 42)
    L.seed_everything(seed, workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_embedded = cfg.get("trm") is not None and cfg.trm.get("n_inner") is not None
    is_trm = cfg.get("trm") is not None and not is_embedded
    is_std = not (is_trm or is_embedded)

    run_name = cfg.get("run_name", None)
    if not run_name and is_std:
        raise ValueError("You must provide a run name (run_name) for sampling standard diffusion.")

    ckpt_path = cfg.get("checkpoint_path", "best")
    if ckpt_path == "best":
        ckpt_path = find_best_checkpoint(cfg.output_dir, run_name or "", is_std)
    elif not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")

    num_bits = cfg.num_bits
    k_values = cfg.dataset.get("k_values", [1, 3, 5, 8, 10])
    use_self_cond = cfg.get("self_cond", False)

    if cfg.conditioning == "k_steps":
        in_channels = num_bits * (2 if use_self_cond else 1) + num_bits
        num_classes = len(k_values)
    else:
        in_channels = num_bits * (2 if use_self_cond else 1)
        num_classes = cfg.num_classes

    core_model = Transformer2DModel(
        sample_size=cfg.resolution,
        in_channels=in_channels,
        out_channels=num_bits,
        num_layers=cfg.model.num_layers,
        patch_size=cfg.model.patch_size,
        attention_head_dim=cfg.model.attention_head_dim,
        num_attention_heads=cfg.model.num_attention_heads,
        cross_attention_dim=cfg.model.get("cross_attention_dim", None),
        activation_fn=cfg.model.get("activation_fn", "gelu-approximate"),
        dropout=cfg.model.get("dropout", 0.0),
        num_embeds_ada_norm=num_classes + 1,
        norm_type="ada_norm_zero",
    ) if not is_embedded else None

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=cfg.get("ddpm_num_train_timesteps", 1000),
        beta_schedule=cfg.get("beta_schedule", "squaredcos_cap_v2"),
        prediction_type=cfg.get("prediction_type", "sample"),
        rescale_betas_zero_snr=cfg.get("rescale_betas_zero_snr", True),
        clip_sample=True,
        clip_sample_range=1.0,
    )

    if is_embedded:
        model = EmbeddedTRMDiffusion(
            num_attention_heads_dit=cfg.model.num_attention_heads,
            attention_head_dim_dit=cfg.model.attention_head_dim,
            num_embeds_ada_norm=num_classes + 1,
            num_attention_heads_trm=cfg.trm.get("num_attention_heads", cfg.model.num_attention_heads),
            attention_head_dim_trm=cfg.trm.get("attention_head_dim", cfg.model.attention_head_dim),
            num_layers_trm=cfg.trm.get("num_inner_layers", 1),
            n_inner=cfg.trm.n_inner,
            T=cfg.trm.T,
            num_layers=cfg.model.num_layers,
            resolution=cfg.resolution,
            in_channels=in_channels,
            out_channels=num_bits,
            ffn_mult=cfg.model.get("ffn_mult", 4),
            activation_fn=cfg.model.get("activation_fn", "gelu-approximate"),
            dropout=cfg.model.get("dropout", 0.0),
            patch_size=cfg.model.get("patch_size", 1),
            use_grid_pos_embed=cfg.trm.get("use_grid_pos_embed", True),
            refine_residual=cfg.trm.get("refine_residual", True),
            shared_stack=cfg.trm.get("shared_stack", False),
            weight_tied=cfg.trm.get("weight_tied", False),
            isolate_transform=cfg.trm.get("isolate_transform", False),
            use_gate=cfg.trm.get("use_gate", False),
        )
        lit_model = SokobanEmbeddedTRMDiffusion.load_from_checkpoint(
            ckpt_path, model=model, noise_scheduler=noise_scheduler, map_location=device,
        )
    elif is_trm:
        model = TRMDiT(
            core_model=core_model,
            resolution=cfg.resolution,
            n=cfg.trm.n,
            T=cfg.trm.T,
            n_sup_max=cfg.n_sup_max,
            use_grid_pos_embed=cfg.trm.get("use_grid_pos_embed", True),
        )
        lit_model = SokobanTRMBitDiffusion.load_from_checkpoint(
            ckpt_path, model=model, noise_scheduler=noise_scheduler, map_location=device,
        )
    else:
        lit_model = SokobanBitDiffusion.load_from_checkpoint(
            ckpt_path, model=core_model, noise_scheduler=noise_scheduler, map_location=device,
        )
    print(f"Loaded checkpoint ({'embedded' if is_embedded else 'trm' if is_trm else 'std'}): {ckpt_path}")

    lit_model.eval()
    lit_model.to(device)

    lit_model.time_shift_xi = cfg.get("time_shift_xi", 0.0)

    num_samples = cfg.get("num_samples", 200)
    batch_size = cfg.get("sample_batch_size", 50)

    data_module = SokobanBitDataModule(
        data_path=cfg.dataset.get("test_path", cfg.dataset.data_path),
        conditioning=cfg.conditioning,
        total_train_size=cfg.dataset.total_train_size,
        total_eval_size=cfg.dataset.total_eval_size,
        batch_size=batch_size,
        num_workers=cfg.num_workers,
        num_bits=num_bits,
        k_values=k_values,
    )
    data_module.setup()

    print(f"Generating {num_samples} boards...")
    metrics, gen_bits = lit_model.evaluate(data_module.val_dataloader(), num_samples=num_samples)

    print("\n--- Sokoban Metrics ---")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")

    shift_str = f"_shift_{lit_model.time_shift_xi}"
    prefix = "emb" if is_embedded else "trm" if is_trm else "std"
    run_dir = run_name if run_name else "trm_default"

    sample_dir = Path(cfg.output_dir) / run_dir / f"{prefix}_samples{shift_str}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving rendered boards to {sample_dir}...")
    val_ds = data_module.val_ds
    for i in tqdm(range(min(num_samples, 100)), desc="Rendering"):
        img = val_ds.render_bit_boards(gen_bits[i])
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img.save(sample_dir / f"board_{i:04d}.png")

    metrics_path = sample_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # --- Log test metrics back onto the original training run in W&B ---
    if cfg.get("log_to_wandb", True):
        run_id = _resolve_wandb_run_id(cfg, ckpt_path)
        if run_id is None:
            print(
                "W&B: no run id found. Pass wandb_run_id=<id> (copy it from the run URL) "
                "or retrain so wandb_run_id.txt is saved next to the checkpoint. "
                "Skipping W&B logging."
            )
        else:
            try:
                shift = lit_model.time_shift_xi
                wandb.init(
                    project=cfg.get("wandb_project", None),
                    entity=cfg.get("wandb_entity", None),
                    id=run_id,
                    resume="must",
                )
                wandb.define_metric("test/time_shift_xi")
                wandb.define_metric("test/sokoban/*", step_metric="test/time_shift_xi")

                log_dict = {f"test/{k}": v for k, v in metrics.items()}
                log_dict["test/time_shift_xi"] = shift

                # Per-shift board panel so renders from different shifts don't overwrite.
                boards = []
                for i in range(min(num_samples, 16)):
                    img = val_ds.render_bit_boards(gen_bits[i])
                    if isinstance(img, np.ndarray):
                        img = Image.fromarray(img)
                    boards.append(wandb.Image(img, caption=f"shift={shift} sample_{i}"))
                log_dict[f"test/boards_shift_{shift}"] = boards

                wandb.log(log_dict)
                wandb.finish()
                print(f"Logged {len(metrics)} test metrics (shift={shift}) to W&B run '{run_id}'.")
            except Exception as e:
                print(f"W&B logging failed ({e}). Local artifacts were still saved.")


if __name__ == "__main__":
    sys.argv = [a for a in sys.argv if not a.startswith("--") or a.startswith("--config")]
    main()

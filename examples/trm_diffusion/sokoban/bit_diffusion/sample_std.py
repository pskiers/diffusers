"""
Sampling script for Bit-Diffusion Sokoban.

Loads the best checkpoint, generates boards, prints metrics and saves rendered images.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import lightning as L
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from tqdm.auto import tqdm
from sokoban.bit_diffusion.train_std import SokobanBitDataModule, SokobanBitDiffusion

from diffusers import DDPMScheduler, Transformer2DModel


def find_best_checkpoint(output_dir: str, run_name: str) -> str:
    ckpt_dir = Path(output_dir) / run_name / "std_checkpoints"
    candidates = list(ckpt_dir.glob("best-*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No best-*.ckpt found in {ckpt_dir}")
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


@hydra.main(version_base=None, config_path="config", config_name="standard_diffusion")
def main(cfg: DictConfig):
    seed = cfg.get("seed", 42)
    L.seed_everything(seed, workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = cfg.get("run_name", None)
    if not run_name:
        raise ValueError("You must provide training run name to sample")

    ckpt_path = cfg.get("checkpoint_path", "best")
    if ckpt_path == "best":
        ckpt_path = find_best_checkpoint(cfg.output_dir, run_name)
    elif not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"Loading checkpoint: {ckpt_path}")

    num_bits = cfg.num_bits
    k_values = cfg.dataset.get("k_values")

    use_self_cond = cfg.get("self_cond", False)
    if cfg.conditioning == "k_steps":
        in_channels = num_bits * (2 if use_self_cond else 1) + num_bits
        num_classes = len(k_values)
    else:
        in_channels = num_bits * (2 if use_self_cond else 1)
        num_classes = cfg.num_classes

    model = Transformer2DModel(
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
    )
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=cfg.get("ddpm_num_train_timesteps", 1000),
        beta_schedule=cfg.get("beta_schedule", "squaredcos_cap_v2"),
        prediction_type=cfg.get("prediction_type", "sample"),
        rescale_betas_zero_snr=cfg.get("rescale_betas_zero_snr", True),
        clip_sample=True,
        clip_sample_range=1.0,
    )
    lit_model = SokobanBitDiffusion.load_from_checkpoint(
        ckpt_path, model=model, noise_scheduler=noise_scheduler, map_location=device,
    )
    lit_model.eval()
    lit_model.to(device)
    lit_model.time_shift_xi = cfg.time_shift_xi

    num_samples = cfg.get("num_samples", 256)
    batch_size = cfg.get("sample_batch_size", 64)
    data_module = SokobanBitDataModule(
        data_path=cfg.dataset.test_path,
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
        print(f"  {k}: {v:.2f}")

    # Save rendered images
    sample_dir = Path(cfg.output_dir) / run_name / f"std_samples_shift_{cfg.time_shift_xi}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving rendered boards to {sample_dir}...")
    val_ds = data_module.val_ds
    for i in tqdm(range(min(num_samples, 100)), desc="Rendering"):
        img = val_ds.render_bit_boards(gen_bits[i])
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img.save(sample_dir / f"board_{i:04d}.png")

    # Save metrics
    metrics_path = sample_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    sys.argv = [a for a in sys.argv if not a.startswith("--")]
    main()

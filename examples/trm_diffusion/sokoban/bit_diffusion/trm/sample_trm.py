"""
Sampling script for Bit-Diffusion Sokoban (standard and TRM).

Loads the best checkpoint, generates boards via lit_model.evaluate(),
prints metrics, and saves rendered images.

Usage (standard diffusion):
  python sample.py --config-name=standard_diffusion checkpoint_path=best

Usage (TRM diffusion):
  python sample.py --config-name=trm_diffusion checkpoint_path=best
"""

import json
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from tqdm.auto import tqdm

from diffusers import Transformer2DModel
from diffusers.training_utils import EMAModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset.sokoban_dataset import SokobanBitsDataset

from train import SokobanBitDiffusion, SokobanBitDataModule
from train_trm import SokobanTRMBitDiffusion, TRMDiT


def find_best_checkpoint(output_dir: str) -> str:
    ckpt_dir = Path(output_dir) / "checkpoints"
    candidates = list(ckpt_dir.glob("best-*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No best-*.ckpt found in {ckpt_dir}")
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def build_model(cfg: DictConfig):
    """Build the appropriate model (standard or TRM) from config."""
    num_bits = cfg.num_bits
    in_channels = num_bits * 3 if cfg.conditioning == "k_steps" else num_bits * 2

    core_model = Transformer2DModel(
        sample_size=cfg.resolution,
        in_channels=in_channels,
        out_channels=num_bits,
        num_layers=cfg.model.num_layers,
        patch_size=cfg.model.patch_size,
        attention_head_dim=cfg.model.attention_head_dim,
        num_attention_heads=cfg.model.num_attention_heads,
        cross_attention_dim=None,
        activation_fn=cfg.model.get("activation_fn", "gelu-approximate"),
        dropout=cfg.model.get("dropout", 0.0),
        num_embeds_ada_norm=cfg.num_classes + 1,
        norm_type="ada_norm_zero",
    )

    if cfg.get("trm") is not None:
        return TRMDiT(
            core_model=core_model,
            resolution=cfg.resolution,
            n=cfg.trm.n,
            T=cfg.trm.T,
            n_sup_max=cfg.n_sup_max,
            use_grid_pos_embed=cfg.trm.get("use_grid_pos_embed", True),
        )
    return core_model


@hydra.main(version_base=None, config_path="../config", config_name="trm_diffusion")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve checkpoint
    ckpt_path = cfg.get("checkpoint_path", "best")
    if ckpt_path == "best":
        ckpt_path = find_best_checkpoint(cfg.output_dir)
    elif not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")

    # Build model and load checkpoint
    model = build_model(cfg)
    is_trm = cfg.get("trm") is not None

    if is_trm:
        lit_model = SokobanTRMBitDiffusion.load_from_checkpoint(
            ckpt_path, model=model, map_location=device,
        )
        # Apply EMA weights for inference (if EMA was used during training)
        ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "ema_state_dict" in ckpt_data:
            ema = EMAModel(lit_model.model.parameters(), decay=cfg.get("ema_decay", 0.9999))
            ema.load_state_dict(ckpt_data["ema_state_dict"])
            ema.copy_to(lit_model.model.parameters())
            print("Applied EMA weights for inference.")
        del ckpt_data
    else:
        lit_model = SokobanBitDiffusion.load_from_checkpoint(
            ckpt_path, model=model, map_location=device,
        )

    lit_model.eval()
    lit_model.to(device)

    # Data
    num_bits = cfg.num_bits
    num_samples = cfg.get("num_samples", 200)
    batch_size = cfg.get("sample_batch_size", 50)

    data_module = SokobanBitDataModule(
        data_path=cfg.dataset.test_path,
        conditioning=cfg.conditioning,
        total_train_size=cfg.dataset.total_train_size,
        total_eval_size=cfg.dataset.total_eval_size,
        batch_size=batch_size,
        num_workers=cfg.num_workers,
        num_bits=num_bits,
        k_values=cfg.dataset.get("k_values", [1, 3, 5, 8, 10]),
    )
    data_module.setup()

    # Generate & evaluate
    print(f"Generating {num_samples} boards...")
    metrics, gen_bits = lit_model.evaluate(data_module.val_dataloader(), num_samples=num_samples)

    print("\n--- Sokoban Metrics ---")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")

    # Save rendered images
    sample_dir = Path(cfg.output_dir) / "samples"
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

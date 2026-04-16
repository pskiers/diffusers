"""
sample_mnist_sudoku.py – Sample from a trained MNIST-Sudoku Ratatouille model.

Usage:
    python sample_mnist_sudoku.py experiment=v0 checkpoint_path=runs/mnist_sudoku_v0/checkpoint_best.pt
    accelerate launch --num_processes=1 sample_mnist_sudoku.py experiment=v1 \\
        checkpoint_path=runs/mnist_sudoku_v1/checkpoint_best.pt num_samples=64

Outputs:
    {output_dir}/samples/
        {rank:04d}_{idx:06d}.png   – generated images
        metadata_rank{rank}.jsonl  – one JSON line per sample
"""

import inspect
import json
import logging
import os
from pathlib import Path

import hydra
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import DDIMScheduler, DDPMScheduler
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm.auto import tqdm

from mnist_sudoku_models import (
    MNISTRatatouilleV0,
    MNISTRatatouilleV1,
    MNISTRatatouilleV2,
    MNISTRatatouilleV3,
    MNISTRatatouilleV4,
    MNISTRatatouilleV0Control,
    MNISTRatatouilleV1Control,
    MNISTRatatouilleV2Control,
    MNISTRatatouilleV3Control,
    MNISTRatatouilleV4Control,
    MNISTRatatouilleV0SPADE,
    MNISTRatatouilleV1SPADE,
    MNISTRatatouilleV2SPADE,
    MNISTRatatouilleV3SPADE,
    MNISTRatatouilleV4SPADE,
    MNISTRatatouilleV0Tok,
)

logger = get_logger(__name__, log_level="INFO")

MODEL_REGISTRY = {
    "v0": MNISTRatatouilleV0,
    "v0tok": MNISTRatatouilleV0Tok,
    "v1": MNISTRatatouilleV1,
    "v2": MNISTRatatouilleV2,
    "v3": MNISTRatatouilleV3,
    "v4": MNISTRatatouilleV4,
    "v0control": MNISTRatatouilleV0Control,
    "v1control": MNISTRatatouilleV1Control,
    "v2control": MNISTRatatouilleV2Control,
    "v3control": MNISTRatatouilleV3Control,
    "v4control": MNISTRatatouilleV4Control,
    "v0spade": MNISTRatatouilleV0SPADE,
    "v1spade": MNISTRatatouilleV1SPADE,
    "v2spade": MNISTRatatouilleV2SPADE,
    "v3spade": MNISTRatatouilleV3SPADE,
    "v4spade": MNISTRatatouilleV4SPADE,
}


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """(1, H, W) float32 → PIL grayscale."""
    arr = t.squeeze(0).clamp(0, 1).cpu().numpy()
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


@hydra.main(version_base=None, config_path="configs/mnist_sudoku", config_name="config")
def main(args: DictConfig):
    accelerator = Accelerator(mixed_precision=args.get("mixed_precision", "no"))
    logging.basicConfig(level=logging.INFO)

    if accelerator.is_main_process:
        logger.info(OmegaConf.to_yaml(args))

    checkpoint_path = args.sample.get("checkpoint_path")
    if not checkpoint_path:
        raise ValueError("checkpoint_path must be specified (set sample.checkpoint_path=...)")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # ── Build model ──────────────────────────────────────────────────────────
    cell_size    = args.data.get("cell_size", 32)
    painter_size = cell_size * 9
    variant      = args.model.variant
    ModelCls     = MODEL_REGISTRY[variant]

    model_kwargs = dict(OmegaConf.to_container(args.model.get("kwargs", {}), resolve=True))
    model_kwargs.setdefault("painter_size", painter_size)
    model_kwargs.setdefault("cell_size", cell_size)
    valid_params = set(inspect.signature(ModelCls.__init__).parameters) - {"self"}
    model_kwargs = {k: v for k, v in model_kwargs.items() if k in valid_params}
    model = ModelCls(**model_kwargs)

    # ── Load weights ─────────────────────────────────────────────────────────
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    logger.info(f"Loaded checkpoint from step {ckpt.get('step', '?')}")

    model = accelerator.prepare(model)
    model.eval()

    # ── Scheduler ────────────────────────────────────────────────────────────
    use_ddim = args.sample.get("use_ddim", False)
    num_inference_steps = args.sample.get("num_steps", 100)

    prediction_type = args.get("prediction_type", "epsilon")
    if use_ddim:
        scheduler = DDIMScheduler(
            num_train_timesteps=args.get("num_timesteps", 1000),
            beta_schedule=args.get("beta_schedule", "squaredcos_cap_v2"),
            prediction_type=prediction_type,
        )
    else:
        scheduler = DDPMScheduler(
            num_train_timesteps=args.get("num_timesteps", 1000),
            beta_schedule=args.get("beta_schedule", "squaredcos_cap_v2"),
            prediction_type=prediction_type,
        )
    scheduler.set_timesteps(num_inference_steps)

    # ── Sampling ─────────────────────────────────────────────────────────────
    num_samples       = args.sample.get("num_samples", 16)
    sample_batch_size = args.sample.get("batch_size", 8)
    output_dir        = Path(args.output_dir) / "samples"

    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    device    = accelerator.device
    rank      = accelerator.process_index
    generated = 0
    metadata_lines: list[str] = []

    pbar = tqdm(
        total=num_samples,
        disable=not accelerator.is_local_main_process,
        desc="Sampling",
    )

    while generated < num_samples:
        B = min(sample_batch_size, num_samples - generated)

        # Condition: all-blank sudoku (zeros) unless a dataset path is given.
        condition_source = args.sample.get("condition_source", "blank")
        if condition_source == "blank":
            conditions = torch.zeros(B, 1, painter_size, painter_size, device=device)
        else:
            raise NotImplementedError("Only 'blank' condition_source is currently supported")

        # Reverse diffusion
        noisy = torch.randn(B, 1, painter_size, painter_size, device=device)
        with torch.no_grad():
            for t in scheduler.timesteps:
                t_batch     = torch.full((B,), t, device=device, dtype=torch.long)
                noise_pred, _ = accelerator.unwrap_model(model)(noisy, t_batch, conditions)
                noisy       = scheduler.step(noise_pred, t, noisy).prev_sample

        images = noisy  # (B, 1, H, W) pixel-space

        if accelerator.is_main_process:
            for i in range(B):
                idx  = generated + i
                fname = f"{rank:04d}_{idx:06d}.png"
                _tensor_to_pil(images[i]).save(output_dir / fname)
                metadata_lines.append(json.dumps({"rank": rank, "idx": idx, "file": fname}))

        generated += B
        pbar.update(B)

    if accelerator.is_main_process:
        meta_path = output_dir / f"metadata_rank{rank}.jsonl"
        with open(meta_path, "w") as f:
            f.write("\n".join(metadata_lines) + "\n")
        logger.info(f"Saved {num_samples} samples to {output_dir}")


if __name__ == "__main__":
    main()

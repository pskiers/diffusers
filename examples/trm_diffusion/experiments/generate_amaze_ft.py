#!/usr/bin/env python3
"""Generate K solution images per test puzzle with a fine-tuned model, in the
layout score_amaze_images.py consumes:

    <gen_dir>/<combo>/<puzzle_index>_<attempt>.png

``combo`` = ``{geometry}_n{scale}`` (maze) or ``n{scale}`` (queens); ``puzzle_index``
is the 0-based row index in that test parquet; ``attempt`` is 0..K-1.

Backends:
  dummy  - copies the input puzzle image as each attempt (pipeline smoke-test,
           no model needed — verifies generate -> score -> wandb end to end).
  bagel  - fine-tuned BAGEL via the vendored InterleaveInferencer   [wire on cluster]
  janus  - fine-tuned Janus-Pro two-stage generation                [wire on cluster]

The input image is the native-resolution ``m_original_img`` (the model's own
transforms resize it); the scorer downsizes generated images to 144 for metrics.
"""
from __future__ import annotations

import argparse
import base64
import io
import os
from pathlib import Path

import pandas as pd
from PIL import Image

TRM_ROOT = Path(__file__).resolve().parent.parent

MAZE_SCALES = [5, 7, 8, 9, 11, 13, 16]
MAZE_OOD_SCALES = [3]
MAZE_GEOMETRIES = ["square", "hexagon", "triangle", "circle"]
QUEEN_SCALES = [4, 5, 6, 7, 8, 9, 10]
QUEEN_OOD_SCALES = [12]

MAZE_PROMPT = (
    "Add the blue solution path for the maze, connect start point (solid red circle) "
    "to end point (red 'X' mark). Ensure all original maze elements (walls, points, "
    "etc.) remain unchanged\u2014only add the path."
)
QUEEN_PROMPT = (
    "Given the puzzle image, generate the solved board by placing one queen "
    "(represented by a solid black circle in the center of a grid cell) in each row, "
    "column, and colored region while ensuring queens do not touch in 8-neighborhood."
)


def _decode(cell) -> Image.Image:
    if isinstance(cell, Image.Image):
        return cell.convert("RGB")
    if isinstance(cell, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(cell))).convert("RGB")
    s = cell.split(",", 1)[1] if isinstance(cell, str) and cell.startswith("data:") else cell
    return Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")


def _iter_combos(task: str, data_root: Path):
    if task == "maze":
        for g in MAZE_GEOMETRIES:
            for s in MAZE_SCALES + MAZE_OOD_SCALES:
                yield f"{g}_n{s}", data_root / "test_maze" / g / f"n{s}_{g}_test.parquet"
    else:
        for s in QUEEN_SCALES + QUEEN_OOD_SCALES:
            yield f"n{s}", data_root / "test_queens" / f"n{s}_test.parquet"


class DummyBackend:
    def generate(self, image: Image.Image, prompt: str, k: int):
        return [image.convert("RGB") for _ in range(k)]


# ── Bagel FT inference backend ─────────────────────────────────────────────️
_INFER_ROOT = TRM_ROOT / "third_party" / "amaze" / "infer"


def _prepare_bagel_imports() -> None:
    """Make the vendored ``bagel.*`` package importable on this cluster: put
    third_party/amaze/infer on sys.path and shim torch.nn.attention.flex_attention
    (added in torch 2.5; the venv is 2.4.x). Runs with use_flex disabled, so the
    shimmed symbols are import-time only and never executed."""
    import sys
    import types

    import torch

    if str(_INFER_ROOT) not in sys.path:
        sys.path.insert(0, str(_INFER_ROOT))

    try:
        import torch.nn.attention.flex_attention  # noqa: F401  (present on torch >= 2.5)
    except ModuleNotFoundError:
        mod = types.ModuleType("torch.nn.attention.flex_attention")

        def or_masks(*mask_mods):
            def combined(b, h, q, kv):
                r = q.new_zeros((), dtype=torch.bool)
                for f in mask_mods:
                    r = r | f(b, h, q, kv)
                return r
            return combined

        def and_masks(*mask_mods):
            def combined(b, h, q, kv):
                r = q.new_ones((), dtype=torch.bool)
                for f in mask_mods:
                    r = r & f(b, h, q, kv)
                return r
            return combined

        mod.or_masks = or_masks
        mod.and_masks = and_masks
        mod.create_block_mask = lambda *a, **k: None
        mod.flex_attention = lambda *a, **k: None
        mod.BlockMask = object
        sys.modules["torch.nn.attention.flex_attention"] = mod


def _find_weights(path: Path) -> Path:
    """Resolve a Bagel weights file: a .safetensors file directly, a dir holding
    model/ema.safetensors, or a checkpoints/ parent (latest numeric step subdir)."""
    names = ["model.safetensors", "ema.safetensors", "consolidated.safetensors"]
    if path.is_file():
        return path
    for n in names:
        if (path / n).is_file():
            return path / n
    step_dirs = sorted(
        (d for d in path.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda d: int(d.name),
    )
    for d in reversed(step_dirs):
        for n in names:
            if (d / n).is_file():
                return d / n
    raise FileNotFoundError(f"no Bagel weights ({'/'.join(names)}) found under {path}")


class BagelBackend:
    """Fine-tuned BAGEL-7B-MoT inference via the vendored InterleaveInferencer;
    mirrors third_party/amaze/infer/infer_bagel.py's model assembly."""

    def __init__(self, checkpoint: str, model_path: str, num_timesteps: int = 50):
        _prepare_bagel_imports()

        import torch
        from accelerate import init_empty_weights, load_checkpoint_in_model
        from accelerate.utils import set_module_tensor_to_device
        from safetensors.torch import load_file

        from bagel.data.data_utils import add_special_tokens
        from bagel.data.transforms import ImageTransform
        from bagel.inferencer import InterleaveInferencer
        from bagel.modeling.autoencoder import load_ae
        from bagel.modeling.bagel import (
            Bagel, BagelConfig, Qwen2Config, Qwen2ForCausalLM,
            SiglipVisionConfig, SiglipVisionModel,
        )
        from bagel.modeling.qwen2 import Qwen2Tokenizer

        if not model_path:
            raise ValueError(
                "bagel backend needs the base BAGEL-7B-MoT dir: pass --bagel-model-path "
                "or set BAGEL_MODEL_PATH."
            )
        model_dir = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16
        self.num_timesteps = num_timesteps

        llm_config = Qwen2Config.from_json_file(str(model_dir / "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"

        vit_config = SiglipVisionConfig.from_json_file(str(model_dir / "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers -= 1

        vae_model, vae_config = load_ae(local_path=str(model_dir / "ae.safetensors"))

        bagel_config = BagelConfig(
            visual_gen=True,
            visual_und=True,
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=vae_config,
            vit_max_num_patch_per_side=70,
            connector_act="gelu_pytorch_tanh",
            latent_patch_size=2,
            max_latent_size=64,
        )

        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            model = Bagel(language_model, vit_model, bagel_config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

        tokenizer = Qwen2Tokenizer.from_pretrained(str(model_dir))
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

        vae_transform = ImageTransform(1024, 512, 16)
        vit_transform = ImageTransform(980, 224, 14)

        # Base weights (VAE+LLM+ViT) first, then the FT delta on top (strict=False).
        base_ckpt = _find_weights(model_dir)
        load_checkpoint_in_model(
            model, checkpoint=str(base_ckpt), device_map={"": "cpu"},
            dtype=self.dtype, offload_folder="/tmp/bagel_offload",
        )
        ft_ckpt = _find_weights(Path(checkpoint))
        msg = model.load_state_dict(load_file(str(ft_ckpt)), strict=False)
        print(f">> Bagel FT load ({ft_ckpt}): missing={len(msg.missing_keys)} "
              f"unexpected={len(msg.unexpected_keys)}", flush=True)

        # Any params still on meta (missing from both) -> base values or zeros.
        base_state = None
        for name, p in list(model.named_parameters()) + list(model.named_buffers()):
            if getattr(p, "device", None) is not None and p.device.type == "meta":
                if base_state is None:
                    base_state = load_file(str(base_ckpt))
                val = base_state.get(name)
                if val is None:
                    val = torch.zeros(p.shape, dtype=self.dtype)
                set_module_tensor_to_device(model, name, "cpu", value=val)

        model = model.eval()
        model.requires_grad_(False)
        model = model.to(self.device, dtype=self.dtype)
        vae_model = vae_model.to(self.device, dtype=torch.float32)

        self.inferencer = InterleaveInferencer(
            model=model, vae_model=vae_model, tokenizer=tokenizer,
            vae_transform=vae_transform, vit_transform=vit_transform,
            new_token_ids=new_token_ids,
        )

    def generate(self, image: Image.Image, prompt: str, k: int):
        # One batched call draws k independent noise seeds -> k Pass@K candidates.
        out = self.inferencer(
            image=[image.convert("RGB")] * k,
            text=[prompt] * k,
            num_timesteps=self.num_timesteps,
            cfg_text_scale=1.0,
            cfg_img_scale=1.0,
            cfg_interval=[0.0, 1.0],
            cfg_renorm_min=0.0,
        )
        imgs = out.get("images") or ([out["image"]] if out.get("image") is not None else [])
        imgs = [im.convert("RGB") for im in imgs if im is not None]
        if not imgs:
            imgs = [image.convert("RGB")]
        while len(imgs) < k:
            imgs.append(imgs[-1])
        return imgs[:k]


def build_backend(name: str, checkpoint: str | None, model_path: str | None = None,
                  num_timesteps: int = 50):
    if name == "dummy":
        return DummyBackend()
    if name == "bagel":
        if not checkpoint:
            raise ValueError("bagel backend needs --checkpoint (the FT checkpoint dir).")
        return BagelBackend(checkpoint, model_path or "", num_timesteps=num_timesteps)
    if name == "janus":
        raise NotImplementedError(
            "janus backend: load Janus-Pro FT weights and reproduce the two-stage image generation "
            "from third_party/amaze/infer/infer_janus.py. Wire during the cluster smoke-test."
        )
    raise ValueError(f"unknown backend '{name}'")


def main():
    ap = argparse.ArgumentParser(description="Generate FT solution images in the scorer's layout.")
    ap.add_argument("task", choices=["maze", "queens"])
    ap.add_argument("--backend", choices=["dummy", "bagel", "janus"], required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--bagel-model-path", default=os.environ.get("BAGEL_MODEL_PATH"),
                    help="Base BAGEL-7B-MoT snapshot dir (bagel backend); defaults to $BAGEL_MODEL_PATH.")
    ap.add_argument("--num-timesteps", type=int,
                    default=int(os.environ.get("BAGEL_NUM_TIMESTEPS", "50")),
                    help="Denoising steps for the bagel backend.")
    ap.add_argument("--data-root", type=Path, default=TRM_ROOT / "data" / "amaze")
    ap.add_argument("--gen-dir", type=Path, required=True)
    ap.add_argument("--samples-per-puzzle", type=int, default=5)
    args = ap.parse_args()

    backend = build_backend(args.backend, args.checkpoint, args.bagel_model_path, args.num_timesteps)
    fallback_prompt = MAZE_PROMPT if args.task == "maze" else QUEEN_PROMPT
    k = args.samples_per_puzzle
    total = 0

    for combo, parquet in _iter_combos(args.task, args.data_root):
        if not parquet.exists():
            print(f"WARN: missing {parquet} — skipping {combo}")
            continue
        df = pd.read_parquet(parquet)
        out = args.gen_dir / combo
        out.mkdir(parents=True, exist_ok=True)
        for idx, row in df.iterrows():
            image = _decode(row["m_original_img"])
            prompt = row.get("instruction") or row.get("text") or fallback_prompt
            for a, im in enumerate(backend.generate(image, prompt, k)):
                im.convert("RGB").save(out / f"{idx}_{a}.png")
        total += len(df)
        print(f">> {combo}: {len(df)} puzzles x{k} -> {out}")

    print(f"Done ({args.backend}) -> {args.gen_dir} ({total} puzzles)")


if __name__ == "__main__":
    main()

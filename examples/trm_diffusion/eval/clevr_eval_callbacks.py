"""
eval/clevr_eval_callbacks.py — CLEVR-specific eval callbacks.

ClevrImageLogCallback
    Samples n_scenes from the CLEVR validation set at init (filtered by
    object count range).  Every eval call generates images for those fixed
    scenes and logs them to WandB.  Useful for watching image quality
    evolve over training.

ClevrMetricsCallback
    Every eval call samples n_samples random CLEVR scenes, generates images
    in batches, runs DINO + SigLIP evaluation, and logs all metrics to
    WandB.  DINO / SigLIP are lazy-loaded on the first call (main process
    only) to avoid wasting VRAM on worker processes.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm.auto import tqdm

try:
    import wandb as _wandb
except ImportError:
    _wandb = None

from models.eval_callbacks import EvalCallbackBase
from datasets.data_sample import DataSample, collate_data_samples
from datasets.clevr_dataset import (
    CLEVRHybridDataset,
    calibrate_mask_projection,
    make_mask_from_scene,
    make_reveal_from_scene,
    make_tensor_from_scene,
    ORIG_W,
    ORIG_H,
)


def _load_scenes(root_dir: str, split: str, min_objects: int, max_objects: int):
    """Load CLEVR scenes in mode='absolute' (no pre-computation), filtered by count."""
    ds = CLEVRHybridDataset(root_dir=root_dir, split=split, mode="absolute", download=False)
    return [s for s in ds.scenes if min_objects <= len(s["objects"]) <= max_objects]


def _scene_to_data_sample(
    scene: dict,
    mode: str,
    *,
    image_dir: Optional[str] = None,
    transform=None,
    reveal_n_objects: int = 0,
    reveal_radius_frac: float = 0.12,
    include_centroid_mask: bool = False,
    mask_size: int = 32,
    H_inv=None,
) -> DataSample:
    """Build a conditioning-only DataSample (no generated image yet) from a
    scene dict. Mirrors CLEVRHybridDataset.__getitem__'s spatial_conditions
    logic: any condition encoder whose condition_keys include
    "spatial_conditions" (the reveal/centroid-mask variants) needs it
    populated here too, or sampling crashes with spatial_conditions=None —
    reveal/centroid-mask are diagnostic (using the scene's ground truth), but
    that ground truth is exactly this scene's known real image/attributes, so
    it's available here the same way it is during training."""
    scene = dict(scene)
    scene["mode"] = mode
    cond_tensor, mask = make_tensor_from_scene(scene)

    spatial_conditions = None
    if reveal_n_objects > 0:
        image = Image.open(os.path.join(image_dir, scene["image_filename"])).convert("RGB")
        image_t = transform(image)
        spatial_conditions = make_reveal_from_scene(image_t, scene, reveal_n_objects, reveal_radius_frac)
    elif include_centroid_mask:
        spatial_conditions = make_mask_from_scene(scene, mask_size, H_inv)

    return DataSample(embedding_conditions=cond_tensor[0], embedding_mask=mask[0], spatial_conditions=spatial_conditions)


def _generate_images(model, samples: list[DataSample], device: torch.device) -> torch.Tensor:
    """Generate and decode images for a list of conditioning DataSamples, batched."""
    pipeline = model.sampling_pipeline
    imgs = []
    for start in range(0, len(samples), pipeline.batch_size):
        chunk = samples[start : start + pipeline.batch_size]
        with torch.no_grad():
            batch = collate_data_samples(chunk).to(device)
            latents = pipeline.sample_one_batch(model, batch, device)
        imgs.append(model.decode_for_eval(latents))
    return torch.cat(imgs, dim=0)


# ── Image log callback ────────────────────────────────────────────────────────


class ClevrImageLogCallback(EvalCallbackBase):
    """Sample fixed CLEVR scenes at init; log generated images each eval.

    Args:
        root_dir:       Path to CLEVR dataset root (must contain CLEVR_v1.0/).
        mode:           Conditioning mode — "reduced" or "relative".
        n_scenes:       Number of fixed scenes to sample at init.
        min_objects:    Minimum number of objects per scene.
        max_objects:    Maximum number of objects per scene.
        split:          Dataset split to sample scenes from.
        reveal_n_objects, reveal_radius_frac, include_centroid_mask, image_size:
            must match data.reveal_n_objects / data.reveal_radius_frac /
            data.include_centroid_mask / data.image_size whenever the bound
            condition encoder needs spatial_conditions (reveal/centroid-mask
            variants) — otherwise generation crashes on spatial_conditions=None.
    """

    def __init__(
        self,
        root_dir: str,
        mode: str = "reduced",
        n_scenes: int = 8,
        min_objects: int = 3,
        max_objects: int = 10,
        split: str = "val",
        reveal_n_objects: int = 0,
        reveal_radius_frac: float = 0.12,
        include_centroid_mask: bool = False,
        image_size: int = 256,
    ):
        self._mode = mode
        scenes = _load_scenes(root_dir, split, min_objects, max_objects)
        selected = random.sample(scenes, min(n_scenes, len(scenes)))

        filename_split = "val" if split == "validation" else split
        image_dir = os.path.join(root_dir, "CLEVR_v1.0", "images", filename_split)
        transform = T.Compose([T.Resize((image_size, image_size)), T.ToTensor(), T.Normalize([0.5], [0.5])])
        H_inv = calibrate_mask_projection(scenes) if include_centroid_mask else None

        self._base_samples = [
            _scene_to_data_sample(
                s,
                mode,
                image_dir=image_dir,
                transform=transform,
                reveal_n_objects=reveal_n_objects,
                reveal_radius_frac=reveal_radius_frac,
                include_centroid_mask=include_centroid_mask,
                mask_size=image_size // 8,
                H_inv=H_inv,
            )
            for s in selected
        ]

    @torch.no_grad()
    def __call__(self, model, dataloader, accelerator, **kwargs) -> dict:
        if not accelerator.is_main_process or _wandb is None:
            return {}

        step = kwargs.get("step", None)
        imgs = _generate_images(model, self._base_samples, accelerator.device)

        if step is not None:
            try:
                from torchvision.utils import make_grid
                grid = make_grid(imgs.cpu(), nrow=4)
                grid_np = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                _wandb.log({"val/clevr_samples": _wandb.Image(grid_np)}, step=step)
            except Exception:
                pass

        return {}


# ── Metrics callback ──────────────────────────────────────────────────────────


class ClevrMetricsCallback(EvalCallbackBase):
    """Sample random CLEVR scenes each eval, generate images, compute metrics.

    Precision/recall, per-attribute accuracy, and spatial relationship accuracy
    are computed using DINO (detection) + SigLIP (classification) and logged
    to WandB.  DINO / SigLIP are lazy-loaded on the first main-process call.

    Args:
        root_dir:       Path to CLEVR dataset root.
        mode:           Conditioning mode — "reduced" or "relative".
        n_samples:      Number of random scenes to evaluate per eval step.
        min_objects:    Minimum number of objects per scene.
        max_objects:    Maximum number of objects per scene.
        batch_size:     Generation batch size.
        split:          Dataset split to sample scenes from.
        reveal_n_objects, reveal_radius_frac, include_centroid_mask, image_size:
            must match data.reveal_n_objects / data.reveal_radius_frac /
            data.include_centroid_mask / data.image_size whenever the bound
            condition encoder needs spatial_conditions (reveal/centroid-mask
            variants) — otherwise generation crashes on spatial_conditions=None.
    """

    def __init__(
        self,
        root_dir: str,
        mode: str = "reduced",
        n_samples: int = 100,
        min_objects: int = 3,
        max_objects: int = 10,
        split: str = "val",
        reveal_n_objects: int = 0,
        reveal_radius_frac: float = 0.12,
        include_centroid_mask: bool = False,
        image_size: int = 256,
    ):
        self._root_dir = root_dir
        self._mode = mode
        self._n_samples = n_samples
        self._split = split
        self._min_objects = min_objects
        self._max_objects = max_objects
        self._reveal_n_objects = reveal_n_objects
        self._reveal_radius_frac = reveal_radius_frac
        self._include_centroid_mask = include_centroid_mask
        self._image_size = image_size

        # Lazy-loaded on first main-process call.
        self._scenes: Optional[list] = None
        self._calibration = None
        self._eval_models = None
        self._image_dir = None
        self._transform = None
        self._H_inv = None

    def _load(self, device: torch.device):
        from eval.evaluate_clevr import calibrate_camera_and_size, JOINT_PROMPTS
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        from transformers import SiglipProcessor, SiglipModel

        self._scenes = _load_scenes(self._root_dir, self._split, self._min_objects, self._max_objects)
        self._calibration = calibrate_camera_and_size(self._root_dir, split=self._split)

        filename_split = "val" if self._split == "validation" else self._split
        self._image_dir = os.path.join(self._root_dir, "CLEVR_v1.0", "images", filename_split)
        self._transform = T.Compose(
            [T.Resize((self._image_size, self._image_size)), T.ToTensor(), T.Normalize([0.5], [0.5])]
        )
        if self._include_centroid_mask:
            self._H_inv = calibrate_mask_projection(self._scenes)

        dino_proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
        dino_mod = AutoModelForZeroShotObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-base"
        ).to(device)
        sig_proc = SiglipProcessor.from_pretrained("google/siglip-base-patch16-224")
        sig_mod = SiglipModel.from_pretrained("google/siglip-base-patch16-224").to(device)

        with torch.no_grad():
            t_inputs = sig_proc(text=JOINT_PROMPTS, padding="max_length", return_tensors="pt").to(device)
            text_embeds = sig_mod.get_text_features(**t_inputs)
            text_embeds /= text_embeds.norm(p=2, dim=-1, keepdim=True)

        self._eval_models = (dino_proc, dino_mod, sig_proc, sig_mod, text_embeds)

    @torch.no_grad()
    def __call__(self, model, dataloader, accelerator, **kwargs) -> dict:
        if not accelerator.is_main_process:
            return {}

        device = accelerator.device
        step = kwargs.get("step", None)

        if self._scenes is None:
            self._load(device)

        from eval.evaluate_clevr import _score_image

        H, l_vec, f_vec, sz_thresh = self._calibration
        dino_proc, dino_mod, sig_proc, sig_mod, text_embeds = self._eval_models

        selected = random.sample(self._scenes, min(self._n_samples, len(self._scenes)))
        m = {k: 0 for k in ("t_req", "t_pred", "v_matches", "hallucinations",
                              "c_col", "c_sh", "c_mat", "c_sz", "perf", "t_rel", "c_rel")}

        cond_samples = [
            _scene_to_data_sample(
                s,
                self._mode,
                image_dir=self._image_dir,
                transform=self._transform,
                reveal_n_objects=self._reveal_n_objects,
                reveal_radius_frac=self._reveal_radius_frac,
                include_centroid_mask=self._include_centroid_mask,
                mask_size=self._image_size // 8,
                H_inv=self._H_inv,
            )
            for s in selected
        ]
        imgs = _generate_images(model, cond_samples, device)

        for img_tensor, scene in tqdm(zip(imgs, selected), "CLEVR metrics eval", total=len(selected)):
            pil_img = Image.fromarray(
                (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            ).resize((ORIG_W, ORIG_H), Image.BILINEAR)

            inc = _score_image(
                pil_img, scene["objects"], scene["relationships"],
                H, l_vec, f_vec, sz_thresh,
                dino_proc, dino_mod, sig_proc, sig_mod, text_embeds,
            )
            for k, v in inc.items():
                m[k] += v

        v = max(1, m["v_matches"])
        return {
            "clevr_precision": m["v_matches"] / max(1, m["t_pred"]),
            "clevr_recall": m["v_matches"] / max(1, m["t_req"]),
            "clevr_color_acc": m["c_col"] / v,
            "clevr_shape_acc": m["c_sh"] / v,
            "clevr_material_acc": m["c_mat"] / v,
            "clevr_size_acc": m["c_sz"] / v,
            "clevr_perfect_gen": m["perf"] / v,
            "clevr_spatial_acc": m["c_rel"] / max(1, m["t_rel"]),
        }

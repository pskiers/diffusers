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

import random
from typing import Optional

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

try:
    import wandb as _wandb
except ImportError:
    _wandb = None

from models.eval_callbacks import EvalCallbackBase
from datasets.data_sample import DataSample, collate_data_samples
from datasets.clevr_dataset import CLEVRHybridDataset, make_tensor_from_scene, ORIG_W, ORIG_H


def _load_scenes(root_dir: str, split: str, min_objects: int, max_objects: int):
    """Load CLEVR scenes in mode='absolute' (no pre-computation), filtered by count."""
    ds = CLEVRHybridDataset(root_dir=root_dir, split=split, mode="absolute", download=False)
    return [s for s in ds.scenes if min_objects <= len(s["objects"]) <= max_objects]


def _scene_to_data_sample(scene: dict, mode: str) -> DataSample:
    """Build a conditioning-only DataSample (no images) from a scene dict."""
    scene = dict(scene)
    scene["mode"] = mode
    cond_tensor, mask = make_tensor_from_scene(scene)
    return DataSample(embedding_conditions=cond_tensor[0], embedding_mask=mask[0])


def _batch_to_device(samples: list[DataSample], device: torch.device) -> DataSample:
    batch = collate_data_samples(samples)
    return DataSample(
        embedding_conditions=batch.embedding_conditions.to(device),
        embedding_mask=batch.embedding_mask.to(device) if batch.embedding_mask is not None else None,
    )


@torch._dynamo.disable
def _generate_images(model, samples: list[DataSample], device: torch.device, cfg_scale: float) -> torch.Tensor:
    """Generate and decode images for a list of conditioning DataSamples."""
    from models.sampling import CFGPredictor, DirectPredictor, SamplingPipeline, SchedulerConfig

    sched_cfg = SchedulerConfig(
        num_train_timesteps=model.scheduler.config.num_train_timesteps,
        beta_schedule=model.scheduler.config.beta_schedule,
        prediction_type=model.scheduler.config.prediction_type,
        num_inference_steps=model.eval_cfg.num_ddim_steps,
        sampler="ddim",
    )
    pipeline = SamplingPipeline(sched_cfg)
    predictor = CFGPredictor(cfg_scale) if cfg_scale > 1.0 else DirectPredictor()

    batch = _batch_to_device(samples, device)
    B = len(samples)
    with torch.no_grad():
        latents = pipeline.sample(model, batch, predictor, shape=(B, *model._noise_shape), device=device)
    return model._decode_for_eval(latents)  # (B, C, H, W) in [0, 1]


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
    """

    def __init__(
        self,
        root_dir: str,
        mode: str = "reduced",
        n_scenes: int = 8,
        min_objects: int = 3,
        max_objects: int = 10,
        split: str = "val",
    ):
        self._mode = mode
        scenes = _load_scenes(root_dir, split, min_objects, max_objects)
        selected = random.sample(scenes, min(n_scenes, len(scenes)))

        self._base_samples = [_scene_to_data_sample(s, mode) for s in selected]

    @torch.no_grad()
    def __call__(self, model, dataloader, accelerator, **kwargs) -> dict:
        if not accelerator.is_main_process or _wandb is None:
            return {}

        step = kwargs.get("step", None)
        imgs = _generate_images(model, self._base_samples, accelerator.device, model.eval_cfg.cfg_scale)

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
    """

    def __init__(
        self,
        root_dir: str,
        mode: str = "reduced",
        n_samples: int = 100,
        min_objects: int = 3,
        max_objects: int = 10,
        batch_size: int = 8,
        split: str = "val",
    ):
        self._root_dir = root_dir
        self._mode = mode
        self._n_samples = n_samples
        self._batch_size = batch_size
        self._split = split
        self._min_objects = min_objects
        self._max_objects = max_objects

        # Lazy-loaded on first main-process call.
        self._scenes: Optional[list] = None
        self._calibration = None
        self._eval_models = None

    def _load(self, device: torch.device):
        from eval.evaluate_clevr import calibrate_camera_and_size, JOINT_PROMPTS
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        from transformers import SiglipProcessor, SiglipModel

        self._scenes = _load_scenes(self._root_dir, self._split, self._min_objects, self._max_objects)
        self._calibration = calibrate_camera_and_size(self._root_dir, split=self._split)

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

        for start in tqdm(range(0, len(selected), self._batch_size), "CLEVR metrics eval"):
            batch_scenes = selected[start:start + self._batch_size]
            cond_samples = [_scene_to_data_sample(s, self._mode) for s in batch_scenes]

            imgs = _generate_images(model, cond_samples, device, model.eval_cfg.cfg_scale)

            for img_tensor, scene in zip(imgs, batch_scenes):
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
        result = {
            "val/clevr_precision": m["v_matches"] / max(1, m["t_pred"]),
            "val/clevr_recall": m["v_matches"] / max(1, m["t_req"]),
            "val/clevr_color_acc": m["c_col"] / v,
            "val/clevr_shape_acc": m["c_sh"] / v,
            "val/clevr_material_acc": m["c_mat"] / v,
            "val/clevr_size_acc": m["c_sz"] / v,
            "val/clevr_perfect_gen": m["perf"] / v,
            "val/clevr_spatial_acc": m["c_rel"] / max(1, m["t_rel"]),
        }

        if _wandb is not None and step is not None:
            _wandb.log(result, step=step)

        return result

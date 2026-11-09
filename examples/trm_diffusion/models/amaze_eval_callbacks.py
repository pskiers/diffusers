from __future__ import annotations

import torch
import wandb as _wandb
from datasets.data_sample import DataSample
from torchvision.utils import make_grid
from tqdm.auto import tqdm

from models.eval_callbacks import EvalCallbackBase


class _AmazeEvalCallbackBase(EvalCallbackBase):
    """Shared DDIM-sampling + AmazeMetrics-scoring harness for Amaze tasks.

    Uses ``eval.amaze_eval.AmazeMetrics``, which scores each generated image and
    accumulates the per-sample metrics. ``AmazeMetrics.return_metrics()`` then
    yields the aggregate; the callback returns the paper's metric set from it:
    coverage, violation, pass (max(0, coverage - violation)) and exact1, one
    attempt per board.

    The AmazeMetrics import is lazy so a missing optional dep (OpenCV/scikit-image)
    just disables scoring (only the sample images are returned) instead of breaking the callback.
    """

    _task: str = ""
    _log_caption: str = "Amaze sample"

    def __init__(
        self,
        num_samples: int = 1000,
        num_log_images: int = 8,
        num_attempts: int = 1,
        crop_to_content: bool = False,
        canvas_color=(0.0, 0.0, 0.0),
        bbox_threshold: float = 0.05,
        bbox_min_frac: float = 0.1,
    ):
        self.num_samples = num_samples
        self.num_log_images = num_log_images
        self.num_attempts = num_attempts  # >1 enables Pass@{num_attempts} (best-of-K, K× sampling cost)
        # crop_to_content: for AmazeScaledDataset targets (solution painted at a random
        # scale/position) — crop each generation to its content bounding box and resize
        # it back before scoring (eval.amaze_eval.crop_to_content). Panels stay raw.
        self.crop_to_content = crop_to_content
        self.canvas_color = tuple(canvas_color)
        self.bbox_threshold = bbox_threshold
        self.bbox_min_frac = bbox_min_frac
        self._scorer_unavailable = False

    def _make_scorer(self, device):
        """A fresh AmazeMetrics accumulator for one eval run (None if unavailable)."""
        if self._scorer_unavailable:
            return None
        try:
            from eval.amaze_eval import AmazeMetrics
        except Exception:
            self._scorer_unavailable = True
            return None
        return AmazeMetrics(device=device, task=self._task)

    @torch.no_grad()
    def __call__(self, model, dataloader, accelerator, **kwargs) -> dict:
        if not accelerator.is_main_process:
            return {}

        # The metrics compare each generated image against a specific puzzle's GT
        def _uses_spatial_conditioning(model) -> bool:
            """True if ``model`` actually reads ``spatial_conditions`` as input."""
            keys: set[str] = set()
            painter_keys = getattr(model, "condition_keys", None)
            if painter_keys:
                keys.update(painter_keys)
            condition_encoder = getattr(model, "condition_encoder", None)
            if condition_encoder is not None:
                keys.update(getattr(condition_encoder, "condition_keys", None) or [])
            return "spatial_conditions" in keys

        uses_conditioning = _uses_spatial_conditioning(model)

        scorer = self._make_scorer(accelerator.device) if uses_conditioning else None
        pipeline = model.sampling_pipeline

        n_total = self.num_samples if scorer is not None else self.num_log_images # Unconditional painter logs a few images
        n_attempts = self.num_attempts if scorer is not None else 1
        n_done = 0
        panels = []

        for batch in tqdm(dataloader, desc=self._log_caption + " eval"):
            if n_done >= n_total:
                break

            batch_size = batch["images"].shape[0]
            B = min(batch_size, n_total - n_done)
            conditions = model._batch_to_sample(batch, accelerator.device)

            # K attempts/sample (K=1 -> plain Pass@1). Distinct seed per attempt so
            # the attempts differ.
            attempts = []
            for k in range(n_attempts):
                # Offset by n_done so different batches get different noise: seeding
                # with just k reused one noise sequence for the whole eval set.
                gen_k = torch.Generator(device=accelerator.device).manual_seed(k * 1_000_003 + n_done)
                sampled = pipeline.sample_one_batch(model, conditions, accelerator.device, generator=gen_k)
                attempts.append(model.decode_for_eval(sampled)[:B].cpu())
            generated = attempts[0]                       # first attempt -> WandB panels
            inputs = torch.stack(attempts, dim=1)         # (B, K, C, H, W)

            if isinstance(batch, DataSample):
                metadata = [batch.metadata[i] if batch.metadata is not None else {} for i in range(B)]
            else:
                metadata = [x or {} for x in batch.get("metadata", [{}])[:B]]

            if scorer is not None:
                if self.crop_to_content:
                    from eval.amaze_eval import crop_to_content
                    inputs = crop_to_content(inputs, self.canvas_color, self.bbox_threshold, self.bbox_min_frac)
                scorer.compute_and_accumulate_metrics(inputs, metadata)

            if _wandb is not None and len(panels) < self.num_log_images:
                cond_imgs = getattr(conditions, "spatial_conditions", None) if uses_conditioning else None
                if cond_imgs is not None:
                    cond_imgs = cond_imgs[:B].detach().float().cpu().clamp(0.0, 1.0)

                for i in range(min(B, self.num_log_images - len(panels))):
                    if cond_imgs is not None:
                        # Side-by-side [generated | condition]. (cond_imgs is already float) so
                        panel = make_grid([generated[i].float(), cond_imgs[i]], nrow=2)
                        caption = f"{self._log_caption} sample {n_done + i} [generated | condition]"
                    else:
                        panel = generated[i]
                        caption = f"{self._log_caption} sample {n_done + i}"
                    panels.append(_wandb.Image(
                        (panel.permute(1, 2, 0).numpy() * 255).astype("uint8"),
                        caption=caption,
                    ))

            n_done += B

        # Logged as val/<key>: the paper's metric set, single attempt per board
        # (exact1; there is no exact5 in-training). AmazeMetrics.return_metrics()
        # also has std/min/max and the MSEs; those are not logged.
        result: dict = {}
        if scorer is not None:
            m = scorer.return_metrics()
            result.update({"coverage": m.get("mean_gt_cell_coverage"), "violation": m.get("mean_background_violation"),
                           "pass": m.get("mean_pass"), "exact1": m.get("mean_exact")})
            result = {k: v for k, v in result.items() if v is not None}
        if panels:
            result["samples"] = panels
        return result


class AmazeEvalCallback(_AmazeEvalCallbackBase):
    """Evaluation callback for Amaze Maze metrics (blue solution-path scoring)."""

    _task = "maze"
    _log_caption = "Amaze maze"


class QueenEvalCallback(_AmazeEvalCallbackBase):
    """Evaluation callback for Amaze Queen metrics (placed-queen cell scoring)."""

    _task = "queens"
    _log_caption = "Amaze queen"

from __future__ import annotations

import torch
import wandb as _wandb
from datasets.data_sample import DataSample
from tqdm.auto import tqdm

from models.eval_callbacks import EvalCallbackBase


class _AmazeEvalCallbackBase(EvalCallbackBase):
    """Shared DDIM-sampling + AmazeMetrics-scoring harness for Amaze tasks.

    Uses ``eval.amaze_eval.AmazeMetrics``, which scores each generated image and
    accumulates the per-sample metrics. ``AmazeMetrics.return_metrics()`` then
    yields the aggregate (global mean/std/min/max per key + generated_samples).
    For both tasks that covers:
    - mse_inside/mse_outside
    - gt_cell_coverage
    - background_violation
    - Pass@1.

    The AmazeMetrics import is lazy so a missing optional dep (OpenCV/scikit-image)
    just disables scoring (reward_available=False) instead of breaking the callback.
    """

    _task: str = ""
    _log_caption: str = "Amaze sample"

    def __init__(self, num_samples: int = 1000, num_log_images: int = 8, num_attempts: int = 1):
        self.num_samples = num_samples
        self.num_log_images = num_log_images
        self.num_attempts = num_attempts  # >1 enables Pass@{num_attempts} (best-of-K, K× sampling cost)
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

        scorer = self._make_scorer(accelerator.device)
        pipeline = model.sampling_pipeline
        n_total = self.num_samples
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
            for k in range(self.num_attempts):
                gen_k = torch.Generator(device=accelerator.device).manual_seed(k)
                sampled = pipeline.sample_one_batch(model, conditions, accelerator.device, generator=gen_k)
                attempts.append(model.decode_for_eval(sampled)[:B].cpu())
            generated = attempts[0]                       # first attempt -> WandB panels
            inputs = torch.stack(attempts, dim=1)         # (B, K, C, H, W)

            if isinstance(batch, DataSample):
                metadata = [batch.metadata[i] if batch.metadata is not None else {} for i in range(B)]
            else:
                metadata = [x or {} for x in batch.get("metadata", [{}])[:B]]

            if scorer is not None:
                scorer.compute_and_accumulate_metrics(inputs, metadata)

            if _wandb is not None and len(panels) < self.num_log_images:
                for i in range(min(B, self.num_log_images - len(panels))):
                    panels.append(_wandb.Image(
                        (generated[i].permute(1, 2, 0).numpy() * 255).astype("uint8"),
                        caption=f"{self._log_caption} sample {n_done + i}",
                    ))

            n_done += B

        result: dict = {"reward_available": scorer is not None}
        if scorer is not None:
            result.update(scorer.return_metrics())      # includes generated_samples
        else:
            result["generated_samples"] = float(n_done)
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

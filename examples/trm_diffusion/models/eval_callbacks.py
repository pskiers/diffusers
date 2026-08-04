"""
models/eval_callbacks.py — Pluggable eval callbacks.

Callbacks have the signature:
    callback(model, dataloader, accelerator, **kwargs) -> dict[str, float]

They are invoked from eval_step after generic loss metrics are computed.
Models accept a list of callbacks and merge all returned dicts.

Sampling parameters (num_inference_steps, cfg_scale, batch_size) are owned
by model.sampling_pipeline rather than individual callbacks.  Callbacks own
only evaluation-specific knobs: num_samples, num_log_images, cell_size, etc.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm.auto import tqdm

import wandb as _wandb
from eval.mnist_eval import (
    evaluate_grids,
    extract_and_resize_sudoku,
    load_or_train_classifier,
    make_panel_image,
    plot_thinker_ts_curve,
    sample_grids,
)
from eval.maze_eval import evaluate_mazes, make_maze_panel_image
from eval.steiner_eval import evaluate_steiner, make_steiner_panel_image
from eval.polygon_eval import evaluate_polygon, make_polygon_panel_image, _orders_equivalent
from eval.ball_drop_eval import evaluate_ball_drop, make_ball_drop_panel_image
from datasets.data_sample import DataSample


def _select_best_of_n(valid: np.ndarray, objective: np.ndarray, maximize: bool):
    """Per-instance best-of-N selection, matching the geometric-solver
    paper's evaluation protocol (arXiv 2510.21697): among the N independent
    candidates generated for one instance, discard invalid ones and keep the
    single best by objective (max area / min length). An instance with zero
    valid candidates is excluded from the ratio (any_valid=False).

    Args:
        valid, objective: (B, N) arrays.

    Returns:
        any_valid: (B,) bool — instance had >=1 valid candidate.
        best_val:  (B,) float — objective of the best valid candidate (NaN if none).
        best_idx:  (B,) int — index in [0, N) of the best valid candidate (-1 if none).
    """
    B, N = valid.shape
    obj = objective.astype(np.float64).copy()
    obj[~valid] = -np.inf if maximize else np.inf
    best_idx = obj.argmax(axis=1) if maximize else obj.argmin(axis=1)
    any_valid = valid.any(axis=1)
    best_val = obj[np.arange(B), best_idx]
    best_val = np.where(any_valid, best_val, np.nan)
    best_idx = np.where(any_valid, best_idx, -1)
    return any_valid, best_val, best_idx


def _weighted_mean(pairs: list) -> float:
    """pairs: list of (value, weight). Returns NaN for an empty list."""
    if not pairs:
        return float("nan")
    total_w = sum(w for _, w in pairs)
    if total_w == 0:
        return float("nan")
    return float(sum(v * w for v, w in pairs) / total_w)


class EvalCallbackBase:
    """Abstract base for eval callbacks."""

    def __call__(self, model, dataloader, accelerator, **kwargs) -> dict:
        raise NotImplementedError


# ── CLEVR / latent-DiT ────────────────────────────────────────────────────────


class ImageGenEvalCallback(EvalCallbackBase):
    """
    DDIM sampling + side-by-side WandB logging for latent painters (CLEVR etc.).

    Uses model.sampling_pipeline for all sampling parameters.  Reads from the
    model at call time:
      model.sampling_pipeline
      model.noise_shape     — (C, H, W) tuple, no batch dim
      model.decode_for_eval(z) -> Tensor in [0, 1]
      model.condition_keys   — DataSample fields to pull from batch
      model.null_condition_sample(sample) -> DataSample

    Args:
        num_log_images: number of images to log to WandB.
    """

    def __init__(self, num_log_images: int = 8):
        self.num_log_images = num_log_images

    @torch.no_grad()
    def __call__(self, model, dataloader, accelerator, **kwargs) -> dict:
        if not accelerator.is_main_process:
            return {}

        step = kwargs.get("step", None)
        device = accelerator.device
        n_log = self.num_log_images
        pipeline = model.sampling_pipeline

        sample_images, gt_images = [], []
        for batch in dataloader:
            if len(sample_images) >= n_log:
                break
            B = min(batch["images"].shape[0], n_log - len(sample_images))

            sample_kwargs = {}
            for k in model.condition_keys:
                val = batch.get(k) if hasattr(batch, "get") else batch.get(k, None)
                if val is not None:
                    sample_kwargs[k] = val[:B].to(device)
            emb_mask = batch.get("embedding_mask") if hasattr(batch, "get") else None
            if emb_mask is not None:
                sample_kwargs["embedding_mask"] = emb_mask[:B].to(device)

            base_sample = DataSample(**sample_kwargs)
            latents = pipeline.sample_one_batch(model, base_sample, device)
            imgs = model.decode_for_eval(latents)
            sample_images.extend(imgs.cpu().unbind(0))

            gt_raw = batch["images"][:B]
            gt_images.extend(model.images_to_log(gt_raw).cpu().unbind(0))

        if sample_images and step is not None:
            try:
                combined = []
                for gt, s in zip(gt_images, sample_images):
                    combined.extend([gt, s])
                grid = make_grid(torch.stack(combined), nrow=4)
                grid_np = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                _wandb.log({"val/samples": _wandb.Image(grid_np)}, step=step)
            except Exception:
                pass

        return {}


# ── MNIST Sudoku ──────────────────────────────────────────────────────────────


class SudokuDDIMEvalCallback(EvalCallbackBase):
    """
    DDIM sampling eval for MNIST Sudoku models with full thinker trajectory
    tracking.  Sampling parameters come from model.sampling_pipeline.

    Runs sample_grids with the thinker condition, evaluates with a digit
    classifier, and logs WandB panels + the thinker-timestep accuracy curve.

    Returns: cell_acc, puzzle_acc and (when include_thinker_metrics=True)
             thinker_cell_acc_best/mean, thinker_puzzle_acc_best/mean,
             thinker_deviation_from_best, painter_dev_from_best/mean_thinker.

    Args:
        classifier_path: path passed to load_or_train_classifier. If None the
                         callback is a no-op.
        cell_size: pixel size of each Sudoku cell.
        include_thinker_metrics: set False to skip thinker accuracy metrics.
        num_samples: total number of samples to evaluate.
        num_log_images: number of images to log to WandB.
        painter_size: pixel size of the full painter image. If None, derived
                      as cell_size * 9.
    """

    def __init__(
        self,
        classifier_path: str | None = None,
        cell_size: int = 16,
        include_thinker_metrics: bool = True,
        num_samples: int = 1000,
        num_log_images: int = 8,
        painter_size: int | None = None,
    ):
        self.cell_size = cell_size
        self.include_thinker_metrics = include_thinker_metrics
        self.num_samples = num_samples
        self.num_log_images = num_log_images
        self.painter_size = painter_size if painter_size is not None else cell_size * 9
        self.eval_clf = None
        if classifier_path is not None:
            self.eval_clf = load_or_train_classifier(classifier_path, None, cell_size, "cuda")
            for p in self.eval_clf.parameters():
                p.requires_grad_(False)

    def __call__(self, model, dataloader, accelerator, **kwargs) -> dict:
        if self.eval_clf is None or not accelerator.is_main_process:
            return {}
        if not hasattr(model, "_batch_to_sample"):
            import logging
            logging.getLogger(__name__).warning(
                "SudokuDDIMEvalCallback: model has no _batch_to_sample method, skipping eval."
            )
            return {}

        device = accelerator.device
        pipeline = model.sampling_pipeline
        painter_size = self.painter_size
        n_total = self.num_samples
        n_log = self.num_log_images
        token_offset = getattr(model, "token_offset", 0)

        all_cell_acc, all_puzzle_acc, all_constraint_acc, all_given_consistent_acc = [], [], [], []
        all_thinker_cell_best, all_thinker_cell_mean = [], []
        all_thinker_puzzle_best, all_thinker_puzzle_mean = [], []
        all_thinker_deviation = []
        all_painter_dev_best, all_painter_dev_mean = [], []
        ts_cell_accs: dict[int, list] = {}
        ts_puzzle_accs: dict[int, list] = {}
        panels: list = []
        n_done = 0

        n_batches = (n_total + pipeline.batch_size - 1) // pipeline.batch_size
        for batch in tqdm(dataloader, "Sampling eval", total=n_batches):
            if n_done >= n_total:
                break
            solutions = batch["solution"]
            given_masks = batch.get("solution_mask")
            B_cur = solutions.shape[0]

            sr = sample_grids(
                model,
                model._batch_to_sample(batch, device),
                num_train_timesteps=model.scheduler.config.num_train_timesteps,
                beta_schedule=model.scheduler.config.beta_schedule,
                prediction_type=model.scheduler.config.prediction_type,
                num_steps=pipeline.num_inference_steps,
                device=device,
                solutions=solutions,
                painter_size=painter_size,
                given_masks=given_masks,
                cfg_scale=pipeline.cfg_scale,
            )
            acc = evaluate_grids(sr["generated"], solutions, self.eval_clf, self.cell_size, given_masks=given_masks)
            all_cell_acc.append(acc["cell_acc"])
            all_puzzle_acc.append(acc["puzzle_acc"])
            all_constraint_acc.append(acc.get("constraint_puzzle_acc", 0.0))
            if acc.get("given_consistent_puzzle_acc") is not None:
                all_given_consistent_acc.append(acc["given_consistent_puzzle_acc"])

            if self.include_thinker_metrics:
                for key, lst in [
                    ("thinker_cell_acc_best", all_thinker_cell_best),
                    ("thinker_cell_acc_mean", all_thinker_cell_mean),
                    ("thinker_puzzle_acc_best", all_thinker_puzzle_best),
                    ("thinker_puzzle_acc_mean", all_thinker_puzzle_mean),
                    ("thinker_deviation_from_best", all_thinker_deviation),
                ]:
                    if key in sr:
                        lst.append(sr[key])

                for t_step, a in sr.get("ts_cell_acc", []):
                    ts_cell_accs.setdefault(t_step, []).append(a)
                for t_step, a in sr.get("ts_puzzle_acc", []):
                    ts_puzzle_accs.setdefault(t_step, []).append(a)

                painter_preds = acc["preds"]
                _gm = given_masks[:B_cur].cpu() if given_masks is not None else None
                for tp_raw, dev_lst in [
                    (sr.get("best_thinker_preds"), all_painter_dev_best),
                    (sr.get("mean_thinker_preds"), all_painter_dev_mean),
                ]:
                    if tp_raw is not None:
                        tp = tp_raw - token_offset
                        N = tp.shape[1]
                        if N > painter_preds.shape[1]:
                            continue
                        diff = painter_preds[:, :N] != tp
                        if _gm is not None:
                            blank = ~_gm[:, :N]
                            dev = diff[blank].float().mean().item() if blank.any() else diff.float().mean().item()
                        else:
                            dev = diff.float().mean().item()
                        dev_lst.append(dev)

            if _wandb is not None and len(panels) < n_log:
                n_new = min(n_log - len(panels), B_cur)
                _sc = batch.get("spatial_conditions")
                conds_vis = _sc if _sc is not None else batch.get("token_conditions")
                if conds_vis is not None and conds_vis.dim() == 4:
                    conds_vis = conds_vis.cpu()
                tp_all = sr.get("best_thinker_preds")
                tt_all = sr.get("best_thinker_ts")
                if tp_all is not None and tp_all.shape[1] != 81:
                    tp_all = None
                sols_np = solutions.cpu().numpy()
                for i in range(n_new):
                    tp = (tp_all[i] - token_offset).numpy() if tp_all is not None else None
                    tt = tt_all[i] if tp_all is not None and tt_all is not None else None
                    cond_img = conds_vis[i] if (conds_vis is not None and conds_vis.dim() == 4) else None
                    panel = make_panel_image(cond_img, sr["generated"][i], sols_np[i], thinker_preds=tp, thinker_t=tt)
                    panels.append(_wandb.Image(panel, caption=f"sample[{n_done + i}]"))

            n_done += B_cur

        result: dict = {
            "cell_acc": float(np.mean(all_cell_acc)),
            "puzzle_acc": float(np.mean(all_puzzle_acc)),
            "constraint_puzzle_acc": float(np.mean(all_constraint_acc)),
        }
        if all_given_consistent_acc:
            result["given_consistent_puzzle_acc"] = float(np.mean(all_given_consistent_acc))
        if self.include_thinker_metrics:
            if all_thinker_cell_best:
                result["thinker_cell_acc_best"] = float(np.mean(all_thinker_cell_best))
                result["thinker_cell_acc_mean"] = float(np.mean(all_thinker_cell_mean))
                result["thinker_puzzle_acc_best"] = float(np.mean(all_thinker_puzzle_best))
                result["thinker_puzzle_acc_mean"] = float(np.mean(all_thinker_puzzle_mean))
                result["thinker_deviation_from_best"] = float(np.mean(all_thinker_deviation))
            if all_painter_dev_best:
                result["painter_dev_from_best_thinker"] = float(np.mean(all_painter_dev_best))
                result["painter_dev_from_mean_thinker"] = float(np.mean(all_painter_dev_mean))
        if panels:
            result["samples"] = panels
        if ts_cell_accs and _wandb is not None:
            result["thinker_vs_timestep"] = _wandb.Image(plot_thinker_ts_curve(ts_cell_accs, ts_puzzle_accs))

        return result


class SudokuRealSolutionCallback(EvalCallbackBase):
    """
    Oracle eval for MNIST Sudoku: DDIM sampling with real solution tokens as the
    condition (teacher forcing) to establish an accuracy upper bound.
    Sampling parameters come from model.sampling_pipeline.

    Only runs when model.has_realsolution_eval is True.

    Returns: real_cell_acc, real_puzzle_acc

    Args:
        classifier_path: same semantics as SudokuDDIMEvalCallback.
        cell_size: pixel size of each Sudoku cell.
        num_samples: total number of samples to evaluate.
        painter_size: pixel size of the full painter image. If None, derived
                      as cell_size * 9.
    """

    def __init__(
        self,
        classifier_path: str | None = None,
        cell_size: int = 16,
        num_samples: int = 1000,
        painter_size: int | None = None,
    ):
        self.cell_size = cell_size
        self.num_samples = num_samples
        self.painter_size = painter_size if painter_size is not None else cell_size * 9
        self.eval_clf = None
        if classifier_path is not None:
            self.eval_clf = load_or_train_classifier(classifier_path, None, cell_size, "cuda")
            for p in self.eval_clf.parameters():
                p.requires_grad_(False)

    def __call__(self, model, dataloader, accelerator, **kwargs) -> dict:
        if not getattr(model, "has_realsolution_eval", False):
            return {}
        if not accelerator.is_main_process:
            return {}
        if not hasattr(model, "_batch_to_sample"):
            import logging
            logging.getLogger(__name__).warning(
                "SudokuRealSolutionCallback: model has no _batch_to_sample method, skipping eval."
            )
            return {}

        device = accelerator.device
        pipeline = model.sampling_pipeline
        painter_size = self.painter_size
        n_total = self.num_samples

        all_real_cell, all_real_puzzle = [], []
        all_real_constraint, all_real_given_consistent = [], []
        n_real = 0
        n_batches = (n_total + pipeline.batch_size - 1) // pipeline.batch_size

        for batch in tqdm(dataloader, "Realsolution eval", total=n_batches):
            if n_real >= n_total:
                break
            solutions = batch["solution"]
            given_masks = batch.get("solution_mask")

            sr_r = sample_grids(
                model,
                model._batch_to_sample(batch, device),
                num_train_timesteps=model.scheduler.config.num_train_timesteps,
                beta_schedule=model.scheduler.config.beta_schedule,
                prediction_type=model.scheduler.config.prediction_type,
                num_steps=pipeline.num_inference_steps,
                device=device,
                solutions=solutions,
                painter_size=painter_size,
                given_masks=given_masks,
                cfg_scale=pipeline.cfg_scale,
            )
            acc_r = evaluate_grids(sr_r["generated"], solutions, self.eval_clf, self.cell_size, given_masks=given_masks)
            all_real_cell.append(acc_r["cell_acc"])
            all_real_puzzle.append(acc_r["puzzle_acc"])
            all_real_constraint.append(acc_r.get("constraint_puzzle_acc", 0.0))
            if acc_r.get("given_consistent_puzzle_acc") is not None:
                all_real_given_consistent.append(acc_r["given_consistent_puzzle_acc"])
            n_real += solutions.shape[0]

        result = {
            "real_cell_acc": float(np.mean(all_real_cell)),
            "real_puzzle_acc": float(np.mean(all_real_puzzle)),
            "real_constraint_puzzle_acc": float(np.mean(all_real_constraint)),
        }
        if all_real_given_consistent:
            result["real_given_consistent_puzzle_acc"] = float(np.mean(all_real_given_consistent))
        return result


class SudokuEvalCallback(EvalCallbackBase):
    """
    Simple DDIM sampling eval for MNIST Sudoku: sample images then evaluate.

    No thinker trajectory tracking.  Uses model.sampling_pipeline for all
    sampling parameters.  Logs WandB panels (puzzle | generated | solution).

    Returns: cell_acc, puzzle_acc

    Args:
        classifier_path: path passed to load_or_train_classifier. If None the
                         callback is a no-op.
        cell_size: pixel size of each Sudoku cell.
        num_samples: total number of samples to evaluate.
        num_log_images: number of panel images to log to WandB.
        painter_size: pixel size of the full painter image. If None, derived
                      as cell_size * 9.
    """

    def __init__(
        self,
        classifier_path: str | None = None,
        cell_size: int = 16,
        num_samples: int = 1000,
        num_log_images: int = 8,
        painter_size: int | None = None,
    ):
        self.cell_size = cell_size
        self.num_samples = num_samples
        self.num_log_images = num_log_images
        self.painter_size = painter_size if painter_size is not None else cell_size * 9
        self.eval_clf = None
        if classifier_path is not None:
            self.eval_clf = load_or_train_classifier(classifier_path, None, cell_size, "cuda")
            for p in self.eval_clf.parameters():
                p.requires_grad_(False)

    def _prepare_for_eval(self, generated: torch.Tensor) -> torch.Tensor:
        """Hook for subclasses to preprocess generated images before classification.

        Only affects the accuracy computation — logged panel images always show
        the raw model output.
        """
        return generated

    def __call__(self, model, dataloader, accelerator, **kwargs) -> dict:
        if self.eval_clf is None or not accelerator.is_main_process:
            return {}
        if not hasattr(model, "_batch_to_sample"):
            import logging
            logging.getLogger(__name__).warning(
                "SudokuEvalCallback: model has no _batch_to_sample method, skipping eval."
            )
            return {}

        device = accelerator.device
        pipeline = model.sampling_pipeline
        n_total = self.num_samples
        n_log = self.num_log_images
        token_offset = getattr(model, "token_offset", 0)

        all_cell_acc, all_puzzle_acc, all_constraint_acc, all_given_consistent_acc = [], [], [], []
        panels: list = []
        n_done = 0

        n_batches = (n_total + pipeline.batch_size - 1) // pipeline.batch_size
        for batch in tqdm(dataloader, "Sudoku eval", total=n_batches):
            if n_done >= n_total:
                break
            solutions = batch["solution"]
            given_masks = batch.get("solution_mask")
            B_cur = solutions.shape[0]

            conditions = model._batch_to_sample(batch, device)
            generated = pipeline.sample_one_batch(model, conditions, device)
            generated = model.decode_for_eval(generated)  # (B, 1, H, W) in [0, 1]
            eval_images = self._prepare_for_eval(generated)

            acc = evaluate_grids(eval_images, solutions, self.eval_clf, self.cell_size, given_masks=given_masks)
            all_cell_acc.append(acc["cell_acc"])
            all_puzzle_acc.append(acc["puzzle_acc"])
            all_constraint_acc.append(acc.get("constraint_puzzle_acc", 0.0))
            if acc.get("given_consistent_puzzle_acc") is not None:
                all_given_consistent_acc.append(acc["given_consistent_puzzle_acc"])

            if _wandb is not None and len(panels) < n_log:
                n_new = min(n_log - len(panels), B_cur)
                _sc = batch.get("spatial_conditions")
                conds_vis = _sc if _sc is not None else batch.get("token_conditions")
                if conds_vis is not None and conds_vis.dim() == 4:
                    conds_vis = conds_vis.cpu()
                sols_np = solutions.cpu().numpy()
                gen_np = generated.cpu()
                for i in range(n_new):
                    cond_img = conds_vis[i] if (conds_vis is not None and conds_vis.dim() == 4) else None
                    panel = make_panel_image(cond_img, gen_np[i], sols_np[i])
                    panels.append(_wandb.Image(panel, caption=f"sample[{n_done + i}]"))

            n_done += B_cur

        result: dict = {
            "cell_acc": float(np.mean(all_cell_acc)),
            "puzzle_acc": float(np.mean(all_puzzle_acc)),
            "constraint_puzzle_acc": float(np.mean(all_constraint_acc)),
        }
        if all_given_consistent_acc:
            result["given_consistent_puzzle_acc"] = float(np.mean(all_given_consistent_acc))
        if panels:
            result["samples"] = panels

        return result


class ScaledSudokuEvalCallback(SudokuEvalCallback):
    """SudokuEvalCallback variant for MNISTSudokuScaledDataset targets.

    There, the solved grid is pasted onto a black canvas at a random
    scale/offset, so the generated image can't be evaluated by unfolding it
    into 9×9 cells directly — the true scale/position isn't known at
    generation time. Crops each generated image to its non-background content
    bounding box and resizes back to painter_size before running the standard
    cell classifier.

    Args (in addition to SudokuEvalCallback's):
        bbox_threshold: pixel intensity above which a pixel counts as content
                        when finding the crop bounding box.
    """

    def __init__(self, *args, bbox_threshold: float = 0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.bbox_threshold = bbox_threshold

    def _prepare_for_eval(self, generated: torch.Tensor) -> torch.Tensor:
        return extract_and_resize_sudoku(generated, self.painter_size, self.bbox_threshold)


# ── Maze ──────────────────────────────────────────────────────────────────────


class MazeEvalCallback(EvalCallbackBase):
    """
    DDIM sampling eval for Maze models: sample images, then evaluate against
    the maze's own wall structure. No learned classifier is needed — unlike
    Sudoku's MNIST digit cells, MazeDataset renders every cell as a flat
    color from a small fixed palette, so cells are classified by nearest-
    color matching (see eval/maze_eval.py).

    Returns: cell_acc, puzzle_acc, constraint_puzzle_acc (see
    eval.maze_eval.evaluate_mazes for exact definitions).

    Args:
        grid_size, cell_size: must match the dataset's rendering parameters.
        num_samples: total number of samples to evaluate.
        num_log_images: number of panel images to log to WandB.
    """

    def __init__(
        self,
        grid_size: int = 8,
        cell_size: int = 16,
        num_samples: int = 1000,
        num_log_images: int = 8,
    ):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.num_samples = num_samples
        self.num_log_images = num_log_images

    def __call__(self, model, dataloader, accelerator, **kwargs) -> dict:
        if not accelerator.is_main_process:
            return {}
        if not hasattr(model, "_batch_to_sample"):
            import logging
            logging.getLogger(__name__).warning(
                "MazeEvalCallback: model has no _batch_to_sample method, skipping eval."
            )
            return {}

        device = accelerator.device
        pipeline = model.sampling_pipeline
        n_total = self.num_samples
        n_log = self.num_log_images

        per_sample_cell_acc, per_sample_exact, per_sample_valid, per_sample_size = [], [], [], []
        panels: list = []
        n_done = 0

        n_batches = (n_total + pipeline.batch_size - 1) // pipeline.batch_size
        for batch in tqdm(dataloader, "Maze eval", total=n_batches):
            if n_done >= n_total:
                break
            B_cur = batch["images"].shape[0]

            conditions = model._batch_to_sample(batch, device)
            generated = pipeline.sample_one_batch(model, conditions, device)
            generated = model.decode_for_eval(generated)  # (B, 3, H, W) in [0, 1]

            acc = evaluate_mazes(
                generated,
                batch["spatial_conditions"].to(device),
                batch["solution"].to(device),
                batch["solution_mask"].to(device),
                batch["token_conditions"].to(device),
                self.grid_size,
                self.cell_size,
            )
            per_sample_cell_acc.append(acc["per_sample_cell_acc"])
            per_sample_exact.append(acc["per_sample_exact"])
            per_sample_valid.append(acc["per_sample_valid"])
            per_sample_size.append(acc["per_sample_active_size"])

            if _wandb is not None and len(panels) < n_log:
                n_new = min(n_log - len(panels), B_cur)
                sol_np = batch["solution"].numpy()
                mask_np = batch["solution_mask"].numpy()
                tok_np = batch["token_conditions"].numpy()
                cond_cpu = batch["spatial_conditions"].cpu()
                gen_cpu = generated.cpu()
                for i in range(n_new):
                    panel = make_maze_panel_image(
                        cond_cpu[i], gen_cpu[i], tok_np[i], sol_np[i], mask_np[i],
                        self.grid_size, self.cell_size,
                    )
                    panels.append(_wandb.Image(panel, caption=f"sample[{n_done + i}]"))

            n_done += B_cur

        cell_acc = np.concatenate(per_sample_cell_acc)
        exact = np.concatenate(per_sample_exact)
        valid = np.concatenate(per_sample_valid)
        size = np.concatenate(per_sample_size)

        result: dict = {
            "cell_acc": float(cell_acc.mean()),
            "puzzle_acc": float(exact.mean()),
            "constraint_puzzle_acc": float(valid.mean()),
        }
        # Difficulty breakdown: same three metrics, restricted to each active
        # maze size seen in this eval pass — e.g. constraint_puzzle_acc_size3
        # (matches the published SketchVLM benchmark's scale) vs
        # constraint_puzzle_acc_size8 (hardest size trained on), so a curriculum
        # over min/max_active_size doesn't hide a harder-size regression inside
        # one pooled average.
        for s in sorted(set(size.tolist())):
            m = size == s
            result[f"cell_acc_size{s}"] = float(cell_acc[m].mean())
            result[f"puzzle_acc_size{s}"] = float(exact[m].mean())
            result[f"constraint_puzzle_acc_size{s}"] = float(valid[m].mean())
        if panels:
            result["samples"] = panels
        return result


# ── Steiner Tree ──────────────────────────────────────────────────────────────


class SteinerEvalCallback(EvalCallbackBase):
    """
    Best-of-N DDIM sampling eval for Steiner Tree models, matching the
    original paper's protocol (arXiv 2510.21697, Table 2): for each puzzle
    instance, draw `num_candidates` independent noise samples, extract a
    graph from each (vertex/edge detection — see eval/steiner_eval.py),
    discard invalid ones (not a tree, disconnected, or missing a terminal),
    and score the single best (shortest) survivor. No learned classifier is
    needed — like Maze, pixels are read directly off the known rendering
    scheme.

    Metrics, one group per evaluated point range (see extra_eval_sets),
    logged as "{range}/valid_rate", "{range}/ratio_mean", "{range}/ratio_std"
    — e.g. "10-20/valid_rate", "21-30/ratio_mean" — no other metrics are
    logged (kept deliberately minimal):
      {range}/valid_rate     — fraction of instances with >=1 valid
                               candidate among the N generated.
      {range}/ratio_mean/std — mean/std of (best valid candidate's length /
                               exact optimal length), over instances with
                               >=1 valid candidate. The exact optimal length
                               is looked up from the dataset's own
                               generation-time record via
                               dataloader.dataset.optimal_length_for(puzzle_id)
                               (see SteinerTreeDataset), not re-solved or
                               re-derived lossily from a rendered image.

    The primary (in-distribution) dataloader is evaluated as its own named
    range (primary_range_name, default "10-20") exactly like extra_eval_sets
    entries — no special-cased unsuffixed keys. extra_eval_sets adds
    out-of-distribution point-range test sets (the paper's own 21-30/31-40/
    41-50 generalization splits, evaluated with the *same trained model* —
    no retraining needed, since embedding_conditions/embedding_mask are
    eval-only fields never fed to the network; the model only ever sees the
    rendered condition image regardless of point count). Each entry:
        {name, hf_filename, max_points, hf_repo (optional),
         num_samples (optional, defaults to this callback's num_samples)}

    Args:
        image_size: must match the dataset's rendering resolution.
        num_samples: number of puzzle instances to evaluate on the primary
                     (in-distribution) dataloader.
        num_candidates: independent noise samples per instance for the
                        best-of-N selection (paper uses 10). Lower this (or
                        num_samples) if periodic training-time eval is too
                        slow — the model batch actually sent through the
                        painter per step is chunked to roughly
                        sampling_pipeline.batch_size // num_candidates
                        instances at a time, so raising num_candidates alone
                        doesn't blow up GPU memory, only wall-clock.
        num_log_images: WandB panel images (primary dataloader only), shown
                        using each logged instance's best-of-N candidate.
        primary_range_name: label for the primary dataloader's metric group.
        extra_eval_sets: optional list of OOD point-range test-set specs.
    """

    def __init__(
        self,
        image_size: int = 128,
        num_samples: int = 1000,
        num_candidates: int = 10,
        num_log_images: int = 8,
        primary_range_name: str = "10-20",
        extra_eval_sets: Optional[list] = None,
    ):
        self.image_size = image_size
        self.num_samples = num_samples
        self.num_candidates = num_candidates
        self.num_log_images = num_log_images
        self.primary_range_name = primary_range_name
        self._extra_specs = list(extra_eval_sets) if extra_eval_sets else []
        self._extra_dataloaders = None  # built lazily on first __call__

    def _build_extra_dataloaders(self, batch_size: int) -> list:
        from datasets.steiner_dataset import SteinerTreeDataset

        loaders = []
        for spec in self._extra_specs:
            ds_kwargs = {"image_size": self.image_size, "max_points": spec.get("max_points", 20)}
            if spec.get("hf_repo") is not None:
                ds_kwargs["hf_repo"] = spec["hf_repo"]
            if spec.get("hf_filename") is not None:
                ds_kwargs["hf_filename"] = spec["hf_filename"]
            ds = SteinerTreeDataset(**ds_kwargs)
            dl = DataLoader(
                ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=SteinerTreeDataset.collate_fn
            )
            loaders.append((spec["name"], spec.get("num_samples", self.num_samples), dl))
        return loaders

    def __call__(self, model, dataloader, accelerator, **kwargs) -> dict:
        if not accelerator.is_main_process:
            return {}
        if not hasattr(model, "_batch_to_sample"):
            import logging
            logging.getLogger(__name__).warning(
                "SteinerEvalCallback: model has no _batch_to_sample method, skipping eval."
            )
            return {}

        if self._extra_dataloaders is None:
            self._extra_dataloaders = self._build_extra_dataloaders(dataloader.batch_size)

        result = self._eval_one(
            model, dataloader, accelerator, self.num_samples, self.primary_range_name, log_panels=True
        )
        for name, n_samples, extra_dl in self._extra_dataloaders:
            result.update(self._eval_one(model, extra_dl, accelerator, n_samples, name, log_panels=False))
        return result

    def _eval_one(self, model, dataloader, accelerator, n_total: int, range_name: str, log_panels: bool) -> dict:
        device = accelerator.device
        pipeline = model.sampling_pipeline
        n_log = self.num_log_images if log_panels else 0
        dataset = getattr(dataloader, "dataset", None)
        length_lookup = getattr(dataset, "optimal_length_for", None)
        # Model batch = chunk * num_candidates instances; keeps GPU memory
        # roughly matching the pipeline's configured batch size regardless
        # of how many candidates-per-instance we're generating.
        chunk = max(1, pipeline.batch_size // self.num_candidates)

        all_any_valid: list = []
        all_ratios: list = []
        panels: list = []
        n_done = 0

        n_batches = (n_total + dataloader.batch_size - 1) // dataloader.batch_size
        for batch in tqdm(dataloader, f"Steiner eval [{range_name}]", total=n_batches):
            if n_done >= n_total:
                break
            B_full = batch["images"].shape[0]

            for start in range(0, B_full, chunk):
                end = min(start + chunk, B_full)
                sub = batch.slice(start, end)
                B = end - start

                conditions = model._batch_to_sample(sub, device)
                gen_bn = pipeline.sample_best_of_n(model, conditions, device, self.num_candidates)  # (B, N, C, H, W)
                N = gen_bn.shape[1]
                gen_flat = model.decode_for_eval(gen_bn.reshape(B * N, *gen_bn.shape[2:]))  # (B*N, 1, H, W)

                emb_cond_rep = sub["embedding_conditions"].to(device).repeat_interleave(N, dim=0)
                emb_mask_rep = sub["embedding_mask"].to(device).repeat_interleave(N, dim=0)
                acc = evaluate_steiner(gen_flat, emb_cond_rep, emb_mask_rep, self.image_size)

                valid_bn = np.asarray(acc["per_sample_valid"]).reshape(B, N)
                length_bn = np.asarray(acc["per_sample_length"]).reshape(B, N)
                any_valid, best_length, best_idx = _select_best_of_n(valid_bn, length_bn, maximize=False)
                all_any_valid.append(any_valid)

                if length_lookup is not None:
                    puzzle_ids = sub["puzzle_id"].cpu().tolist()
                    for i in range(B):
                        if not any_valid[i]:
                            continue
                        opt_len = length_lookup(puzzle_ids[i])
                        if opt_len is not None and opt_len > 0:
                            all_ratios.append(float(best_length[i]) / float(opt_len))

                if _wandb is not None and len(panels) < n_log:
                    n_new = min(n_log - len(panels), B)
                    cond_cpu = sub["spatial_conditions"].cpu()
                    true_cpu = sub["images"].cpu()
                    gen_bn_cpu = gen_flat.reshape(B, N, *gen_flat.shape[1:]).cpu()
                    for i in range(n_new):
                        idx = int(best_idx[i]) if best_idx[i] >= 0 else 0
                        panel = make_steiner_panel_image(cond_cpu[i], gen_bn_cpu[i, idx], true_cpu[i])
                        panels.append(_wandb.Image(panel, caption=f"sample[{n_done + i}]"))

            n_done += B_full

        any_valid_all = np.concatenate(all_any_valid) if all_any_valid else np.zeros(0, dtype=bool)
        result: dict = {
            f"{range_name}/valid_rate": float(any_valid_all.mean()) if len(any_valid_all) else float("nan"),
        }
        if all_ratios:
            result[f"{range_name}/ratio_mean"] = float(np.mean(all_ratios))
            result[f"{range_name}/ratio_std"] = float(np.std(all_ratios))
        if panels:
            result["samples"] = panels
        return result


# ── Max-Area Polygon ───────────────────────────────────────────────────────────


class PolygonEvalCallback(EvalCallbackBase):
    """
    Best-of-N DDIM sampling eval for Max-Area Polygon models, matching the
    original paper's protocol (arXiv 2510.21697, Table 3): for each puzzle
    instance, draw `num_candidates` independent noise samples, detect which
    known-point-pairs form edges in each (distance-transform-based — see
    eval/polygon_eval.py), discard candidates that don't form a valid simple
    polygon through every point (Hamiltonian cycle + no self-intersections,
    checked from exact known coordinates, not pixels), and score the single
    best (largest-area) survivor.

    Metrics, one group per evaluated point range (see extra_eval_sets),
    logged as "{range}/valid_rate", "{range}/ratio_mean", "{range}/ratio_std",
    "{range}/opt_rate" — e.g. "7-12/valid_rate", "13-15/opt_rate" — no other
    metrics are logged (kept deliberately minimal):
      {range}/valid_rate     — fraction of instances with >=1 valid
                               candidate among the N generated.
      {range}/ratio_mean/std — mean/std of (best valid candidate's area /
                               exact optimal area), over instances with >=1
                               valid candidate, looked up via
                               dataloader.dataset.optimal_area_for(puzzle_id)
                               (see PolygonDataset) — not re-solved or
                               re-derived lossily from a rendered image.
      {range}/opt_rate       — fraction of instances (with >=1 valid
                               candidate) whose best candidate's recovered
                               vertex order is *exactly* the optimal polygon
                               (up to rotation/reflection — see
                               eval.polygon_eval._orders_equivalent and
                               PolygonDataset.optimal_order_for), not just
                               close in area — matches the paper's
                               "Opt. Rate" column.

    The primary (in-distribution) dataloader is evaluated as its own named
    range (primary_range_name, default "7-12") exactly like extra_eval_sets
    entries — no special-cased unsuffixed keys. extra_eval_sets adds
    out-of-distribution point-range test sets (the paper's own 13-15
    generalization split, evaluated with the *same trained model* — no
    retraining needed, since embedding_conditions/embedding_mask are
    eval-only fields never fed to the network). Each entry:
        {name, hf_filename, max_points, hf_repo (optional),
         num_samples (optional, defaults to this callback's num_samples)}

    Args:
        image_size: must match the dataset's rendering resolution.
        num_samples: number of puzzle instances to evaluate on the primary
                     (in-distribution) dataloader.
        num_candidates: independent noise samples per instance for the
                        best-of-N selection (paper uses 10). Lower this (or
                        num_samples) if periodic training-time eval is too
                        slow — see SteinerEvalCallback's docstring for the
                        chunking rationale (identical here).
        num_log_images: WandB panel images (primary dataloader only), shown
                        using each logged instance's best-of-N candidate.
        primary_range_name: label for the primary dataloader's metric group.
        extra_eval_sets: optional list of OOD point-range test-set specs.
    """

    def __init__(
        self,
        image_size: int = 128,
        num_samples: int = 1000,
        num_candidates: int = 10,
        num_log_images: int = 8,
        primary_range_name: str = "7-12",
        extra_eval_sets: Optional[list] = None,
    ):
        self.image_size = image_size
        self.num_samples = num_samples
        self.num_candidates = num_candidates
        self.num_log_images = num_log_images
        self.primary_range_name = primary_range_name
        self._extra_specs = list(extra_eval_sets) if extra_eval_sets else []
        self._extra_dataloaders = None  # built lazily on first __call__

    def _build_extra_dataloaders(self, batch_size: int) -> list:
        from datasets.polygon_dataset import PolygonDataset

        loaders = []
        for spec in self._extra_specs:
            ds_kwargs = {"image_size": self.image_size, "max_points": spec.get("max_points", 12)}
            if spec.get("hf_repo") is not None:
                ds_kwargs["hf_repo"] = spec["hf_repo"]
            if spec.get("hf_filename") is not None:
                ds_kwargs["hf_filename"] = spec["hf_filename"]
            ds = PolygonDataset(**ds_kwargs)
            dl = DataLoader(
                ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=PolygonDataset.collate_fn
            )
            loaders.append((spec["name"], spec.get("num_samples", self.num_samples), dl))
        return loaders

    def __call__(self, model, dataloader, accelerator, **kwargs) -> dict:
        if not accelerator.is_main_process:
            return {}
        if not hasattr(model, "_batch_to_sample"):
            import logging
            logging.getLogger(__name__).warning(
                "PolygonEvalCallback: model has no _batch_to_sample method, skipping eval."
            )
            return {}

        if self._extra_dataloaders is None:
            self._extra_dataloaders = self._build_extra_dataloaders(dataloader.batch_size)

        result = self._eval_one(
            model, dataloader, accelerator, self.num_samples, self.primary_range_name, log_panels=True
        )
        for name, n_samples, extra_dl in self._extra_dataloaders:
            result.update(self._eval_one(model, extra_dl, accelerator, n_samples, name, log_panels=False))
        return result

    def _eval_one(self, model, dataloader, accelerator, n_total: int, range_name: str, log_panels: bool) -> dict:
        device = accelerator.device
        pipeline = model.sampling_pipeline
        n_log = self.num_log_images if log_panels else 0
        dataset = getattr(dataloader, "dataset", None)
        area_lookup = getattr(dataset, "optimal_area_for", None)
        order_lookup = getattr(dataset, "optimal_order_for", None)
        chunk = max(1, pipeline.batch_size // self.num_candidates)

        all_any_valid: list = []
        all_ratios: list = []
        all_exact_match: list = []
        panels: list = []
        n_done = 0

        n_batches = (n_total + dataloader.batch_size - 1) // dataloader.batch_size
        for batch in tqdm(dataloader, f"Polygon eval [{range_name}]", total=n_batches):
            if n_done >= n_total:
                break
            B_full = batch["images"].shape[0]

            for start in range(0, B_full, chunk):
                end = min(start + chunk, B_full)
                sub = batch.slice(start, end)
                B = end - start

                conditions = model._batch_to_sample(sub, device)
                gen_bn = pipeline.sample_best_of_n(model, conditions, device, self.num_candidates)  # (B, N, C, H, W)
                N = gen_bn.shape[1]
                gen_flat = model.decode_for_eval(gen_bn.reshape(B * N, *gen_bn.shape[2:]))  # (B*N, 1, H, W)

                emb_cond_rep = sub["embedding_conditions"].to(device).repeat_interleave(N, dim=0)
                emb_mask_rep = sub["embedding_mask"].to(device).repeat_interleave(N, dim=0)
                acc = evaluate_polygon(gen_flat, emb_cond_rep, emb_mask_rep, self.image_size)

                valid_bn = np.asarray(acc["per_sample_valid"]).reshape(B, N)
                area_bn = np.asarray(acc["per_sample_area"]).reshape(B, N)
                orders_bn = [acc["per_sample_order"][k * N : (k + 1) * N] for k in range(B)]
                any_valid, best_area, best_idx = _select_best_of_n(valid_bn, area_bn, maximize=True)
                all_any_valid.append(any_valid)

                puzzle_ids = sub["puzzle_id"].cpu().tolist()
                for i in range(B):
                    if not any_valid[i]:
                        continue
                    if area_lookup is not None:
                        opt_area = area_lookup(puzzle_ids[i])
                        if opt_area is not None and opt_area > 0:
                            all_ratios.append(float(best_area[i]) / float(opt_area))
                    if order_lookup is not None:
                        true_order = order_lookup(puzzle_ids[i])
                        best_order = orders_bn[i][int(best_idx[i])]
                        if true_order is not None and best_order is not None:
                            all_exact_match.append(_orders_equivalent(best_order, true_order))

                if _wandb is not None and len(panels) < n_log:
                    n_new = min(n_log - len(panels), B)
                    cond_cpu = sub["spatial_conditions"].cpu()
                    true_cpu = sub["images"].cpu()
                    gen_bn_cpu = gen_flat.reshape(B, N, *gen_flat.shape[1:]).cpu()
                    for i in range(n_new):
                        idx = int(best_idx[i]) if best_idx[i] >= 0 else 0
                        panel = make_polygon_panel_image(cond_cpu[i], gen_bn_cpu[i, idx], true_cpu[i])
                        panels.append(_wandb.Image(panel, caption=f"sample[{n_done + i}]"))

            n_done += B_full

        any_valid_all = np.concatenate(all_any_valid) if all_any_valid else np.zeros(0, dtype=bool)
        result: dict = {
            f"{range_name}/valid_rate": float(any_valid_all.mean()) if len(any_valid_all) else float("nan"),
        }
        if all_ratios:
            result[f"{range_name}/ratio_mean"] = float(np.mean(all_ratios))
            result[f"{range_name}/ratio_std"] = float(np.std(all_ratios))
        if order_lookup is not None and all_exact_match:
            result[f"{range_name}/opt_rate"] = float(np.mean(all_exact_match))
        if panels:
            result["samples"] = panels
        return result


# ── Ball Drop ───────────────────────────────────────────────────────────────


class BallDropEvalCallback(EvalCallbackBase):
    """
    Best-of-N DDIM sampling eval for Ball Drop models: sample images, extract
    the drawn solution line(s) (pixel-color based — see eval/ball_drop_eval.py),
    re-simulate physics from the instance's recorded ball start position, and
    check whether the ball settles in the recorded target bucket.

    Unlike SteinerEvalCallback/PolygonEvalCallback, there is no ratio/
    optimality dimension — this is a reachability/success task (any line
    placement that lands the ball in the target bucket is equally valid, see
    datasets/ball_drop_generation.py's docstring), so the only per-instance
    question is pass/fail, not "how good."

    Main metric:
      valid_rate — fraction of instances with >=1 candidate (out of
                   num_candidates independent noise samples) whose drawn
                   line(s) route the ball into the target bucket.

    Diagnostic metrics (per individual generated candidate, not best-of-N):
      constraint_puzzle_acc, settled_rate (ball settled at all, regardless
      of bucket — distinguishes "wrong bucket" from "physics never
      resolved"), mean_lines_extracted.

    extra_eval_sets adds additional held-out test sets with metrics suffixed
    `_{name}` — in particular the real loganbolton/sketchvlm-physics-ball-drop
    benchmark (see datasets/ball_drop_real_data.py), a genuine external
    generalization check unlike Steiner/Polygon's synthetic OOD point
    ranges: those 198 instances were solved by SketchVLM's own (undisclosed)
    physics generator, not ours, and re-simulating them with our pymunk
    parameters reproduces their recorded outcome ~88.9% of the time — see
    that module's docstring for the validation. Each entry:
        {name, hf_filename, hf_repo (optional),
         num_samples (optional, defaults to this callback's num_samples)}

    Args:
        image_size: must match the dataset's rendering resolution.
        num_samples: number of puzzle instances to evaluate on the primary
                     (in-distribution) dataloader.
        num_candidates: independent noise samples per instance for the
                        best-of-N selection. Lower this (or num_samples) if
                        periodic training-time eval is too slow — chunking
                        rationale identical to SteinerEvalCallback's.
        num_log_images: WandB panel images (primary dataloader only), shown
                        using each logged instance's best-of-N candidate.
        extra_eval_sets: optional list of held-out test-set specs.
    """

    def __init__(
        self,
        image_size: int = 128,
        num_samples: int = 1000,
        num_candidates: int = 10,
        num_log_images: int = 8,
        extra_eval_sets: Optional[list] = None,
    ):
        self.image_size = image_size
        self.num_samples = num_samples
        self.num_candidates = num_candidates
        self.num_log_images = num_log_images
        self._extra_specs = list(extra_eval_sets) if extra_eval_sets else []
        self._extra_dataloaders = None  # built lazily on first __call__

    def _build_extra_dataloaders(self, batch_size: int) -> list:
        from datasets.ball_drop_dataset import BallDropDataset

        loaders = []
        for spec in self._extra_specs:
            ds_kwargs = {"image_size": self.image_size}
            if spec.get("hf_repo") is not None:
                ds_kwargs["hf_repo"] = spec["hf_repo"]
            if spec.get("hf_filename") is not None:
                ds_kwargs["hf_filename"] = spec["hf_filename"]
            ds = BallDropDataset(**ds_kwargs)
            dl = DataLoader(
                ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=BallDropDataset.collate_fn
            )
            loaders.append((spec["name"], spec.get("num_samples", self.num_samples), dl))
        return loaders

    def __call__(self, model, dataloader, accelerator, **kwargs) -> dict:
        if not accelerator.is_main_process:
            return {}
        if not hasattr(model, "_batch_to_sample"):
            import logging
            logging.getLogger(__name__).warning(
                "BallDropEvalCallback: model has no _batch_to_sample method, skipping eval."
            )
            return {}

        if self._extra_dataloaders is None:
            self._extra_dataloaders = self._build_extra_dataloaders(dataloader.batch_size)

        result = self._eval_one(model, dataloader, accelerator, self.num_samples, suffix="", log_panels=True)
        for name, n_samples, extra_dl in self._extra_dataloaders:
            result.update(
                self._eval_one(model, extra_dl, accelerator, n_samples, suffix=f"_{name}", log_panels=False)
            )
        return result

    def _eval_one(self, model, dataloader, accelerator, n_total: int, suffix: str, log_panels: bool) -> dict:
        device = accelerator.device
        pipeline = model.sampling_pipeline
        n_log = self.num_log_images if log_panels else 0
        dataset = getattr(dataloader, "dataset", None)
        spec_lookup = getattr(dataset, "physics_spec_for", None)
        chunk = max(1, pipeline.batch_size // self.num_candidates)

        weighted_constraint, weighted_settled, weighted_lines = [], [], []
        all_any_valid: list = []
        panels: list = []
        n_done = 0

        if spec_lookup is None:
            import logging
            logging.getLogger(__name__).warning(
                "BallDropEvalCallback: dataset has no physics_spec_for method, skipping eval."
            )
            return {}

        n_batches = (n_total + dataloader.batch_size - 1) // dataloader.batch_size
        for batch in tqdm(dataloader, "Ball drop eval" + suffix, total=n_batches):
            if n_done >= n_total:
                break
            B_full = batch["images"].shape[0]

            for start in range(0, B_full, chunk):
                end = min(start + chunk, B_full)
                sub = batch.slice(start, end)
                B = end - start

                puzzle_ids = sub["puzzle_id"].cpu().tolist()
                specs = [spec_lookup(pid) for pid in puzzle_ids]
                ball_start_x = np.array([s["ball_start_x"] for s in specs], dtype=np.float64)
                target_bucket = np.array([s["target_bucket"] for s in specs], dtype=np.int64)

                conditions = model._batch_to_sample(sub, device)
                gen_bn = pipeline.sample_best_of_n(model, conditions, device, self.num_candidates)  # (B, N, C, H, W)
                N = gen_bn.shape[1]
                gen_flat = model.decode_for_eval(gen_bn.reshape(B * N, *gen_bn.shape[2:]))  # (B*N, 3, H, W)

                ball_start_x_rep = np.repeat(ball_start_x, N)
                target_bucket_rep = np.repeat(target_bucket, N)
                acc = evaluate_ball_drop(gen_flat, ball_start_x_rep, target_bucket_rep, self.image_size)

                w = B * N
                weighted_constraint.append((acc["constraint_puzzle_acc"], w))
                weighted_settled.append((float(acc["per_sample_settled"].mean()), w))
                weighted_lines.append((float(acc["per_sample_num_lines_extracted"].mean()), w))

                valid_bn = np.asarray(acc["per_sample_valid"]).reshape(B, N)
                any_valid = valid_bn.any(axis=1)
                all_any_valid.append(any_valid)

                if _wandb is not None and len(panels) < n_log:
                    n_new = min(n_log - len(panels), B)
                    cond_cpu = sub["spatial_conditions"].cpu()
                    true_cpu = sub["images"].cpu()
                    gen_bn_cpu = gen_flat.reshape(B, N, *gen_flat.shape[1:]).cpu()
                    for i in range(n_new):
                        # No "best" objective here (success is binary) — log
                        # the first valid candidate if any, else candidate 0.
                        row = valid_bn[i]
                        idx = int(np.argmax(row)) if row.any() else 0
                        panel = make_ball_drop_panel_image(cond_cpu[i], gen_bn_cpu[i, idx], true_cpu[i])
                        panels.append(_wandb.Image(panel, caption=f"sample[{n_done + i}]"))

            n_done += B_full

        any_valid_all = np.concatenate(all_any_valid) if all_any_valid else np.zeros(0, dtype=bool)
        result: dict = {
            f"constraint_puzzle_acc{suffix}": _weighted_mean(weighted_constraint),
            f"settled_rate{suffix}": _weighted_mean(weighted_settled),
            f"mean_lines_extracted{suffix}": _weighted_mean(weighted_lines),
            f"valid_rate{suffix}": float(any_valid_all.mean()) if len(any_valid_all) else float("nan"),
        }
        if panels:
            result["samples"] = panels
        return result

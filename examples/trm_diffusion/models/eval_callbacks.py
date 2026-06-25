"""
models/eval_callbacks.py — Pluggable eval callbacks.

Callbacks have the signature:
    callback(model, dataloader, accelerator, **kwargs) -> dict[str, float]

They are invoked from eval_step after generic loss metrics are computed.
Models accept a list of callbacks and merge all returned dicts.

Callbacks read eval_cfg from the model at call time (model.eval_cfg is always
present since it controls CFG scale during inference).  Heavy resources like
classifiers are owned by the callback and loaded in __init__.
"""

from __future__ import annotations

import numpy as np
import torch
from torchvision.utils import make_grid
from tqdm.auto import tqdm

try:
    import wandb as _wandb
except ImportError:
    _wandb = None

try:
    from eval.mnist_eval import (
        evaluate_grids,
        load_or_train_classifier,
        make_panel_image,
        plot_thinker_ts_curve,
        sample_grids,
    )
except ImportError:
    evaluate_grids = load_or_train_classifier = make_panel_image = plot_thinker_ts_curve = sample_grids = None

from datasets.data_sample import DataSample


class EvalCallbackBase:
    """Abstract base for eval callbacks."""

    def __call__(self, model, dataloader, accelerator, **kwargs) -> dict:
        raise NotImplementedError


# ── CLEVR / latent-DiT ────────────────────────────────────────────────────────


class ImageGenEvalCallback(EvalCallbackBase):
    """
    DDIM sampling + side-by-side WandB logging for latent painters (CLEVR etc.).

    Uses SamplingPipeline + CFGPredictor from models.sampling with the
    DataSample interface.  Reads from the model at call time:
      model.eval_cfg.{num_ddim_steps, cfg_scale, num_log_images}
      model.scheduler
      model._noise_shape     — (C, H, W) tuple, no batch dim
      model._decode_for_eval(z) -> Tensor in [0, 1]
      model.condition_keys   — DataSample fields to pull from batch
      model.null_condition_sample(sample) -> DataSample
    """

    @torch.no_grad()
    def __call__(self, model, dataloader, accelerator, **kwargs) -> dict:
        if not accelerator.is_main_process or _wandb is None:
            return {}

        from models.sampling import CFGPredictor, DirectPredictor, SamplingPipeline, SchedulerConfig

        step = kwargs.get("step", None)
        device = accelerator.device
        n_log = model.eval_cfg.num_log_images
        cfg_scale = model.eval_cfg.cfg_scale

        sched_cfg = SchedulerConfig(
            num_train_timesteps=model.scheduler.config.num_train_timesteps,
            beta_schedule=model.scheduler.config.beta_schedule,
            prediction_type=model.scheduler.config.prediction_type,
            num_inference_steps=model.eval_cfg.num_ddim_steps,
            sampler="ddim",
        )
        pipeline = SamplingPipeline(sched_cfg)
        predictor = CFGPredictor(cfg_scale) if cfg_scale > 1.0 else DirectPredictor()

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
            latents = pipeline.sample(model, base_sample, predictor, shape=(B, *model._noise_shape), device=device)
            imgs = model._decode_for_eval(latents)
            sample_images.extend(imgs.cpu().unbind(0))

            gt_raw = batch["images"][:B]
            gt_images.extend(((gt_raw + 1.0) / 2.0).clamp(0.0, 1.0).cpu().unbind(0))

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
    DDIM sampling eval for MNIST Sudoku models.

    Runs sample_grids with the thinker condition, evaluates with a digit
    classifier, and logs WandB panels + the thinker-timestep accuracy curve.

    Returns: cell_acc, puzzle_acc and (when include_thinker_metrics=True)
             thinker_cell_acc_best/mean, thinker_puzzle_acc_best/mean,
             thinker_deviation_from_best, painter_dev_from_best/mean_thinker.

    Args:
        classifier_path: path passed to load_or_train_classifier. If None the
                         callback is a no-op.
        cell_size: pixel size of each Sudoku cell (matches model_cfg.cell_size).
        include_thinker_metrics: set False to skip thinker accuracy metrics.
    """

    def __init__(
        self,
        classifier_path: str | None = None,
        cell_size: int = 16,
        include_thinker_metrics: bool = True,
    ):
        self.cell_size = cell_size
        self.include_thinker_metrics = include_thinker_metrics
        self.eval_clf = None
        if classifier_path is not None:
            self.eval_clf = load_or_train_classifier(classifier_path, None, cell_size, "cuda")
            for p in self.eval_clf.parameters():
                p.requires_grad_(False)

    def __call__(self, model, dataloader, accelerator, **kwargs) -> dict:
        if self.eval_clf is None or not accelerator.is_main_process:
            return {}

        device = accelerator.device
        painter_size = model.model_cfg.painter_size
        n_total = model.eval_cfg.num_samples
        n_ddim = model.eval_cfg.num_ddim_steps
        n_log = model.eval_cfg.num_log_images
        token_offset = getattr(model, "token_offset", 0)

        all_cell_acc, all_puzzle_acc = [], []
        all_thinker_cell_best, all_thinker_cell_mean = [], []
        all_thinker_puzzle_best, all_thinker_puzzle_mean = [], []
        all_thinker_deviation = []
        all_painter_dev_best, all_painter_dev_mean = [], []
        ts_cell_accs: dict[int, list] = {}
        ts_puzzle_accs: dict[int, list] = {}
        panels: list = []
        n_done = 0

        n_batches = (n_total + model.eval_cfg.batch_size - 1) // model.eval_cfg.batch_size
        for batch in tqdm(dataloader, "Sampling eval", total=n_batches):
            if n_done >= n_total:
                break
            solutions = batch["solution"]
            given_masks = batch.get("given_mask")
            B_cur = solutions.shape[0]

            sr = sample_grids(
                model,
                model._batch_to_sample(batch, device),
                num_train_timesteps=model.scheduler.config.num_train_timesteps,
                beta_schedule=model.scheduler.config.beta_schedule,
                prediction_type=model.scheduler.config.prediction_type,
                num_steps=n_ddim,
                device=device,
                solutions=solutions,
                painter_size=painter_size,
                given_masks=given_masks,
                cfg_scale=model.eval_cfg.cfg_scale,
            )
            acc = evaluate_grids(sr["generated"], solutions, self.eval_clf, self.cell_size, given_masks=given_masks)
            all_cell_acc.append(acc["cell_acc"])
            all_puzzle_acc.append(acc["puzzle_acc"])

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
                conds_vis = batch.get("conditions", batch.get("puzzle_tokens", None))
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
        }
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

    Only runs when model.has_realsolution_eval is True.

    Returns: real_cell_acc, real_puzzle_acc

    Args:
        classifier_path: same semantics as SudokuDDIMEvalCallback.
        cell_size: pixel size of each Sudoku cell.
    """

    def __init__(self, classifier_path: str | None = None, cell_size: int = 16):
        self.cell_size = cell_size
        self.eval_clf = None
        if classifier_path is not None:
            self.eval_clf = load_or_train_classifier(classifier_path, None, cell_size, "cuda")
            for p in self.eval_clf.parameters():
                p.requires_grad_(False)

    def __call__(self, model, dataloader, accelerator, **kwargs) -> dict:
        if not getattr(model, "has_realsolution_eval", False):
            return {}
        if self.eval_clf is None or not accelerator.is_main_process:
            return {}

        device = accelerator.device
        painter_size = model.model_cfg.painter_size
        n_total = model.eval_cfg.num_samples
        n_ddim = model.eval_cfg.num_ddim_steps

        all_real_cell, all_real_puzzle = [], []
        n_real = 0
        n_batches = (n_total + model.eval_cfg.batch_size - 1) // model.eval_cfg.batch_size

        for batch in tqdm(dataloader, "Realsolution eval", total=n_batches):
            if n_real >= n_total:
                break
            solutions = batch["solution"]
            given_masks = batch.get("given_mask")

            sr_r = sample_grids(
                model,
                model._batch_to_sample(batch, device),
                num_train_timesteps=model.scheduler.config.num_train_timesteps,
                beta_schedule=model.scheduler.config.beta_schedule,
                prediction_type=model.scheduler.config.prediction_type,
                num_steps=n_ddim,
                device=device,
                solutions=solutions,
                painter_size=painter_size,
                given_masks=given_masks,
                cfg_scale=model.eval_cfg.cfg_scale,
            )
            acc_r = evaluate_grids(sr_r["generated"], solutions, self.eval_clf, self.cell_size, given_masks=given_masks)
            all_real_cell.append(acc_r["cell_acc"])
            all_real_puzzle.append(acc_r["puzzle_acc"])
            n_real += solutions.shape[0]

        return {
            "real_cell_acc": float(np.mean(all_real_cell)),
            "real_puzzle_acc": float(np.mean(all_real_puzzle)),
        }

"""
models/tune_callbacks.py — diagnostics for hyperparameter tuning of
thinker-steered painters (ThinkerFrozenPainterBase), enabled by
tune_callbacks=steering and read by tuning/driver.py.
"""

from __future__ import annotations

import json
import math
import os
from typing import Optional

import numpy as np
import torch

from models.eval_callbacks import EvalCallbackBase


def _per_sample_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred.float() - target.float()).pow(2).flatten(1).mean(1)


class SteeringDiagnosticsCallback(EvalCallbackBase):
    """
    At every eval, on the first `max_batches` eval batches, with noise,
    timesteps and text dropout drawn from `seed` (the same samples at every
    eval and in every run, so runs compare pairwise):

      tune/dloss, _se       mean over samples of loss(steered) - loss(painter
                            alone): what the thinker does to the denoising
                            loss (< 0 helps); tune/dloss_t{k} per timestep
                            quantile k, in ascending raw t
      tune/loss, tune/loss_painter
                            the two losses
      tune/recursion_gain, _se
                            loss(n_sup = H = L = 1) - loss(full budget): what
                            the recursion adds (> 0 helps)
      tune/steer_ratio      mean ||pred - pred_painter|| / ||pred_painter||
      tune/param_norm_{thinker,encoder}
      tune/grad_norm_{thinker,encoder}
                            norm of the first supervision step's loss gradient
                            on one batch per rank ("encoder" = every trainable
                            parameter outside the thinker)
      tune/adam_eps_frac_{i}
                            for each optimizer with an eps (AdamW): fraction
                            of elements with sqrt(v_hat) < eps, where it acts
                            like SGD

    loss = per-sample MSE of the prediction against sample.target, as
    val/diff_loss. Batches are split across ranks and gathered; the main
    process appends {"step", "lrs", metrics} to out_path as one JSON line.
    """

    def __init__(
        self,
        out_path: Optional[str] = None,
        max_batches: int = 16,
        seed: int = 0,
        num_t_bins: int = 4,
        recursion_gain: bool = True,
        grad_norms: bool = True,
    ):
        self.out_path = out_path
        self.max_batches = max_batches
        self.seed = seed
        self.num_t_bins = num_t_bins
        self.recursion_gain = recursion_gain
        self.grad_norms = grad_norms

    @torch.no_grad()
    def __call__(self, model, dataloader, accelerator, **kwargs) -> dict:
        device = accelerator.device
        world, rank = accelerator.num_processes, accelerator.process_index
        per_rank = max(1, self.max_batches // world)
        if len(dataloader) < per_rank * world:
            raise ValueError(f"{len(dataloader)} eval batches, need {per_rank * world} ({per_rank} per rank)")
        mine = range(rank * per_rank, (rank + 1) * per_rank)

        columns, grads = [], None
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            for i, batch in enumerate(dataloader):
                if i >= mine.stop:
                    break
                if i not in mine:
                    continue
                torch.random.default_generator.manual_seed(self.seed + i)
                if device.type == "cuda":
                    torch.cuda.manual_seed(self.seed + i)  # this rank's GPU only
                [d] = model.prep_mb_data([batch], device)
                columns.append(self._sample_columns(model, d["sample"]))
                if self.grad_norms and grads is None:
                    grads = self._grad_norms(model, d)

        names = list(columns[0])
        stats = accelerator.gather(torch.cat([torch.stack(list(c.values()), 1) for c in columns]))
        if grads is not None:
            grads = accelerator.gather(grads[None]).mean(0)
        if not accelerator.is_main_process:
            return {}

        metrics = self._summarize({n: stats[:, j].double().cpu().numpy() for j, n in enumerate(names)})
        if grads is not None:
            metrics["grad_norm_thinker"], metrics["grad_norm_encoder"] = grads.tolist()
        metrics["param_norm_thinker"] = self._norm(model.get_thinker_params())
        metrics["param_norm_encoder"] = self._norm(model.get_encoder_params() + model.get_painter_params())
        optimizers = kwargs.get("optimizers") or []
        metrics.update(self._adam_eps_fracs(optimizers))

        if self.out_path is not None:
            os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)
            record = {"step": kwargs.get("step"), "lrs": [opt.lr for opt in optimizers], **metrics}
            with open(self.out_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        return {f"tune/{k}": v for k, v in metrics.items()}

    # ── Per batch ─────────────────────────────────────────────────────────────

    def _sample_columns(self, model, sample) -> dict[str, torch.Tensor]:
        pred = model(sample).pred
        pred_painter = self._painter_prediction(model, sample)
        cols = {
            "t": sample.timesteps.float(),
            "loss": _per_sample_mse(pred, sample.target),
            "loss_painter": _per_sample_mse(pred_painter, sample.target),
        }
        if self.recursion_gain:
            pred_1, _, _ = model.forward_with_carry(sample, n_sup=1, H_cycles=1, L_cycles=1)
            cols["loss_1rec"] = _per_sample_mse(pred_1.pred, sample.target)
        painter_norm = pred_painter.float().flatten(1).norm(dim=1).clamp_min(1e-12)
        cols["steer_ratio"] = (pred.float() - pred_painter.float()).flatten(1).norm(dim=1) / painter_norm
        return cols

    @staticmethod
    def _painter_prediction(model, sample) -> torch.Tensor:
        """The frozen painter without steering, as run_painter calls it."""
        dtype = getattr(model.painter, "painter_dtype", None)
        if model.train_cfg.force_unconditional_painter:
            sample = model.painter.null_condition_sample(sample)
        with torch.autocast(device_type=sample.x_noisy.device.type, dtype=dtype, enabled=dtype is not None):
            return model.painter(sample).pred

    @staticmethod
    def _grad_norms(model, d) -> torch.Tensor:
        groups = [
            [p for p in model.get_thinker_params() if p.requires_grad],
            [p for p in model.get_encoder_params() + model.get_painter_params() if p.requires_grad],
        ]
        with torch.enable_grad():
            pred, logits, _, _ = model.reasoning_step(d["sample"], d["z_H"], d["z_L"])
            loss, _ = model.loss_fn(pred, logits, d["sample"])
            grads = torch.autograd.grad(loss, groups[0] + groups[1], allow_unused=True)
        norms = []
        for g in (grads[: len(groups[0])], grads[len(groups[0]) :]):
            g = [x.float().norm() for x in g if x is not None]
            norms.append(torch.stack(g).norm() if g else torch.zeros((), device=loss.device))
        return torch.stack(norms)

    # ── Summaries ─────────────────────────────────────────────────────────────

    def _summarize(self, s: dict[str, np.ndarray]) -> dict[str, float]:
        n = len(s["t"])
        dloss = s["loss"] - s["loss_painter"]
        out = {
            "dloss": dloss.mean(),
            "dloss_se": dloss.std(ddof=1) / math.sqrt(n),
            "loss": s["loss"].mean(),
            "loss_painter": s["loss_painter"].mean(),
            "steer_ratio": s["steer_ratio"].mean(),
        }
        edges = np.quantile(s["t"], np.linspace(0, 1, self.num_t_bins + 1))
        bins = np.clip(np.searchsorted(edges, s["t"], side="right") - 1, 0, self.num_t_bins - 1)
        for k in range(self.num_t_bins):
            out[f"dloss_t{k}"] = dloss[bins == k].mean() if (bins == k).any() else float("nan")
        if "loss_1rec" in s:
            gain = s["loss_1rec"] - s["loss"]
            out["recursion_gain"] = gain.mean()
            out["recursion_gain_se"] = gain.std(ddof=1) / math.sqrt(n)
        return {k: float(v) for k, v in out.items()}

    @staticmethod
    def _norm(params) -> float:
        return float(torch.stack([p.detach().float().norm() for p in params]).norm()) if params else 0.0

    @staticmethod
    def _adam_eps_fracs(optimizers) -> dict[str, float]:
        out = {}
        for i, opt in enumerate(optimizers):
            below = total = 0
            for group in opt.optimizer.param_groups:
                if "eps" not in group:
                    continue
                beta2 = group["betas"][1]
                for p in group["params"]:
                    state = opt.optimizer.state.get(p)
                    if not state:
                        continue
                    v_hat = state["exp_avg_sq"] / (1 - beta2 ** float(state["step"]))
                    below += int((v_hat.sqrt() < group["eps"]).sum())
                    total += v_hat.numel()
            if total:
                out[f"adam_eps_frac_{i}"] = below / total
        return out

"""
models/losses.py — Pluggable loss functions for TRM-diffusion training.

All losses follow the interface:
    forward(noise_pred, logits, batch_dict) -> (total_loss, {name: float})

where total_loss is a backwardable scalar tensor and the second return value
is a flat dict of named loss values for logging.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets.sudoku_dataset import IGNORE_LABEL_ID
from models.diffusion_utils import x0_from_noise_pred, ddim_prev_sample
from configs.schemas import ClassifierLossConfig


class LossBase(nn.Module):
    """Abstract base for all training losses."""

    @abstractmethod
    def forward(
        self,
        noise_pred: torch.Tensor,
        logits: Optional[torch.Tensor],
        batch_dict: dict,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            noise_pred: (B, C, H, W) model prediction
            logits:     (B, N, vocab_size) thinker logits, may be None
            batch_dict: dict with at minimum:
                "noisy", "timesteps", "target", "images", "solution", "ce_labels"

        Returns:
            total_loss: scalar tensor (backwardable)
            components: {loss_name: float} for logging
        """
        pass


class MSELoss(LossBase):
    """
    Diffusion MSE loss with optional min-SNR weighting (Hang et al. 2023).

    For epsilon prediction:  w(t) = min(SNR(t), gamma)
    For x0 prediction:       w(t) = min(SNR(t), gamma) / SNR(t)
    Set minsnr_gamma=None to disable weighting (plain MSE).
    """

    def __init__(self, weight: float, scheduler, minsnr_gamma: Optional[float] = None):
        super().__init__()
        self.weight = weight
        self.minsnr_gamma = minsnr_gamma
        self._scheduler = scheduler

    def _minsnr_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        alphas = self._scheduler.alphas_cumprod.to(timesteps.device)[timesteps]
        snr = alphas / (1.0 - alphas).clamp(min=1e-8)
        if self._scheduler.config.prediction_type == "epsilon":
            return snr.clamp(max=self.minsnr_gamma)
        else:
            return snr.clamp(max=self.minsnr_gamma) / snr.clamp(min=1e-8)

    def forward(self, noise_pred, logits, batch_dict):
        target = batch_dict["target"]
        if self.minsnr_gamma is not None:
            per_sample = (noise_pred.float() - target).pow(2).flatten(1).mean(1)
            w = self._minsnr_weights(batch_dict["timesteps"])
            diff_loss = (w * per_sample).mean()
        else:
            diff_loss = F.mse_loss(noise_pred.float(), target)
        return self.weight * diff_loss, {"diff_loss": diff_loss.item()}


class ThinkerCELoss(LossBase):
    """
    Cross-entropy loss on thinker logits against solution labels.

    Expects batch_dict["ce_labels"] to hold (B, N) integer labels
    with IGNORE_LABEL_ID for positions to skip.
    """

    def __init__(self, weight: float):
        super().__init__()
        self.weight = weight

    def forward(self, noise_pred, logits, batch_dict):
        zero = torch.tensor(0.0, device=noise_pred.device)
        if logits is None or self.weight == 0.0:
            return zero, {"sudoku_loss": 0.0}
        ce_labels = batch_dict.get("ce_labels")
        if ce_labels is None:
            return zero, {"sudoku_loss": 0.0}
        B_, N, C = logits.shape
        if N > ce_labels.shape[1]:
            return zero, {"sudoku_loss": 0.0}
        sudoku_loss = F.cross_entropy(
            logits.float().reshape(B_ * N, C),
            ce_labels[:, :N].reshape(B_ * N).clamp(min=0),
            ignore_index=IGNORE_LABEL_ID,
        )
        return self.weight * sudoku_loss, {"sudoku_loss": sudoku_loss.item()}


class ClassifierLoss(LossBase):
    """
    Classifier-guided training loss on predicted x0 images.

    Loads and freezes the classifier at construction time.
    """

    def __init__(self, cfg, cell_size: int, scheduler):
        super().__init__()
        self.cfg = cfg
        self.weight = cfg.weight
        self.cell_size = cell_size
        self._scheduler = scheduler

        from eval.mnist_eval import load_or_train_classifier
        from train_noisy_classifier import load_noisy_classifier

        if cfg.noisy_classifier:
            clf = load_noisy_classifier(cfg.classifier_path, "cuda")
        else:
            clf = load_or_train_classifier(cfg.classifier_path, None, cell_size, "cuda")
        for p in clf.parameters():
            p.requires_grad_(False)
        self.clf = clf

    def forward(self, noise_pred, logits, batch_dict):
        from models.painters import classifier_loss as _classifier_loss

        x0_pred = x0_from_noise_pred(noise_pred, batch_dict["noisy"], batch_dict["timesteps"], self._scheduler)
        clf_loss = _classifier_loss(
            x0_pred,
            batch_dict["noisy"],
            batch_dict["images"],
            batch_dict["solution"],
            batch_dict["timesteps"],
            self.cell_size,
            self.clf,
            self._scheduler,
            self.cfg,
        )
        return self.weight * clf_loss, {"clf_loss": clf_loss.item()}


class CombinedLoss(LossBase):
    """Weighted sum of multiple LossBase components."""

    def __init__(self, *losses: LossBase, diff_thinker_weight: float = 1.0):
        super().__init__()
        self.diff_thinker_weight = diff_thinker_weight
        self.losses = nn.ModuleList(losses)

    def scale_logits_for_painter(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale diffusion gradient back through thinker logits.

        w=1: full gradient; w=0: stop gradient; w in (0,1): partial gradient.
        """
        w = self.diff_thinker_weight
        if w == 0.0:
            return logits.detach()
        if w != 1.0:
            return w * logits + (1.0 - w) * logits.detach()
        return logits

    def forward(self, noise_pred, logits, batch_dict):
        scaled_logits = self.scale_logits_for_painter(logits) if logits is not None else None
        total = torch.tensor(0.0, device=noise_pred.device)
        breakdown: dict[str, float] = {}
        for loss_fn in self.losses:
            t, d = loss_fn(noise_pred, scaled_logits, batch_dict)
            total = total + t
            breakdown.update(d)
        return total, breakdown


def build_loss(train_cfg, scheduler, cell_size: Optional[int] = None) -> LossBase:
    """
    Build a loss object from a TrainConfig.

    Creates MSELoss + ThinkerCELoss (and ClassifierLoss if configured),
    wrapping them in CombinedLoss if there is more than one.
    """
    from configs.schemas import TrainConfig  # avoid circular import at module level

    components: list[LossBase] = []
    if train_cfg.mse_loss_weight > 0.0:
        components.append(MSELoss(train_cfg.mse_loss_weight, scheduler, train_cfg.minsnr_gamma))
    if train_cfg.sudoku_loss_weight > 0.0:
        components.append(ThinkerCELoss(train_cfg.sudoku_loss_weight))
    if (
        train_cfg.classifier_loss is not None
        and train_cfg.classifier_loss.weight > 0.0
        and train_cfg.classifier_loss.classifier_path is not None
        and cell_size is not None
    ):
        components.append(ClassifierLoss(train_cfg.classifier_loss, cell_size, scheduler))

    if not components:
        # Fallback: plain MSE with weight 1
        return MSELoss(1.0, scheduler)
    if len(components) == 1:
        return components[0]
    return CombinedLoss(*components)


def classifier_loss(
    x0_pred, noisy, images, solution, timesteps, cell_size, classifier, scheduler, loss_cfg: ClassifierLossConfig
):
    """
    Classifier-based training loss on predicted images.
    """
    eligible = (timesteps < loss_cfg.t_max).nonzero(as_tuple=True)[0]
    if eligible.numel() == 0:
        return x0_pred.sum() * 0.0

    x0_sel = x0_pred[eligible]
    noi_sel = noisy[eligible]
    ts_sel = timesteps[eligible]
    sol_sel = solution[eligible].to(x0_pred.device)
    N = eligible.numel()

    img_sel = ddim_prev_sample(x0_sel, noi_sel, ts_sel, scheduler) if loss_cfg.target == "x_tm1" else x0_sel

    cells = img_sel.unfold(2, cell_size, cell_size).unfold(3, cell_size, cell_size)
    cells = cells.permute(0, 2, 3, 1, 4, 5).contiguous().reshape(N * 81, 1, cell_size, cell_size)

    if loss_cfg.loss_type == "perceptual":
        clean_sel = images[eligible]
        clean_cells = clean_sel.unfold(2, cell_size, cell_size).unfold(3, cell_size, cell_size)
        clean_cells = clean_cells.permute(0, 2, 3, 1, 4, 5).contiguous().reshape(N * 81, 1, cell_size, cell_size)
        feats_pred = classifier.encoder(cells)
        with torch.no_grad():
            feats_ref = classifier.encoder(clean_cells)
        return F.mse_loss(feats_pred.flatten(1), feats_ref.flatten(1))

    logits = classifier(cells)
    labels = sol_sel.reshape(N * 81)
    return F.cross_entropy(logits, labels, ignore_index=IGNORE_LABEL_ID)

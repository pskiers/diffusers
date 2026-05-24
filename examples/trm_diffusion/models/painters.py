from typing import Optional, Any

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np

from diffusers import UNet2DModel

from models.base import BaseModel
from models.optim_utils import ScheduledOptimizer, apply_lr_and_step
from models.control_painter_unet2d import ControlPainterUNet
from models.spade_painter_unet2d import SPADEUNet2D
from models.utility_models import SpatialBridge, ConditioningPyramid
from configs.schemas import TrainConfig, EvalConfig, PainterOptimConfig, PainterConfig, ClassifierLossConfig
from train_noisy_classifier import load_noisy_classifier
from eval.mnist_eval import load_or_train_classifier, sample_grids, evaluate_grids
from datasets.mnist_sudoku_dataset import get_solution_tokens
from datasets.sudoku_dataset import IGNORE_LABEL_ID, make_tok_labels
from models.diffusion_utils import apply_noisy_swap, x0_from_noise_pred, ddim_prev_sample


def make_painter(
    painter_size: int,
    bridge_channels: int,
    painter_channels: tuple[int, ...],
    layers_per_block: int = 2,
) -> UNet2DModel:
    """
    Build the denoising UNet. Uses plain conv blocks throughout (no attention).
    """
    n = len(painter_channels)
    norm_num_groups = 32
    while norm_num_groups > 1 and any(c % norm_num_groups != 0 for c in painter_channels):
        norm_num_groups //= 2
    return UNet2DModel(
        sample_size=painter_size,
        in_channels=1 + bridge_channels,
        out_channels=1,
        block_out_channels=painter_channels,
        down_block_types=("DownBlock2D",) * n,
        up_block_types=("UpBlock2D",) * n,
        norm_num_groups=norm_num_groups,
        layers_per_block=layers_per_block,
    )


def make_painter_control(
    painter_size: int,
    painter_channels: tuple[int, ...],
    layers_per_block: int = 2,
) -> ControlPainterUNet:
    """
    Build a ControlNet-capable painter UNet (in_channels=1, no bridge concat).
    """
    n = len(painter_channels)
    norm_num_groups = 32
    while norm_num_groups > 1 and any(c % norm_num_groups != 0 for c in painter_channels):
        norm_num_groups //= 2
    return ControlPainterUNet(
        sample_size=painter_size,
        in_channels=1,
        out_channels=1,
        block_out_channels=painter_channels,
        down_block_types=("DownBlock2D",) * n,
        up_block_types=("UpBlock2D",) * n,
        norm_num_groups=norm_num_groups,
        layers_per_block=layers_per_block,
    )


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


def compute_losses_painter(
    model, condition, batch, scheduler, accelerator, sudoku_loss_weight, token_input=True
) -> dict:
    """Single forward pass (eval). Ported from train_mnist_sudoku.py compute_losses."""
    device = accelerator.device
    images = batch["images"].to(device)
    solution = batch["solution"].to(device)  # (B,81) 0-8 or IGNORE for given
    given_mask = batch.get("given_mask")
    puzzle_ids = batch["puzzle_id"].to(device) if "puzzle_id" in batch else None

    B = images.shape[0]
    noise = torch.randn_like(images)
    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=device, dtype=torch.long)
    noisy = scheduler.add_noise(images, noise, timesteps)

    noise_pred, sudoku_logits = model(noisy, timesteps, condition, puzzle_ids=puzzle_ids)

    target = noise if scheduler.config.prediction_type == "epsilon" else images
    diff_loss = F.mse_loss(noise_pred.float(), target)

    sudoku_loss = torch.tensor(0.0, device=device)
    if token_input:
        ce_labels = make_tok_labels(solution)
    else:
        ce_labels = solution
    if sudoku_logits is not None and sudoku_loss_weight > 0:
        B_, N, C = sudoku_logits.shape
        sudoku_loss = F.cross_entropy(
            sudoku_logits.float().reshape(B_ * N, C),
            ce_labels[:, :N].reshape(B_ * N).clamp(min=0),
            ignore_index=IGNORE_LABEL_ID,
        )

    total_loss = diff_loss + sudoku_loss_weight * sudoku_loss

    thinker_cell_acc = None
    thinker_puzzle_acc = None
    if sudoku_logits is not None:
        B_, N, C = sudoku_logits.shape
        preds = sudoku_logits.argmax(dim=-1)
        targets = ce_labels[:B_, :N]
        correct = preds == targets  # (B_, N)

        thinker_puzzle_acc = correct.all(dim=1).float().mean()

        if given_mask is not None:
            blank = ~given_mask.to(device)[:B_, :N]
            n_blank = blank.sum()
            thinker_cell_acc = correct[blank].float().mean() if n_blank > 0 else correct.float().mean()
        else:
            thinker_cell_acc = correct.float().mean()

    return {
        "loss": total_loss,
        "diff_loss": diff_loss,
        "sudoku_loss": sudoku_loss,
        "thinker_cell_acc": thinker_cell_acc,
        "thinker_puzzle_acc": thinker_puzzle_acc,
    }


class StandalonePainter(BaseModel):
    """
    Pure diffusion painter with no thinker.
    """

    token_input: bool = False  # uses solution tokens, handled via _get_condition  TODO remove
    has_realsolution_eval: bool = True  # realsolution IS the only conditioning  TODO remove

    def __init__(
        self,
        model_cfg: PainterConfig,
        optim_cfg: PainterOptimConfig,
        train_cfg: TrainConfig,
        eval_cfg: EvalConfig,
        scheduler: Any,
        classifier_data_path=None,
    ):
        super().__init__()
        self.model_cfg = model_cfg
        self.optim_cfg = optim_cfg
        self.train_cfg = train_cfg
        self.eval_cfg = eval_cfg
        self.scheduler = scheduler
        self.cell_size = model_cfg.cell_size
        self.grid = model_cfg.painter_size // model_cfg.cell_size
        self.vocab_size = model_cfg.vocab_size
        self.dtype: Optional[torch.dtype] = (
            {"bfloat16": torch.bfloat16, "float16": torch.float16}[model_cfg.painter_dtype]
            if model_cfg.painter_dtype is not None else None
        )
        self.bridge = SpatialBridge(
            in_channels=model_cfg.vocab_size,
            out_channels=model_cfg.bridge_channels,
            painter_size=model_cfg.painter_size,
        )
        self.painter = make_painter(
            painter_size=model_cfg.painter_size,
            bridge_channels=model_cfg.bridge_channels,
            painter_channels=tuple(model_cfg.painter_channels),
            layers_per_block=model_cfg.painter_layers_per_block,
        )

        self.mse_w = float(self.train_cfg.mse_loss_weight)

        # training classifier
        self.train_clf = None
        self.clf_cfg: ClassifierLossConfig | None = self.train_cfg.classifier_loss
        if self.clf_cfg is not None and self.clf_cfg.classifier_path is not None:
            if self.clf_cfg.noisy_classifier:
                self.train_clf = load_noisy_classifier(self.clf_cfg.classifier_path, self.device)
            else:
                self.train_clf = load_or_train_classifier(
                    self.clf_cfg.classifier_path,
                    classifier_data_path,
                    model_cfg.cell_size,
                    "cuda",
                )
            for p in self.train_clf.parameters():
                p.requires_grad_(False)

        # eval classifier (no training, just load)
        self.eval_clf = None
        if eval_cfg.classifier_path is not None:
            self.eval_clf = load_or_train_classifier(
                eval_cfg.classifier_path, None, model_cfg.cell_size, "cuda"
            )
            for p in self.eval_clf.parameters():
                p.requires_grad_(False)

    def build_optimizers(self, world_size, num_steps):
        optims = []

        optim = torch.optim.AdamW(self.parameters(), lr=0, weight_decay=self.optim_cfg.weight_decay)
        optims.append(
            ScheduledOptimizer(
                optim,
                base_lr=self.optim_cfg.lr,
                warmup_steps=self.optim_cfg.warmup_steps,
                num_steps=num_steps,
                min_ratio=self.optim_cfg.lr_min_ratio,
            )
        )
        return optims

    def _solution_to_spatial(self, solution_tokens: torch.Tensor) -> torch.Tensor:
        """(B, 81) long in 2-10 → (B, vocab_size, grid, grid) one-hot float."""
        B = solution_tokens.shape[0]
        idx = solution_tokens.clamp(min=0, max=self.vocab_size - 1)
        onehot = F.one_hot(idx, num_classes=self.vocab_size).float()  # (B, 81, V)
        return onehot.transpose(1, 2).reshape(B, self.vocab_size, self.grid, self.grid)

    def forward(
        self,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,  # (B, 81) long solution tokens 2-10
        puzzle_ids: Optional[torch.Tensor] = None,
    ):
        """
        Training: randomly drops conditioning per sample at rate cfg_prob.
        Inference (self.training=False, cfg_scale>1): runs conditioned + null
        passes and combines them for classifier-free guidance.
        """
        spatial = self._solution_to_spatial(condition)

        if self.training and self.train_cfg.cfg_prob > 0:
            drop = torch.rand(spatial.shape[0], 1, 1, 1, device=spatial.device) < self.train_cfg.cfg_prob
            spatial = spatial * (~drop)

        ctx = (
            torch.autocast(device_type=noisy.device.type, dtype=self.dtype)
            if self.dtype is not None
            else torch.autocast(device_type=noisy.device.type, enabled=False)
        )

        if not self.training and self.eval_cfg.cfg_scale > 1.0:
            null = torch.zeros_like(spatial)
            s_both = torch.cat([spatial, null], dim=0)
            n_both = noisy.repeat(2, 1, 1, 1)
            t_both = timesteps.repeat(2)
            with ctx:
                bf = self.bridge(s_both)
                pred = self.painter(torch.cat([n_both, bf], dim=1), t_both).sample
            pred_cond, pred_uncond = pred.chunk(2, dim=0)
            return pred_uncond + self.eval_cfg.cfg_scale * (pred_cond - pred_uncond), None

        with ctx:
            bridge_feat = self.bridge(spatial)
            noise_pred = self.painter(torch.cat([noisy, bridge_feat], dim=1), timesteps).sample
        return noise_pred, None

    def train_step(self, micro_batches, accelerator, optimizers, ema, global_batch_size, global_step, **kwargs):
        K = len(micro_batches)
        device = accelerator.device

        total_diff_loss = 0.0
        total_clf_loss = 0.0

        for mb in micro_batches:
            images = mb["images"].to(device)
            solution = mb["solution"].to(device)  # (B, 81) int64  0-8
            solution_tokens = get_solution_tokens(solution)

            bsz = images.shape[0]
            noise = torch.randn_like(images)
            timesteps = torch.randint(
                0, self.scheduler.config.num_train_timesteps, (bsz,), device=device, dtype=torch.long
            )
            noisy = self.scheduler.add_noise(images, noise, timesteps)
            target = noise if self.scheduler.config.prediction_type == "epsilon" else images
            noisy, target = apply_noisy_swap(
                images=images,
                noisy=noisy,
                target=target,
                timesteps=timesteps,
                scheduler=self.scheduler,
                swap_cfg=self.train_cfg.noisy_swap,
            )

            noise_pred, _ = self(noisy, timesteps, solution_tokens)

            step_loss = torch.tensor(0.0, device=device)
            if self.train_cfg.mse_loss_weight > 0.0:
                diff_loss = F.mse_loss(noise_pred.float(), target)
                step_loss = step_loss + self.train_cfg.mse_loss_weight * diff_loss
                total_diff_loss += diff_loss.item()

            if self.clf_cfg is not None and self.clf_cfg.weight > 0.0:
                x0_pred = x0_from_noise_pred(noise_pred, noisy, timesteps, self.scheduler)
                clf_loss = classifier_loss(
                    x0_pred,
                    noisy,
                    images,
                    solution,
                    timesteps,
                    self.cell_size,
                    self.train_clf,
                    self.scheduler,
                    self.train_cfg.classifier_loss,
                )
                step_loss = step_loss + self.clf_cfg.weight * clf_loss
                total_clf_loss += clf_loss.item()

            accelerator.backward(step_loss / (global_batch_size * K))

        accelerator.clip_grad_norm_(self.parameters(), 1.0)
        lr = apply_lr_and_step(optimizers, global_step)
        if ema is not None:
            ema.update(self)
        global_step += 1

        losses = {"diff_loss": total_diff_loss / K}
        if self.clf_cfg is not None and self.clf_cfg.weight > 0.0:
            losses["clf_loss"] = total_clf_loss / K
        return losses, lr, global_step

    def compile_submodules(self):
        self.painter = torch.compile(self.painter)
        if self.bridge is not None:
            self.bridge = torch.compile(self.bridge)

    @torch.no_grad()
    def eval_step(self, dataloader, accelerator, **kwargs) -> dict:
        max_batches = kwargs.get("max_batches", 100)
        self.eval()

        # ── Loss eval ─────────────────────────────────────────────────────────
        diff_losses = []
        for i, batch in tqdm(enumerate(dataloader), "Eval loss", total=max_batches):
            if i >= max_batches:
                break
            m = compute_losses_painter(
                model=self,
                condition=get_solution_tokens(batch["solution"].to(accelerator.device)),
                batch=batch,
                scheduler=self.scheduler,
                accelerator=accelerator,
                sudoku_loss_weight=0,
                token_input=False,
            )
            if m.get("diff_loss") is not None:
                diff_losses.append(float(m["diff_loss"]))
        result = {"diff_loss": float(np.mean(diff_losses))} if diff_losses else {}

        # ── Sampling eval ─────────────────────────────────────────────────────
        if self.eval_clf is not None and accelerator.is_main_process:
            n_total = self.eval_cfg.num_samples
            n_ddim = self.eval_cfg.num_ddim_steps
            all_cell_acc, all_puzzle_acc, n_done = [], [], 0
            for batch in tqdm(dataloader, "Sampling eval"):
                if n_done >= n_total:
                    break
                solutions = batch["solution"]
                puzzle_ids = batch.get("puzzle_id")
                if puzzle_ids is not None:
                    puzzle_ids = puzzle_ids.to(accelerator.device)
                sr = sample_grids(
                    self,
                    get_solution_tokens(solutions.to(accelerator.device)),
                    num_train_timesteps=self.scheduler.config.num_train_timesteps,
                    beta_schedule=self.scheduler.config.beta_schedule,
                    prediction_type=self.scheduler.config.prediction_type,
                    num_steps=n_ddim,
                    device=accelerator.device,
                    puzzle_ids=puzzle_ids,
                    solutions=solutions,
                    painter_size=self.model_cfg.painter_size,
                    given_masks=batch.get("given_mask"),
                )
                acc = evaluate_grids(
                    sr["generated"], solutions, self.eval_clf, self.cell_size,
                    given_masks=batch.get("given_mask"),
                )
                all_cell_acc.append(acc["cell_acc"])
                all_puzzle_acc.append(acc["puzzle_acc"])
                n_done += solutions.shape[0]
            result["cell_acc"] = float(np.mean(all_cell_acc))
            result["puzzle_acc"] = float(np.mean(all_puzzle_acc))

        self.train()
        return result


class StandalonePainterSPADE(StandalonePainter):
    """
    Standalone painter using SPADE conditioning instead of a bridge+concat.

    Solution tokens → one-hot (B, vocab_size, 9, 9) → bilinearly upsampled to
    painter_size → fed as semantic map `s` to SPADEUNet2D (in_channels=1, no concat).
    """

    def __init__(
        self,
        model_cfg: PainterConfig,
        optim_cfg: PainterOptimConfig,
        train_cfg: TrainConfig,
        eval_cfg: EvalConfig,
        scheduler: Any,
    ):
        super().__init__(
            model_cfg=model_cfg,
            optim_cfg=optim_cfg,
            train_cfg=train_cfg,
            eval_cfg=eval_cfg,
            scheduler=scheduler,
        )
        self.painter_size = model_cfg.painter_size
        # Discard the bridge (unused in SPADE) and replace UNet.
        del self.bridge
        self.bridge = None
        self.painter = SPADEUNet2D(
            painter_size=model_cfg.painter_size,
            sem_channels=model_cfg.vocab_size,
            block_out_channels=tuple(model_cfg.painter_channels),
            layers_per_block=model_cfg.painter_layers_per_block,
        )

    def _solution_to_spatial(self, solution_tokens: torch.Tensor) -> torch.Tensor:
        """(B, 81) long 2-10 → (B, vocab_size, painter_size, painter_size) float upsampled."""
        spatial = super()._solution_to_spatial(solution_tokens)  # (B, V, grid, grid)
        return F.interpolate(spatial, size=self.painter_size, mode="bilinear", align_corners=False)

    def forward(
        self,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
        puzzle_ids: Optional[torch.Tensor] = None,
    ):
        s = self._solution_to_spatial(condition)

        if self.training and self.train_cfg.cfg_prob > 0:
            drop = torch.rand(s.shape[0], 1, 1, 1, device=s.device) < self.train_cfg.cfg_prob
            s = s * (~drop)

        ctx = (
            torch.autocast(device_type=noisy.device.type, dtype=self.dtype)
            if self.dtype is not None
            else torch.autocast(device_type=noisy.device.type, enabled=False)
        )

        if not self.training and self.eval_cfg.cfg_scale > 1.0:
            null = torch.zeros_like(s)
            s_both = torch.cat([s, null], dim=0)
            n_both = noisy.repeat(2, 1, 1, 1)
            t_both = timesteps.repeat(2)
            with ctx:
                pred = self.painter(n_both, t_both, s_both)
            pred_cond, pred_uncond = pred.chunk(2, dim=0)
            return pred_uncond + self.eval_cfg.cfg_scale * (pred_cond - pred_uncond), None

        with ctx:
            noise_pred = self.painter(noisy, timesteps, s)
        return noise_pred, None


class StandalonePainterControl(StandalonePainter):
    """
    Standalone painter using ControlNet-style residual injection instead of bridge+concat.

    Solution tokens → one-hot (B, vocab_size, 9, 9) → bilinearly upsampled to
    painter_size → ConditioningPyramid → per-layer residuals injected into
    _ControlPainterUNet (in_channels=1, no bridge concatenation).
    """

    def __init__(
        self,
        model_cfg: PainterConfig,
        optim_cfg: PainterOptimConfig,
        train_cfg: TrainConfig,
        eval_cfg: EvalConfig,
        scheduler: Any,
    ):
        super().__init__(
            model_cfg=model_cfg,
            optim_cfg=optim_cfg,
            train_cfg=train_cfg,
            eval_cfg=eval_cfg,
            scheduler=scheduler,
        )
        self.painter_size = model_cfg.painter_size
        del self.bridge
        self.bridge = None
        self.painter = make_painter_control(
            painter_size=model_cfg.painter_size,
            painter_channels=tuple(model_cfg.painter_channels),
            layers_per_block=model_cfg.painter_layers_per_block,
        )
        self.control_pyramid = ConditioningPyramid(
            in_channels=model_cfg.vocab_size,
            block_out_channels=tuple(model_cfg.painter_channels),
            layers_per_block=model_cfg.painter_layers_per_block,
        )

    def _solution_to_spatial(self, solution_tokens: torch.Tensor) -> torch.Tensor:
        """(B, 81) long 2-10 → (B, vocab_size, painter_size, painter_size) float upsampled."""
        spatial = super()._solution_to_spatial(solution_tokens)
        return F.interpolate(spatial, size=self.painter_size, mode="bilinear", align_corners=False)

    def _run_painter_ctrl(self, noisy, s, timesteps):
        down_res, mid_res = self.control_pyramid(s)
        return self.painter(
            noisy,
            timesteps,
            down_block_additional_residuals=down_res,
            mid_block_additional_residual=mid_res,
        ).sample

    def forward(
        self,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
        puzzle_ids: Optional[torch.Tensor] = None,
    ):
        s = self._solution_to_spatial(condition)

        if self.training and self.train_cfg.cfg_prob > 0:
            drop = torch.rand(s.shape[0], 1, 1, 1, device=s.device) < self.train_cfg.cfg_prob
            s = s * (~drop)

        ctx = (
            torch.autocast(device_type=noisy.device.type, dtype=self.dtype)
            if self.dtype is not None
            else torch.autocast(device_type=noisy.device.type, enabled=False)
        )

        if not self.training and self.eval_cfg.cfg_scale > 1.0:
            B = noisy.shape[0]
            null = torch.zeros_like(s)
            with ctx:
                pred_cond = self._run_painter_ctrl(noisy, s, timesteps)
                pred_uncond = self._run_painter_ctrl(noisy, null, timesteps)
            return pred_uncond + self.eval_cfg.cfg_scale * (pred_cond - pred_uncond), None

        with ctx:
            noise_pred = self._run_painter_ctrl(noisy, s, timesteps)
        return noise_pred, None

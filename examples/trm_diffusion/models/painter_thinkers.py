from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.trm_wrappers import SpatialTRM
from models.utility_models import SpatialEncoder, AttentiveBridge, SpatialBridge, TimestepMLP
import numpy as np
from tqdm.auto import tqdm

from models.painters import (
    make_painter,
    StandalonePainter,
    StandalonePainterControl,
    compute_losses_painter,
    classifier_loss as _classifier_loss,
)
from models.diffusion_utils import apply_noisy_swap, x0_from_noise_pred
from train_noisy_classifier import load_noisy_classifier
from models.utility_models import ConditioningPyramid
from models.eval_callbacks import EvalCallbackBase
from models.optim_utils import ScheduledOptimizer, apply_lr_and_step
from datasets.sudoku_dataset import IGNORE_LABEL_ID, make_tok_labels
from eval.mnist_eval import load_or_train_classifier
from configs.schemas import (
    ThinkerModelConfig,
    PainterThinkerConfig,
    ImageEncoderConfig,
    TimestepCondConfig,
    ThinkerOptimConfig,
    PainterOptimConfig,
    TrainConfig,
    EvalConfig,
    ClassifierLossConfig,
)
from models.base import BaseModel


class PainterThinkerV0Tok(BaseModel):
    """
    Painter-thinker model using the original TRM as the thinker (token input).
    """

    token_input: bool = True
    has_realsolution_eval: bool = True  # eval with full solution tokens as condition

    def __init__(
        self,
        thinker_cfg: ThinkerModelConfig,
        model_cfg: PainterThinkerConfig,
        train_cfg: TrainConfig,
        eval_cfg: EvalConfig,
        thinker_optim_cfg: ThinkerOptimConfig,
        painter_optim_cfg: PainterOptimConfig,
        scheduler,
        eval_callbacks=None,
    ):
        super().__init__()
        self.thinker_cfg = thinker_cfg
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.eval_cfg = eval_cfg
        self.thinker_optim_cfg = thinker_optim_cfg
        self.painter_optim_cfg = painter_optim_cfg
        self.scheduler = scheduler

        self.diff_thinker_weight = model_cfg.diff_thinker_weight
        self.thinker_bridge_mode = model_cfg.thinker_bridge_mode
        self._grid = model_cfg.painter_size // model_cfg.cell_size
        # Thinker vocab uses 0=PAD, 1=blank, 2-10=digits 1-9.
        # sample_grids compares argmax predictions against raw solution labels (0-8),
        # so it needs to know to shift by this offset when comparing.
        self.token_offset = 2
        self._painter_dtype: Optional[torch.dtype] = (
            {"bfloat16": torch.bfloat16, "float16": torch.float16}[model_cfg.painter_dtype]
            if model_cfg.painter_dtype is not None
            else None
        )

        # Training classifier — loaded from train.classifier_loss.classifier_path,
        # kept separate from the eval classifier so a noisy/different classifier
        # can be used for the loss without polluting the evaluation metric.
        self.train_clf = None
        self.clf_cfg: Optional[ClassifierLossConfig] = train_cfg.classifier_loss
        if self.clf_cfg is not None and self.clf_cfg.classifier_path is not None:
            if self.clf_cfg.noisy_classifier:
                self.train_clf = load_noisy_classifier(self.clf_cfg.classifier_path, "cuda")
            else:
                self.train_clf = load_or_train_classifier(
                    self.clf_cfg.classifier_path, None, model_cfg.cell_size, "cuda"
                )
            for p in self.train_clf.parameters():
                p.requires_grad_(False)

        self.eval_callbacks: list[EvalCallbackBase] = list(eval_callbacks) if eval_callbacks is not None else []

        self.thinker = SpatialTRM(
            optim_cfg=thinker_optim_cfg,
            vocab_size=thinker_cfg.vocab_size,
            seq_len=thinker_cfg.seq_len,
            hidden_size=thinker_cfg.hidden_size,
            n_heads=thinker_cfg.n_heads,
            L_layers=thinker_cfg.L_layers,
            L_cycles=thinker_cfg.L_cycles,
            H_cycles=thinker_cfg.H_cycles,
            n_sup=thinker_cfg.n_sup,
            expansion=thinker_cfg.expansion,
            forward_dtype=thinker_cfg.forward_dtype,
            mlp_t=thinker_cfg.mlp_t,
            pos_encodings=thinker_cfg.pos_encodings,
            puzzle_emb_ndim=thinker_cfg.puzzle_emb_ndim,
            puzzle_emb_len=thinker_cfg.puzzle_emb_len,
            num_puzzle_identifiers=thinker_cfg.num_puzzle_identifiers,
            halt_exploration_prob=thinker_cfg.halt_exploration_prob,
            batch_size=thinker_cfg.batch_size,
            freeze_weights=thinker_cfg.freeze_weights,
        )
        self.bridge = SpatialBridge(
            in_channels=thinker_cfg.vocab_size,
            out_channels=model_cfg.bridge_channels,
            painter_size=model_cfg.painter_size,
        )
        self.painter = make_painter(
            painter_size=model_cfg.painter_size,
            bridge_channels=model_cfg.bridge_channels,
            painter_channels=tuple(model_cfg.painter_channels),
            layers_per_block=model_cfg.painter_layers_per_block,
        )

    @property
    def n_sup(self) -> int:
        return self.thinker.n_sup

    def get_initial_states(self, bsz: int):
        return self.thinker.get_initial_states(bsz)

    def get_painter_params(self) -> list:
        """Parameters belonging to the bridge and painter UNet (for a separate optimizer)."""
        return list(self.bridge.parameters()) + list(self.painter.parameters())

    def get_thinker_params(self) -> list:
        """Parameters belonging to the thinker (excluding painter/bridge)."""
        painter_ids = {id(p) for p in self.get_painter_params()}
        return [p for p in self.parameters() if id(p) not in painter_ids]

    def build_optimizers(self, world_size, num_steps) -> list[ScheduledOptimizer]:
        thinker_optims = self.thinker.build_optimizers(world_size, num_steps)
        painter_params = self.get_painter_params()
        painter_optim = torch.optim.AdamW(painter_params, lr=0, weight_decay=self.painter_optim_cfg.weight_decay)
        painter_scheduled = ScheduledOptimizer(
            painter_optim,
            base_lr=self.painter_optim_cfg.lr,
            warmup_steps=self.painter_optim_cfg.warmup_steps,
            num_steps=num_steps,
            min_ratio=self.painter_optim_cfg.lr_min_ratio,
        )
        return thinker_optims + [painter_scheduled]

    # ── Condition helpers (overrideable by subclasses) ────────────────────────

    def _get_condition(self, mb: dict, device) -> torch.Tensor:
        return mb["puzzle_tokens"].to(device)

    def _get_teacher_condition(self, mb: dict, device) -> torch.Tensor:
        solution = mb["solution"].to(device)
        return solution.clamp(min=0) + self.token_offset

    def _make_ce_labels(self, solution: torch.Tensor) -> torch.Tensor:
        return make_tok_labels(solution)

    # ── Training step ─────────────────────────────────────────────────────────

    def train_step(self, micro_batches, accelerator, optimizers, ema, global_batch_size, global_step, **kwargs):
        if self.train_cfg.two_stage is not None:
            return self._train_step_two_stage(
                micro_batches, accelerator, optimizers, ema, global_batch_size, global_step
            )
        return self._train_step_standard(micro_batches, accelerator, optimizers, ema, global_batch_size, global_step)

    def _prep_mb_data(self, micro_batches, device):
        mb_data = []
        for mb in micro_batches:
            images = mb["images"].to(device)
            solution = mb["solution"].to(device) if "solution" in mb else None
            puzzle_ids = mb["puzzle_id"].to(device) if "puzzle_id" in mb else None
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
            z_H, z_L = self.get_initial_states(bsz)
            mb_data.append(
                {
                    "images": images,
                    "condition": self._get_condition(mb, device),
                    "solution": solution,
                    "ce_labels": self._make_ce_labels(solution) if solution is not None else None,
                    "puzzle_ids": puzzle_ids,
                    "noisy": noisy,
                    "timesteps": timesteps,
                    "target": target,
                    "z_H": z_H.to(device),
                    "z_L": z_L.to(device),
                }
            )
        return mb_data

    def _minsnr_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Per-sample min-SNR weights (Hang et al. 2023).

        For x0/sample prediction: w(t) = min(SNR(t), γ) / SNR(t)
          → downweights easy low-noise steps, upweights hard high-noise steps.
        For ε prediction:         w(t) = min(SNR(t), γ)
          → clips the implicit high-SNR dominance from the ε parameterisation.
        """
        gamma = self.train_cfg.minsnr_gamma
        alphas_cumprod = self.scheduler.alphas_cumprod.to(timesteps.device)[timesteps]
        snr = alphas_cumprod / (1.0 - alphas_cumprod).clamp(min=1e-8)
        if self.scheduler.config.prediction_type == "epsilon":
            return snr.clamp(max=gamma)
        else:  # "sample" / x0 prediction
            return snr.clamp(max=gamma) / snr.clamp(min=1e-8)

    def _compute_step_loss(self, noise_pred, logits, d, device, *, include_sudoku=True):
        sudoku_w = self.train_cfg.sudoku_loss_weight
        mse_w = self.train_cfg.mse_loss_weight
        step_loss = torch.tensor(0.0, device=device)
        diff_loss = sudoku_loss = clf_loss = torch.tensor(0.0, device=device)

        if mse_w > 0.0:
            if self.train_cfg.minsnr_gamma is not None:
                per_sample = (noise_pred.float() - d["target"]).pow(2).flatten(1).mean(1)
                w = self._minsnr_weights(d["timesteps"])
                diff_loss = (w * per_sample).mean()
            else:
                diff_loss = F.mse_loss(noise_pred.float(), d["target"])
            step_loss = step_loss + mse_w * diff_loss

        if include_sudoku and logits is not None and sudoku_w > 0 and d.get("ce_labels") is not None:
            B_, N, C = logits.shape
            if N <= d["ce_labels"].shape[1]:
                sudoku_loss = F.cross_entropy(
                    logits.float().reshape(B_ * N, C),
                    d["ce_labels"][:, :N].reshape(B_ * N).clamp(min=0),
                    ignore_index=IGNORE_LABEL_ID,
                )
            step_loss = step_loss + sudoku_w * sudoku_loss

        if self.clf_cfg is not None and self.clf_cfg.weight > 0.0 and self.train_clf is not None:
            x0_pred = x0_from_noise_pred(noise_pred, d["noisy"], d["timesteps"], self.scheduler)
            clf_loss = _classifier_loss(
                x0_pred,
                d["noisy"],
                d["images"],
                d["solution"],
                d["timesteps"],
                self.model_cfg.cell_size,
                self.train_clf,
                self.scheduler,
                self.clf_cfg,
            )
            step_loss = step_loss + self.clf_cfg.weight * clf_loss

        return step_loss, diff_loss, sudoku_loss, clf_loss

    def _train_step_standard(self, micro_batches, accelerator, optimizers, ema, global_batch_size, global_step):
        K = len(micro_batches)
        device = accelerator.device
        mb_data = self._prep_mb_data(micro_batches, device)

        total_diff_loss = 0.0
        total_sudoku_loss = 0.0
        total_clf_loss = 0.0
        lr = 0.0

        for _ in range(self.n_sup):
            for d in mb_data:
                noise_pred, logits, d["z_H"], d["z_L"] = self.reasoning_step(
                    d["condition"],
                    d["noisy"],
                    d["z_H"],
                    d["z_L"],
                    d["timesteps"],
                    d["puzzle_ids"],
                )
                step_loss, diff_loss, sudoku_loss, clf_loss = self._compute_step_loss(noise_pred, logits, d, device)
                total_diff_loss += diff_loss.item()
                total_sudoku_loss += sudoku_loss.item()
                total_clf_loss += clf_loss.item()
                if step_loss.requires_grad:
                    accelerator.backward(step_loss / (global_batch_size * K))

            accelerator.clip_grad_norm_(self.get_thinker_params(), 1.0)
            accelerator.clip_grad_norm_(self.get_painter_params(), 1.0)
            lr = apply_lr_and_step(optimizers, global_step)
            global_step += 1
            if ema is not None:
                ema.update(self)

        n = self.n_sup * K
        losses = {"diff_loss": total_diff_loss / n, "sudoku_loss": total_sudoku_loss / n}
        if self.clf_cfg is not None and self.clf_cfg.weight > 0.0:
            losses["clf_loss"] = total_clf_loss / n
        return losses, lr, global_step

    def _train_step_two_stage(self, micro_batches, accelerator, optimizers, ema, global_batch_size, global_step):
        K = len(micro_batches)
        device = accelerator.device
        ts_cfg = self.train_cfg.two_stage
        ps = ts_cfg.painter
        ts = ts_cfg.thinker
        ps_n_sup = self.n_sup if ps.n_sup < 0 else ps.n_sup
        ts_n_sup = self.n_sup if ts.n_sup < 0 else ts.n_sup
        run_painter_stage = global_step % ps.every == 0
        run_thinker_stage = global_step % ts.every == 0

        mb_data = []
        for mb in micro_batches:
            images = mb["images"].to(device)
            solution = mb["solution"].to(device)
            puzzle_ids = mb["puzzle_id"].to(device) if "puzzle_id" in mb else None
            bsz = images.shape[0]

            noise = torch.randn_like(images)
            timesteps = torch.randint(
                0, self.scheduler.config.num_train_timesteps, (bsz,), device=device, dtype=torch.long
            )
            noisy = self.scheduler.add_noise(images, noise, timesteps)
            target = noise if self.scheduler.config.prediction_type == "epsilon" else images
            z_H_p, z_L_p = self.get_initial_states(bsz)
            z_H_t, z_L_t = self.get_initial_states(bsz)
            mb_data.append(
                {
                    "condition": self._get_condition(mb, device),
                    "teacher": self._get_teacher_condition(mb, device),
                    "ce_labels": self._make_ce_labels(solution),
                    "puzzle_ids": puzzle_ids,
                    "noisy": noisy,
                    "timesteps": timesteps,
                    "target": target,
                    "z_H_p": z_H_p.to(device),
                    "z_L_p": z_L_p.to(device),
                    "z_H_t": z_H_t.to(device),
                    "z_L_t": z_L_t.to(device),
                }
            )

        thinker_optims = optimizers[:-1]
        painter_optims = optimizers[-1:]
        painter_params = self.get_painter_params()

        total_diff_loss = 0.0
        total_sudoku_loss = 0.0
        n_painter_ticks = 0
        n_thinker_ticks = 0
        lr = 0.0
        n_ticks = max(ps_n_sup if run_painter_stage else 0, ts_n_sup if run_thinker_stage else 0)

        for tick in range(n_ticks):
            # Painter sub-step: teacher-forced, updates painter (and optionally thinker)
            if run_painter_stage and tick < ps_n_sup:
                thinker_params = self.get_thinker_params()
                if ps.freeze_thinker:
                    for p in thinker_params:
                        p.requires_grad_(False)

                for d in mb_data:
                    noise_pred, logits, d["z_H_p"], d["z_L_p"] = self.reasoning_step(
                        d["teacher"],
                        d["noisy"],
                        d["z_H_p"],
                        d["z_L_p"],
                        d["timesteps"],
                        d["puzzle_ids"],
                        H_cycles=ps.H_cycles,
                        L_cycles=ps.L_cycles,
                    )
                    step_loss, diff_loss, sudoku_loss, clf_loss = self._compute_step_loss(
                        noise_pred, logits, d, device, include_sudoku=not ps.freeze_thinker
                    )
                    accelerator.backward(step_loss / (global_batch_size * K))
                    total_diff_loss += diff_loss.item()
                    total_sudoku_loss += sudoku_loss.item()
                    n_painter_ticks += 1

                if not ps.freeze_thinker:
                    accelerator.clip_grad_norm_(self.get_thinker_params(), 1.0)
                accelerator.clip_grad_norm_(painter_params, 1.0)
                lr = apply_lr_and_step(painter_optims if ps.freeze_thinker else optimizers, global_step)

                if ps.freeze_thinker:
                    for p in thinker_params:
                        p.requires_grad_(True)

            # Thinker sub-step: real condition, painter frozen, only thinker optims step
            if run_thinker_stage and tick < ts_n_sup:
                for p in painter_params:
                    p.requires_grad_(False)

                for d in mb_data:
                    noise_pred, logits, d["z_H_t"], d["z_L_t"] = self.reasoning_step(
                        d["condition"],
                        d["noisy"],
                        d["z_H_t"],
                        d["z_L_t"],
                        d["timesteps"],
                        d["puzzle_ids"],
                    )
                    step_loss, diff_loss, sudoku_loss, clf_loss = self._compute_step_loss(noise_pred, logits, d, device)
                    accelerator.backward(step_loss / (global_batch_size * K))
                    total_sudoku_loss += sudoku_loss.item()
                    n_thinker_ticks += 1

                accelerator.clip_grad_norm_(self.get_thinker_params(), 1.0)
                lr_t = apply_lr_and_step(thinker_optims, global_step)
                if lr == 0.0:
                    lr = lr_t

                for p in painter_params:
                    p.requires_grad_(True)

            if ema is not None:
                ema.update(self)
            global_step += 1

        n_p = n_painter_ticks or 1
        n_t = (n_painter_ticks + n_thinker_ticks) or 1
        return {"diff_loss": total_diff_loss / n_p, "sudoku_loss": total_sudoku_loss / n_t}, lr, global_step

    def compile_submodules(self):
        self.thinker.inner.L_level = torch.compile(self.thinker.inner.L_level, fullgraph=False)
        self.painter = torch.compile(self.painter)
        if self.bridge is not None:
            self.bridge = torch.compile(self.bridge)

    @torch.no_grad()
    def eval_step(self, dataloader, accelerator, **kwargs) -> dict:
        max_batches = kwargs.get("max_batches", 100)
        self.eval()

        # ── Loss eval (fast, always runs) ─────────────────────────────────────
        metrics: dict[str, list] = {
            "diff_loss": [],
            "sudoku_loss": [],
            "thinker_cell_acc": [],
            "thinker_puzzle_acc": [],
        }
        for i, batch in tqdm(enumerate(dataloader), "Eval loss", total=max_batches):
            if i >= max_batches:
                break
            m = compute_losses_painter(
                model=self,
                condition=self._get_condition(batch, accelerator.device),
                batch=batch,
                scheduler=self.scheduler,
                accelerator=accelerator,
                sudoku_loss_weight=self.train_cfg.sudoku_loss_weight,
                token_input=self.token_input,
            )
            for k in metrics:
                val = m.get(k)
                if val is not None:
                    metrics[k].append(val.item() if torch.is_tensor(val) else float(val))
        result = {k: float(np.mean(v)) for k, v in metrics.items() if v}

        # ── Sampling eval (slow, only when classifier provided) ───────────────
        for cb in self.eval_callbacks:
            result.update(cb(self, dataloader, accelerator, **kwargs))

        self.train()
        return result

    def _logits_to_spatial(self, logits: torch.Tensor) -> torch.Tensor:
        """(B, N, C) logits → (B, C, grid, grid) spatial conditioning.

        Conversion respects self.thinker_bridge_mode:
          "logits"  - raw float logits (default)
          "onehot"  - argmax → one-hot
          "softmax" - softmax probabilities
        """
        B, _, C = logits.shape
        mode = getattr(self, "thinker_bridge_mode", "logits")
        if mode == "onehot":
            # Straight-through estimator: hard one-hot in the forward pass,
            # softmax gradient in the backward pass so thinker weights can update.
            soft = logits.float().softmax(dim=-1)
            hard = F.one_hot(logits.argmax(dim=-1), num_classes=C).float()
            onehot = hard - soft.detach() + soft  # forward≈hard, grad flows via soft
            return onehot.transpose(1, 2).reshape(B, C, self._grid, self._grid)
        elif mode == "softmax":
            probs = logits.float().softmax(dim=-1)
            return probs.transpose(1, 2).reshape(B, C, self._grid, self._grid)
        else:
            return logits.float().transpose(1, 2).reshape(B, C, self._grid, self._grid)

    def _run_painter(
        self,
        noisy: torch.Tensor,
        spatial_cond: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        # Autocast applies to bridge + UNet only; TRM handles its own dtype.
        # Loss is always computed in float32 by callers (.float() before MSE/CE).
        ctx = (
            torch.autocast(device_type=noisy.device.type, dtype=self._painter_dtype)
            if self._painter_dtype is not None
            else torch.autocast(device_type=noisy.device.type, enabled=False)
        )
        with ctx:
            bridge_feat = self.bridge(spatial_cond)
            return self.painter(torch.cat([noisy, bridge_feat], dim=1), timesteps).sample

    def reasoning_step(
        self,
        puzzle_tokens: torch.Tensor,
        noisy: torch.Tensor,
        z_H: torch.Tensor,
        z_L: torch.Tensor,
        timesteps: torch.Tensor,
        puzzle_ids: Optional[torch.Tensor] = None,
        H_cycles: Optional[int] = None,
        L_cycles: Optional[int] = None,
    ):
        """
        One supervision step: thinker → bridge → painter.

        diff_thinker_weight scales the gradient that diffusion loss sends back
        through the bridge into the thinker (1.0 = full, 0.0 = detached).
        The sudoku CE loss always flows through unscaled logits.

        H_cycles / L_cycles: override thinker config for this call only.

        Returns: (noise_pred, sudoku_logits, z_H_detached, z_L_detached)
        """
        logits, z_H_next, z_L_next = self.thinker.reasoning_step(
            puzzle_tokens, z_H, z_L, puzzle_ids, H_cycles=H_cycles, L_cycles=L_cycles
        )
        # spatial_cond in float for bridge (inner model runs in bf16)
        spatial_cond = self._logits_to_spatial(logits.float())

        if self.diff_thinker_weight == 0.0:
            sc_for_painter = spatial_cond.detach()
        elif self.diff_thinker_weight != 1.0:
            sc_for_painter = (
                self.diff_thinker_weight * spatial_cond + (1.0 - self.diff_thinker_weight) * spatial_cond.detach()
            )
        else:
            sc_for_painter = spatial_cond

        # CFG training dropout: randomly zero conditioning per sample.
        if self.training and self.train_cfg.cfg_prob > 0:
            drop = torch.rand(sc_for_painter.shape[0], 1, 1, 1, device=sc_for_painter.device) < self.train_cfg.cfg_prob
            sc_for_painter = sc_for_painter * (~drop)

        noise_pred = self._run_painter(noisy, sc_for_painter, timesteps)
        return noise_pred, logits, z_H_next, z_L_next

    def forward(
        self,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        puzzle_tokens: torch.Tensor,
        puzzle_ids: Optional[torch.Tensor] = None,
    ):
        """
        Full inference: run all n_sup thinker steps, then one painter pass.
        Used for eval (no gradient).
        """
        bsz = noisy.shape[0]
        z_H, z_L = self.get_initial_states(bsz)
        z_H = z_H.to(noisy.device)
        z_L = z_L.to(noisy.device)

        logits = None
        for _ in range(self.n_sup):
            logits, z_H, z_L = self.thinker.reasoning_step(puzzle_tokens, z_H, z_L, puzzle_ids)

        spatial_cond = self._logits_to_spatial(logits.float())

        if not self.training and self.eval_cfg.cfg_scale > 1.0:
            null = torch.zeros_like(spatial_cond)
            pred_cond = self._run_painter(noisy, spatial_cond, timesteps)
            pred_uncond = self._run_painter(noisy, null, timesteps)
            noise_pred = pred_uncond + self.eval_cfg.cfg_scale * (pred_cond - pred_uncond)
        else:
            noise_pred = self._run_painter(noisy, spatial_cond, timesteps)
        return noise_pred, logits


# ── Painter-thinker (V0: image-conditioned) ───────────────────────────────────


class OriginalTRMRatatouilleV0(PainterThinkerV0Tok):
    """
    Image-conditioned painter-thinker (V0).

    Identical to V0Tok except the thinker receives CNN-encoded puzzle image
    features instead of discrete puzzle tokens.  A SpatialEncoder + 1×1 Conv2d
    projects the condition image (B, 1, H, W) to float embeddings
    (B, 81, hidden_size) which are fed directly to _SpatialInputTRMInner
    (bypassing embed_tokens).

    token_input = False  → train_trm uses batch["conditions"] not puzzle_tokens
    token_offset = 0     → logits are already in 0-8 digit space
    """

    token_input: bool = False
    has_realsolution_eval: bool = True  # eval with full MNIST image as condition

    def __init__(
        self,
        thinker_cfg: ThinkerModelConfig,
        encoder_cfg: ImageEncoderConfig,
        model_cfg: PainterThinkerConfig,
        train_cfg: TrainConfig,
        eval_cfg: EvalConfig,
        thinker_optim_cfg: ThinkerOptimConfig,
        painter_optim_cfg: PainterOptimConfig,
        scheduler,
        timestep_cfg: Optional[TimestepCondConfig] = None,
    ):
        # If thinker_grid_size is set, override seq_len before super() creates the TRM.
        if model_cfg.thinker_grid_size is not None:
            thinker_cfg.seq_len = model_cfg.thinker_grid_size * model_cfg.thinker_grid_size

        super().__init__(
            thinker_cfg=thinker_cfg,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            eval_cfg=eval_cfg,
            thinker_optim_cfg=thinker_optim_cfg,
            painter_optim_cfg=painter_optim_cfg,
            scheduler=scheduler,
        )

        # Override _grid set by PainterThinkerV0Tok (painter_size // cell_size).
        if model_cfg.thinker_grid_size is not None:
            self._grid = model_cfg.thinker_grid_size

        self.encoder_cfg = encoder_cfg
        self.token_offset = 0

        self.image_encoder = SpatialEncoder(
            1,
            encoder_cfg.enc_channels,
            factor=model_cfg.cell_size,
            hidden_channels=tuple(encoder_cfg.enc_hidden_channels),
        )
        std = 1.0 / (math.sqrt(thinker_cfg.hidden_size) * math.sqrt(encoder_cfg.enc_channels))
        self.enc_proj = nn.Conv2d(encoder_cfg.enc_channels, thinker_cfg.hidden_size, 1)
        nn.init.normal_(self.enc_proj.weight, std=std)
        nn.init.zeros_(self.enc_proj.bias)

        _toc = (
            encoder_cfg.thinker_out_channels if encoder_cfg.thinker_out_channels is not None else thinker_cfg.vocab_size
        )
        if _toc != thinker_cfg.vocab_size:
            self.logit_expand = nn.Linear(thinker_cfg.vocab_size, _toc, bias=False)
            self.bridge = SpatialBridge(
                in_channels=_toc,
                out_channels=model_cfg.bridge_channels,
                painter_size=model_cfg.painter_size,
            )
        else:
            self.logit_expand = None

        # Timestep conditioning — zero-init so the model starts as identity.
        # enc_film: FiLM on encoder features; thinker_temb_proj: broadcast into
        # thinker token space; dec_film: FiLM on bridge output.
        self.enc_timestep_cond = timestep_cfg.enc_timestep_cond if timestep_cfg is not None else False
        self.thinker_timestep_cond = timestep_cfg.thinker_timestep_cond if timestep_cfg is not None else False
        self.decoder_timestep_cond = timestep_cfg.decoder_timestep_cond if timestep_cfg is not None else False
        if timestep_cfg is not None and (
            timestep_cfg.enc_timestep_cond or timestep_cfg.thinker_timestep_cond or timestep_cfg.decoder_timestep_cond
        ):
            self.timestep_mlp = TimestepMLP(sin_dim=128, out_dim=timestep_cfg.temb_dim)
        if timestep_cfg is not None and timestep_cfg.enc_timestep_cond:
            self.enc_film = nn.Linear(timestep_cfg.temb_dim, 2 * encoder_cfg.enc_channels)
            nn.init.zeros_(self.enc_film.weight)
            nn.init.zeros_(self.enc_film.bias)
        if timestep_cfg is not None and timestep_cfg.thinker_timestep_cond:
            self.thinker_temb_proj = nn.Linear(timestep_cfg.temb_dim, thinker_cfg.hidden_size)
            nn.init.zeros_(self.thinker_temb_proj.weight)
            nn.init.zeros_(self.thinker_temb_proj.bias)
        if timestep_cfg is not None and timestep_cfg.decoder_timestep_cond:
            self.dec_film = nn.Linear(timestep_cfg.temb_dim, 2 * model_cfg.bridge_channels)
            nn.init.zeros_(self.dec_film.weight)
            nn.init.zeros_(self.dec_film.bias)

    def _get_condition(self, mb: dict, device) -> torch.Tensor:
        return mb["conditions"].to(device)

    def _get_teacher_condition(self, mb: dict, device) -> torch.Tensor:
        return mb["images"].to(device)

    def _make_ce_labels(self, solution: torch.Tensor) -> torch.Tensor:
        return solution

    def _logits_to_spatial(self, logits: torch.Tensor) -> torch.Tensor:
        if self.logit_expand is not None:
            logits = self.logit_expand(logits.float())
        return super()._logits_to_spatial(logits)

    def _encode_image(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → float embeddings (B, _grid², hidden_size)"""
        feat = self.image_encoder(x)  # (B, enc_channels, natural_grid, natural_grid)
        proj = self.enc_proj(feat)  # (B, hidden_size, natural_grid, natural_grid)
        if proj.shape[-1] != self._grid:
            proj = F.adaptive_avg_pool2d(proj, (self._grid, self._grid))
        return proj.flatten(2).transpose(1, 2)  # (B, _grid², hidden_size)

    def _prepare_enc_input(
        self, condition: torch.Tensor, noisy: torch.Tensor, timesteps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Return encoder input. V0: condition only. V1 overrides to cat(condition, noisy)."""
        return condition

    def _get_enc_emb(
        self,
        condition: torch.Tensor,
        noisy: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        temb = None
        if (
            timesteps is not None
            and hasattr(self, "timestep_mlp")
            and (self.enc_timestep_cond or self.thinker_timestep_cond)
        ):
            temb = self.timestep_mlp(timesteps)

        feat = self.image_encoder(self._prepare_enc_input(condition, noisy, timesteps=timesteps))
        if temb is not None and self.enc_timestep_cond:
            scale, shift = self.enc_film(temb).chunk(2, dim=1)
            feat = feat * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        proj = self.enc_proj(feat)
        if proj.shape[-1] != self._grid:
            proj = F.adaptive_avg_pool2d(proj, (self._grid, self._grid))
        enc_emb = proj.flatten(2).transpose(1, 2)  # (B, _grid², hidden_size)

        if temb is not None and self.thinker_timestep_cond:
            enc_emb = enc_emb + self.thinker_temb_proj(temb).unsqueeze(1)

        return enc_emb

    def _run_painter(self, noisy: torch.Tensor, spatial_cond: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if not self.decoder_timestep_cond:
            return super()._run_painter(noisy, spatial_cond, timesteps)
        ctx = (
            torch.autocast(device_type=noisy.device.type, dtype=self._painter_dtype)
            if self._painter_dtype is not None
            else torch.autocast(device_type=noisy.device.type, enabled=False)
        )
        with ctx:
            bridge_feat = self.bridge(spatial_cond)
            temb = self.timestep_mlp(timesteps)
            scale, shift = self.dec_film(temb).chunk(2, dim=1)
            bridge_feat = bridge_feat * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
            return self.painter(torch.cat([noisy, bridge_feat], dim=1), timesteps).sample

    def reasoning_step(
        self,
        condition: torch.Tensor,
        noisy: torch.Tensor,
        z_H: torch.Tensor,
        z_L: torch.Tensor,
        timesteps: torch.Tensor,
        puzzle_ids: Optional[torch.Tensor] = None,
        H_cycles: Optional[int] = None,
        L_cycles: Optional[int] = None,
    ):
        enc_emb = self._get_enc_emb(condition, noisy, timesteps=timesteps)
        return super().reasoning_step(
            enc_emb,
            noisy,
            z_H,
            z_L,
            timesteps,
            puzzle_ids=puzzle_ids,
            H_cycles=H_cycles,
            L_cycles=L_cycles,
        )

    def forward(
        self,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
        puzzle_ids: Optional[torch.Tensor] = None,
    ):
        enc_emb = self._get_enc_emb(condition, noisy, timesteps=timesteps)
        bsz = noisy.shape[0]
        z_H, z_L = self.get_initial_states(bsz)
        z_H = z_H.to(noisy.device)
        z_L = z_L.to(noisy.device)

        logits = None
        for _ in range(self.n_sup):
            logits, z_H, z_L = self.thinker.reasoning_step(enc_emb, z_H, z_L, puzzle_ids)

        spatial_cond = self._logits_to_spatial(logits.float())
        if not self.training and self.eval_cfg.cfg_scale > 1.0:
            null = torch.zeros_like(spatial_cond)
            pred_cond = self._run_painter(noisy, spatial_cond, timesteps)
            pred_uncond = self._run_painter(noisy, null, timesteps)
            noise_pred = pred_uncond + self.eval_cfg.cfg_scale * (pred_cond - pred_uncond)
        else:
            noise_pred = self._run_painter(noisy, spatial_cond, timesteps)
        return noise_pred, logits


# ── Painter-thinker (V1: image+noisy-conditioned) ────────────────────────────


class OriginalTRMRatatouilleV1(OriginalTRMRatatouilleV0):
    """
    Same as V0 but the encoder sees cat(condition, noisy_image) (2 channels).

    The thinker reasons from a noisy/corrupted signal, removing the clean-input
    training wheel present in V0.

    Inherits everything from V0; only differences:
      - image_encoder uses SpatialEncoder(2, ...) instead of SpatialEncoder(1, ...)
      - _prepare_enc_input returns cat(condition, noisy) instead of condition only
    """

    def __init__(
        self,
        thinker_cfg: ThinkerModelConfig,
        encoder_cfg: ImageEncoderConfig,
        model_cfg: PainterThinkerConfig,
        train_cfg: TrainConfig,
        eval_cfg: EvalConfig,
        thinker_optim_cfg: ThinkerOptimConfig,
        painter_optim_cfg: PainterOptimConfig,
        scheduler,
        timestep_cfg: Optional[TimestepCondConfig] = None,
    ):
        super().__init__(
            thinker_cfg=thinker_cfg,
            encoder_cfg=encoder_cfg,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            eval_cfg=eval_cfg,
            thinker_optim_cfg=thinker_optim_cfg,
            painter_optim_cfg=painter_optim_cfg,
            scheduler=scheduler,
            timestep_cfg=timestep_cfg,
        )
        # Replace 1-channel encoder with 2-channel (condition + noisy).
        self.image_encoder = SpatialEncoder(
            2,
            encoder_cfg.enc_channels,
            factor=model_cfg.cell_size,
            hidden_channels=tuple(encoder_cfg.enc_hidden_channels),
        )

    def _prepare_enc_input(
        self, condition: torch.Tensor, noisy: torch.Tensor, timesteps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # _enc_noisy_override: set externally (e.g. noisy-guidance eval) to replace
        # the noisy channel fed to the encoder while the UNet still sees the real x_t.
        # Uses the same set-then-reset pattern as SpatialTRM._emb_bias.
        override = getattr(self, "_enc_noisy_override", None)
        if override is not None:
            return torch.cat([condition, override], dim=1)
        p_max = self.train_cfg.noisy_dropout_p_max
        if self.training and p_max > 0.0 and timesteps is not None:
            T = self.scheduler.config.num_train_timesteps
            # p(t) = p_max * (1 - t/T): highest dropout at t=0 (clean, shortcut
            # regime), zero dropout at t=T (pure noise, x_t uninformative anyway).
            t_norm = timesteps.float() / T
            p = p_max * (1.0 - t_norm)
            keep = (torch.rand(p.shape, device=p.device) > p).float()
            noisy = noisy * keep[:, None, None, None]
        return torch.cat([condition, noisy], dim=1)


# ── Painter-thinker (V2: no CE supervision) ───────────────────────────────────


class OriginalTRMRatatouilleV2(OriginalTRMRatatouilleV1):
    """
    Same as V1 but with no sudoku CE loss and unconstrained thinker output channels.

    Training wheel removed: the thinker gets no explicit digit-level supervision.
    The thinker output is a latent spatial map (thinker_out_channels, 9, 9) which
    the bridge upsamples to condition the painter, but its CE loss is suppressed by
    returning None logits so the training loop skips it.

    Use thinker_out_channels=16 (or any value) instead of num_classes=9.
    """

    has_realsolution_eval: bool = False  # latent thinker; no digit-level solution eval

    def reasoning_step(self, condition, noisy, z_H, z_L, timesteps, puzzle_ids=None, H_cycles=None, L_cycles=None):
        noise_pred, _logits, z_H_next, z_L_next = super().reasoning_step(
            condition,
            noisy,
            z_H,
            z_L,
            timesteps,
            puzzle_ids=puzzle_ids,
            H_cycles=H_cycles,
            L_cycles=L_cycles,
        )
        return noise_pred, None, z_H_next, z_L_next

    def forward(self, noisy, timesteps, condition, puzzle_ids=None):
        noise_pred, _logits = super().forward(noisy, timesteps, condition, puzzle_ids)
        return noise_pred, None


# ── Painter-thinker (V3: larger latent, same as V2) ───────────────────────────


class OriginalTRMRatatouilleV3(OriginalTRMRatatouilleV2):
    """
    Same as V2 but with a larger thinker latent (thinker_out_channels=64).

    The only difference from V2 is the default output dimensionality — the
    architecture, loss (no CE), and encoder (condition+noisy) are identical.
    Use this when you want a higher-capacity thinker latent for the bridge.
    """


# ── Painter-thinker (V4: AttentiveBridge + decoupled compression factor) ──────


class OriginalTRMRatatouilleV4(OriginalTRMRatatouilleV3):
    """
    Same as V3 but the thinker grid topology is decoupled from the puzzle cell
    structure via an independent compression_factor, and SpatialBridge is
    replaced by AttentiveBridge (Perceiver-IO cross-attention upsampling).

    Key differences from V3:
      - compression_factor controls encoder downsampling (may differ from cell_size)
      - thinker seq_len = (painter_size // compression_factor)²
      - Bridge: AttentiveBridge with learned positional queries upsamples the
        low-res thinker output to painter_size × painter_size
      - bridge_num_heads: attention heads in AttentiveBridge
    """

    def __init__(
        self,
        thinker_cfg: ThinkerModelConfig,
        encoder_cfg: ImageEncoderConfig,
        model_cfg: PainterThinkerConfig,
        train_cfg: TrainConfig,
        eval_cfg: EvalConfig,
        thinker_optim_cfg: ThinkerOptimConfig,
        painter_optim_cfg: PainterOptimConfig,
        scheduler,
        compression_factor: int = 16,
        bridge_num_heads: int = 4,
        timestep_cfg: Optional[TimestepCondConfig] = None,
    ):
        grid_size = model_cfg.painter_size // compression_factor
        thinker_cfg.seq_len = grid_size * grid_size

        super().__init__(
            thinker_cfg=thinker_cfg,
            encoder_cfg=encoder_cfg,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            eval_cfg=eval_cfg,
            thinker_optim_cfg=thinker_optim_cfg,
            painter_optim_cfg=painter_optim_cfg,
            scheduler=scheduler,
            timestep_cfg=timestep_cfg,
        )

        self._grid = grid_size

        # Replace 2-channel encoder (set by V1) with compression_factor version
        self.image_encoder = SpatialEncoder(
            2,
            encoder_cfg.enc_channels,
            factor=compression_factor,
            hidden_channels=tuple(encoder_cfg.enc_hidden_channels),
        )

        # Replace SpatialBridge with AttentiveBridge
        self.bridge = AttentiveBridge(
            in_channels=thinker_cfg.vocab_size,
            out_channels=model_cfg.bridge_channels,
            out_resolution=model_cfg.painter_size,
            factor=compression_factor,
            num_heads=bridge_num_heads,
        )


# ── Thinker with frozen painter ───────────────────────────────────────────────


class ThinkerWithFrozenPainter(PainterThinkerV0Tok):
    """
    Trains only the thinker; bridge + UNet are loaded from a pretrained
    StandalonePainter checkpoint and kept frozen throughout.

    Inherits everything from OriginalTRMRatatouilleV0Tok except:
      - bridge and painter come from a pre-built StandalonePainter (no new weights)
      - those parameters are frozen (requires_grad=False)
      - get_painter_params() returns [] so the optimizer never touches them

    Usage:
      python train_trm.py experiment=thinker_frozen_painter \\
        painter.painter_checkpoint=runs/standalone_painter/checkpoint_final.pt
    """

    def __init__(
        self,
        painter: StandalonePainter,
        thinker_cfg: ThinkerModelConfig,
        model_cfg: PainterThinkerConfig,
        train_cfg: TrainConfig,
        eval_cfg: EvalConfig,
        thinker_optim_cfg: ThinkerOptimConfig,
        painter_optim_cfg: PainterOptimConfig,
        scheduler,
        adapter_in_channels: int = 0,
        timestep_cfg=None,
    ):
        super().__init__(
            thinker_cfg=thinker_cfg,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            eval_cfg=eval_cfg,
            thinker_optim_cfg=thinker_optim_cfg,
            painter_optim_cfg=painter_optim_cfg,
            scheduler=scheduler,
        )
        # Replace the freshly-built bridge+painter with the pretrained frozen ones.
        self.bridge = painter.bridge
        self.painter = painter.painter
        for p in self.bridge.parameters():
            p.requires_grad_(False)
        for p in self.painter.parameters():
            p.requires_grad_(False)

        # Optional channel-count adaptation at the thinker→bridge interface.
        # When adapter_in_channels != 0 and != the bridge's native input channels:
        #   - A learnable linear projection maps each cell's logits from vocab_size
        #     to adapter_in_channels (handles both fewer and more channels without
        #     information bottlenecks).
        #   - The bridge's first Conv2d is replaced with a new trainable one that
        #     accepts adapter_in_channels; the second conv retains its pretrained
        #     weights.  The bridge's second conv operates in bridge_channels space
        #     and is unaffected by the input channel change.
        native_in = painter.bridge.conv[0].in_channels  # vocab_size the bridge was trained with
        self.logit_projection = None
        self.bridge_input_conv = None
        if adapter_in_channels > 0 and adapter_in_channels != native_in:
            bridge_channels = painter.bridge.conv[0].out_channels
            # Per-cell linear projection on thinker logits: (B,81,vocab) → (B,81,adapter_in)
            self.logit_projection = nn.Linear(thinker_cfg.vocab_size, adapter_in_channels)
            # Replace first bridge conv; second conv (bridge_ch→bridge_ch) stays frozen.
            self.bridge_input_conv = nn.Conv2d(adapter_in_channels, bridge_channels, kernel_size=3, padding=1)

        # Token-based path: no image encoder, so enc_timestep_cond is not applicable.
        # thinker_timestep_cond injects temb as an addend to token embeddings via
        # SpatialTRM.reasoning_step(input_emb_bias=...) — no image encoder needed.
        if timestep_cfg is not None and timestep_cfg.enc_timestep_cond:
            raise ValueError(
                "ThinkerWithFrozenPainter (token-based) does not have an image encoder. " "Set enc_timestep_cond=False."
            )
        self.enc_timestep_cond = False
        self.thinker_timestep_cond = timestep_cfg.thinker_timestep_cond if timestep_cfg is not None else False
        self.decoder_timestep_cond = timestep_cfg.decoder_timestep_cond if timestep_cfg is not None else False
        if timestep_cfg is not None and (self.thinker_timestep_cond or self.decoder_timestep_cond):
            self.timestep_mlp = TimestepMLP(sin_dim=128, out_dim=timestep_cfg.temb_dim)
        if self.thinker_timestep_cond:
            # (B, temb_dim) → (B, 1, hidden_size) broadcast-added to all token positions.
            self.thinker_temb_proj = nn.Linear(timestep_cfg.temb_dim, thinker_cfg.hidden_size)
            nn.init.zeros_(self.thinker_temb_proj.weight)
            nn.init.zeros_(self.thinker_temb_proj.bias)
        if self.decoder_timestep_cond:
            bridge_ch = painter.bridge.conv[0].out_channels
            self.dec_film = nn.Linear(timestep_cfg.temb_dim, 2 * bridge_ch)
            nn.init.zeros_(self.dec_film.weight)
            nn.init.zeros_(self.dec_film.bias)

    def _logits_to_spatial(self, logits: torch.Tensor) -> torch.Tensor:
        if self.logit_projection is not None:
            logits = self.logit_projection(logits)  # (B, 81, adapter_in_channels)
        return super()._logits_to_spatial(logits)

    def _get_thinker_emb_bias(self, timesteps: torch.Tensor) -> Optional[torch.Tensor]:
        """Return (B, 1, hidden_size) temb bias for thinker token embeddings, or None."""
        if not self.thinker_timestep_cond:
            return None
        temb = self.timestep_mlp(timesteps)
        return self.thinker_temb_proj(temb).unsqueeze(1)  # (B, 1, hidden_size)

    def reasoning_step(
        self,
        puzzle_tokens: torch.Tensor,
        noisy: torch.Tensor,
        z_H: torch.Tensor,
        z_L: torch.Tensor,
        timesteps: torch.Tensor,
        puzzle_ids: Optional[torch.Tensor] = None,
        H_cycles: Optional[int] = None,
        L_cycles: Optional[int] = None,
    ):
        emb_bias = self._get_thinker_emb_bias(timesteps)
        logits, z_H_next, z_L_next = self.thinker.reasoning_step(
            puzzle_tokens,
            z_H,
            z_L,
            puzzle_ids,
            H_cycles=H_cycles,
            L_cycles=L_cycles,
            input_emb_bias=emb_bias,
        )
        spatial_cond = self._logits_to_spatial(logits.float())
        if self.diff_thinker_weight == 0.0:
            sc_for_painter = spatial_cond.detach()
        elif self.diff_thinker_weight != 1.0:
            sc_for_painter = (
                self.diff_thinker_weight * spatial_cond + (1.0 - self.diff_thinker_weight) * spatial_cond.detach()
            )
        else:
            sc_for_painter = spatial_cond
        if self.training and self.train_cfg.cfg_prob > 0:
            drop = torch.rand(sc_for_painter.shape[0], 1, 1, 1, device=sc_for_painter.device) < self.train_cfg.cfg_prob
            sc_for_painter = sc_for_painter * (~drop)
        noise_pred = self._run_painter(noisy, sc_for_painter, timesteps)
        return noise_pred, logits, z_H_next, z_L_next

    def _run_painter(self, noisy, spatial_cond, timesteps):
        if not self.bridge_input_conv and not self.decoder_timestep_cond:
            return super()._run_painter(noisy, spatial_cond, timesteps)
        ctx = (
            torch.autocast(device_type=noisy.device.type, dtype=self._painter_dtype)
            if self._painter_dtype is not None
            else torch.autocast(device_type=noisy.device.type, enabled=False)
        )
        with ctx:
            if self.bridge_input_conv is not None:
                spatial_cond = F.interpolate(
                    spatial_cond, size=self.bridge.painter_size, mode="bilinear", align_corners=False
                )
                spatial_cond = torch.nn.functional.silu(self.bridge_input_conv(spatial_cond))
                bridge_feat = self.bridge.conv[2](spatial_cond)
            else:
                bridge_feat = self.bridge(spatial_cond)
            if self.decoder_timestep_cond:
                temb = self.timestep_mlp(timesteps)
                scale, shift = self.dec_film(temb).chunk(2, dim=1)
                bridge_feat = bridge_feat * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
            return self.painter(torch.cat([noisy, bridge_feat], dim=1), timesteps).sample

    def forward(
        self,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        puzzle_tokens: torch.Tensor,
        puzzle_ids: Optional[torch.Tensor] = None,
    ):
        bsz = noisy.shape[0]
        z_H, z_L = self.get_initial_states(bsz)
        z_H = z_H.to(noisy.device)
        z_L = z_L.to(noisy.device)
        emb_bias = self._get_thinker_emb_bias(timesteps)
        logits = None
        for _ in range(self.n_sup):
            logits, z_H, z_L = self.thinker.reasoning_step(puzzle_tokens, z_H, z_L, puzzle_ids, input_emb_bias=emb_bias)
        spatial_cond = self._logits_to_spatial(logits.float())
        if not self.training and self.eval_cfg.cfg_scale > 1.0:
            null = torch.zeros_like(spatial_cond)
            pred_cond = self._run_painter(noisy, spatial_cond, timesteps)
            pred_uncond = self._run_painter(noisy, null, timesteps)
            noise_pred = pred_uncond + self.eval_cfg.cfg_scale * (pred_cond - pred_uncond)
        else:
            noise_pred = self._run_painter(noisy, spatial_cond, timesteps)
        return noise_pred, logits

    def get_painter_params(self) -> list:
        return []  # frozen — excluded from all optimizers

    def build_optimizers(self, world_size, num_steps) -> list[ScheduledOptimizer]:
        return self.thinker.build_optimizers(world_size, num_steps)

    def get_thinker_params(self) -> list:
        params = super().get_thinker_params()
        if self.logit_projection is not None:
            params = params + list(self.logit_projection.parameters())
        if self.bridge_input_conv is not None:
            params = params + list(self.bridge_input_conv.parameters())
        if hasattr(self, "timestep_mlp"):
            params = params + list(self.timestep_mlp.parameters())
        if hasattr(self, "thinker_temb_proj"):
            params = params + list(self.thinker_temb_proj.parameters())
        if hasattr(self, "dec_film"):
            params = params + list(self.dec_film.parameters())
        return params


# ── Thinker with frozen painter (image-conditioned variants) ──────────────────


class ThinkerWithFrozenPainterV0(OriginalTRMRatatouilleV0):
    """
    Trains thinker + image encoder; bridge + UNet are loaded from a pretrained
    StandalonePainter checkpoint and kept frozen throughout.

    Compared to ThinkerWithFrozenPainter (V0Tok), the thinker receives CNN-encoded
    puzzle image features instead of discrete tokens — so the image_encoder and
    enc_proj are trainable new parameters optimized with painter_optim_cfg.
    """

    def __init__(
        self,
        painter: StandalonePainter,
        thinker_cfg: ThinkerModelConfig,
        encoder_cfg: ImageEncoderConfig,
        model_cfg: PainterThinkerConfig,
        train_cfg: TrainConfig,
        eval_cfg: EvalConfig,
        thinker_optim_cfg: ThinkerOptimConfig,
        painter_optim_cfg: PainterOptimConfig,
        scheduler,
        timestep_cfg=None,
    ):
        super().__init__(
            thinker_cfg=thinker_cfg,
            encoder_cfg=encoder_cfg,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            eval_cfg=eval_cfg,
            thinker_optim_cfg=thinker_optim_cfg,
            painter_optim_cfg=painter_optim_cfg,
            scheduler=scheduler,
            timestep_cfg=timestep_cfg,
        )
        self.painter_optim_cfg = painter_optim_cfg
        self.bridge = painter.bridge
        self.painter = painter.painter
        for p in self.bridge.parameters():
            p.requires_grad_(False)
        for p in self.painter.parameters():
            p.requires_grad_(False)

    def get_painter_params(self) -> list:
        return []

    def _get_encoder_params(self) -> list:
        frozen_ids = {id(p) for p in self.bridge.parameters()} | {id(p) for p in self.painter.parameters()}
        thinker_ids = {id(p) for p in self.thinker.parameters()}
        return [
            p for p in self.parameters() if id(p) not in frozen_ids and id(p) not in thinker_ids and p.requires_grad
        ]

    def build_optimizers(self, world_size, num_steps) -> list[ScheduledOptimizer]:
        thinker_optims = self.thinker.build_optimizers(world_size, num_steps)
        encoder_params = self._get_encoder_params()
        if not encoder_params:
            return thinker_optims
        enc_optim = torch.optim.AdamW(encoder_params, lr=0, weight_decay=self.painter_optim_cfg.weight_decay)
        enc_scheduled = ScheduledOptimizer(
            enc_optim,
            base_lr=self.painter_optim_cfg.lr,
            warmup_steps=self.painter_optim_cfg.warmup_steps,
            num_steps=num_steps,
            min_ratio=self.painter_optim_cfg.lr_min_ratio,
        )
        return thinker_optims + [enc_scheduled]


class ThinkerWithFrozenPainterV1(OriginalTRMRatatouilleV1):
    """
    Same as ThinkerWithFrozenPainterV0 but the encoder sees cat(condition, noisy)
    (2 channels) and optionally uses timestep conditioning — matching V1 semantics.

    Trainable: thinker, image_encoder, enc_proj, logit_expand (if present),
               and any V1 timestep modules (timestep_mlp, enc_film, thinker_temb_proj).
    Frozen: bridge + UNet (loaded from StandalonePainter checkpoint).
    """

    def __init__(
        self,
        painter: StandalonePainter,
        thinker_cfg: ThinkerModelConfig,
        encoder_cfg: ImageEncoderConfig,
        model_cfg: PainterThinkerConfig,
        train_cfg: TrainConfig,
        eval_cfg: EvalConfig,
        thinker_optim_cfg: ThinkerOptimConfig,
        painter_optim_cfg: PainterOptimConfig,
        scheduler,
        timestep_cfg=None,
    ):
        super().__init__(
            thinker_cfg=thinker_cfg,
            encoder_cfg=encoder_cfg,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            eval_cfg=eval_cfg,
            thinker_optim_cfg=thinker_optim_cfg,
            painter_optim_cfg=painter_optim_cfg,
            scheduler=scheduler,
            timestep_cfg=timestep_cfg,
        )
        self.painter_optim_cfg = painter_optim_cfg
        self.bridge = painter.bridge
        self.painter = painter.painter
        for p in self.bridge.parameters():
            p.requires_grad_(False)
        for p in self.painter.parameters():
            p.requires_grad_(False)

    def get_painter_params(self) -> list:
        return []

    def _get_encoder_params(self) -> list:
        frozen_ids = {id(p) for p in self.bridge.parameters()} | {id(p) for p in self.painter.parameters()}
        thinker_ids = {id(p) for p in self.thinker.parameters()}
        return [
            p for p in self.parameters() if id(p) not in frozen_ids and id(p) not in thinker_ids and p.requires_grad
        ]

    def build_optimizers(self, world_size, num_steps) -> list[ScheduledOptimizer]:
        thinker_optims = self.thinker.build_optimizers(world_size, num_steps)
        encoder_params = self._get_encoder_params()
        if not encoder_params:
            return thinker_optims
        enc_optim = torch.optim.AdamW(encoder_params, lr=0, weight_decay=self.painter_optim_cfg.weight_decay)
        enc_scheduled = ScheduledOptimizer(
            enc_optim,
            base_lr=self.painter_optim_cfg.lr,
            warmup_steps=self.painter_optim_cfg.warmup_steps,
            num_steps=num_steps,
            min_ratio=self.painter_optim_cfg.lr_min_ratio,
        )
        return thinker_optims + [enc_scheduled]


# ── Thinker with frozen painter + self-verification head ─────────────────────


class ThinkerWithFrozenPainterV1Verif(ThinkerWithFrozenPainterV1):
    """
    ThinkerWithFrozenPainterV1 augmented with a binary self-verification head.

    The thinker is trained to additionally predict whether a given (condition,
    x_noisy) pair is consistent — i.e., whether x_noisy looks like it was
    produced from the correct solution for this puzzle.

    Positive examples: real condition + noise(real_image).
    Negative examples: corrupted condition + same noise(real_image).
      Corruption options (one chosen at random per sample, 1-5 cells):
        1. swap_given   — replace a given cell's patch with a different digit's patch
        2. add_wrong    — add a wrong digit patch to a blank cell
        3. swap_positions — swap two given cells with different digits

    The verification head is a Linear(hidden_size → 1) on mean-pooled z_H after
    all n_sup thinker reasoning steps. Both positive and negative examples share
    the same x_noisy; only the condition image differs.
    """

    def __init__(
        self,
        painter: StandalonePainter,
        thinker_cfg: ThinkerModelConfig,
        encoder_cfg: ImageEncoderConfig,
        model_cfg: PainterThinkerConfig,
        train_cfg: TrainConfig,
        eval_cfg: EvalConfig,
        thinker_optim_cfg: ThinkerOptimConfig,
        painter_optim_cfg: PainterOptimConfig,
        scheduler,
        timestep_cfg=None,
        verif_weight: float = 0.1,
        verif_max_corruptions: int = 5,
    ):
        super().__init__(
            painter=painter,
            thinker_cfg=thinker_cfg,
            encoder_cfg=encoder_cfg,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            eval_cfg=eval_cfg,
            thinker_optim_cfg=thinker_optim_cfg,
            painter_optim_cfg=painter_optim_cfg,
            scheduler=scheduler,
            timestep_cfg=timestep_cfg,
        )
        self.verif_weight = verif_weight
        self.verif_max_corruptions = verif_max_corruptions
        self.verif_head = nn.Linear(thinker_cfg.hidden_size, 1)
        nn.init.zeros_(self.verif_head.weight)
        nn.init.zeros_(self.verif_head.bias)

    def _corrupt_condition(
        self,
        conditions: torch.Tensor,
        solution: torch.Tensor,
        given_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Replace 1-5 cell patches in each condition image to create mismatches.

        conditions:  (B, 1, H, W) float
        solution:    (B, 81) int64, 0-8 or -100
        given_mask:  (B, 81) bool — True = cell is shown in condition
        """
        B = conditions.shape[0]
        cs = self.model_cfg.cell_size
        corrupted = conditions.clone()

        for b in range(B):
            n_corrupt = torch.randint(1, self.verif_max_corruptions + 1, (1,)).item()
            ctype = torch.randint(3, (1,)).item()  # 0=swap_given, 1=add_wrong, 2=swap_pos

            given_indices = given_mask[b].nonzero(as_tuple=True)[0]
            blank_indices = (~given_mask[b]).nonzero(as_tuple=True)[0]

            if ctype == 0 and len(given_indices) >= 1:
                cells = given_indices[torch.randperm(len(given_indices), device=given_indices.device)[:n_corrupt]]
                for cell_idx in cells:
                    cell_idx = cell_idx.item()
                    row, col = divmod(cell_idx, 9)
                    orig_digit = solution[b, cell_idx].item()
                    candidates = [
                        b2
                        for b2 in range(B)
                        if b2 != b
                        and given_mask[b2, cell_idx].item()
                        and solution[b2, cell_idx].item() != orig_digit
                        and solution[b2, cell_idx].item() >= 0
                    ]
                    if not candidates:
                        continue
                    b2 = candidates[torch.randint(len(candidates), (1,)).item()]
                    r0, c0 = row * cs, col * cs
                    corrupted[b, :, r0 : r0 + cs, c0 : c0 + cs] = conditions[b2, :, r0 : r0 + cs, c0 : c0 + cs]

            elif ctype == 1 and len(blank_indices) >= 1:
                cells = blank_indices[torch.randperm(len(blank_indices), device=blank_indices.device)[:n_corrupt]]
                for cell_idx in cells:
                    cell_idx = cell_idx.item()
                    row, col = divmod(cell_idx, 9)
                    real_digit = solution[b, cell_idx].item()
                    candidates = [
                        b2
                        for b2 in range(B)
                        if b2 != b
                        and given_mask[b2, cell_idx].item()
                        and solution[b2, cell_idx].item() != real_digit
                        and solution[b2, cell_idx].item() >= 0
                    ]
                    if not candidates:
                        continue
                    b2 = candidates[torch.randint(len(candidates), (1,)).item()]
                    r0, c0 = row * cs, col * cs
                    corrupted[b, :, r0 : r0 + cs, c0 : c0 + cs] = conditions[b2, :, r0 : r0 + cs, c0 : c0 + cs]

            elif len(given_indices) >= 2:
                n = min(n_corrupt * 2, len(given_indices))
                cells = given_indices[torch.randperm(len(given_indices), device=given_indices.device)[:n]]
                for i in range(0, len(cells) - 1, 2):
                    c1, c2 = cells[i].item(), cells[i + 1].item()
                    if solution[b, c1].item() != solution[b, c2].item():
                        r1, col1 = divmod(c1, 9)
                        r2, col2 = divmod(c2, 9)
                        patch1 = corrupted[b, :, r1 * cs : r1 * cs + cs, col1 * cs : col1 * cs + cs].clone()
                        corrupted[b, :, r1 * cs : r1 * cs + cs, col1 * cs : col1 * cs + cs] = conditions[
                            b, :, r2 * cs : r2 * cs + cs, col2 * cs : col2 * cs + cs
                        ]
                        corrupted[b, :, r2 * cs : r2 * cs + cs, col2 * cs : col2 * cs + cs] = patch1

        return corrupted

    def forward_with_verif(
        self,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
        puzzle_ids=None,
        n_sup_grad: int = 1,
    ):
        """
        Eval forward that returns (noise_pred, sudoku_logits, verif_score).

        noise_pred is obtained via the standard forward() so it is identical to
        what training eval produces — no risk of subtle divergence.

        verif_score is computed by a separate thinker pass where the last step
        keeps z_H undetached, enabling gradient back to noisy for classifier guidance.

        n_sup_grad — steps in the gradient graph for verif_score:
          1  (default): only the last step with grad (cheap)
         -1           : all n_sup steps with grad (accurate, expensive)

        Returns: (noise_pred, sudoku_logits, verif_score)
          verif_score: (B,) float in (0, 1) — P(condition consistent with noisy).
        """
        # Noise prediction: delegate entirely to the inherited forward so it is
        # identical to training eval (correct CFG, correct thinker path, etc.)
        noise_pred, logits = self(noisy, timesteps, condition, puzzle_ids=puzzle_ids)

        # Verifier score: separate thinker forward, last step keeps z_H gradient.
        B = noisy.shape[0]
        z_H, z_L = self.get_initial_states(B)
        z_H, z_L = z_H.to(noisy.device), z_L.to(noisy.device)
        enc_emb = self._get_enc_emb(condition, noisy, timesteps=timesteps)

        n_steps = self.n_sup
        n_no_grad = 0 if n_sup_grad < 0 else max(0, n_steps - n_sup_grad)

        if n_no_grad > 0:
            with torch.no_grad():
                for _ in range(n_no_grad):
                    _, z_H, z_L = self.thinker.reasoning_step(enc_emb, z_H, z_L, puzzle_ids)
            z_H = z_H.detach()
            z_L = z_L.detach()

        n_with_grad = n_steps - n_no_grad
        for step in range(n_with_grad):
            is_last = step == n_with_grad - 1
            _, z_H, z_L = self.thinker.reasoning_step(enc_emb, z_H, z_L, puzzle_ids, keep_carry_grad=is_last)

        seq_len = self.thinker.inner.config.seq_len
        z_H_feats = z_H[:, :seq_len, :].float().mean(dim=1)  # (B, hidden_size)
        verif_score = torch.sigmoid(self.verif_head(z_H_feats).squeeze(-1))  # (B,)

        return noise_pred, logits, verif_score

    def _thinker_forward(
        self,
        condition: torch.Tensor,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Run encoder + n_sup thinker steps; return mean-pooled z_H. No painter."""
        B = noisy.shape[0]
        z_H, z_L = self.get_initial_states(B)
        z_H, z_L = z_H.to(noisy.device), z_L.to(noisy.device)
        enc_emb = self._get_enc_emb(condition, noisy, timesteps=timesteps)
        for _ in range(self.n_sup):
            _, z_H, z_L = self.thinker.reasoning_step(enc_emb, z_H, z_L)
        seq_len = self.thinker.inner.config.seq_len
        return z_H[:, :seq_len, :].float().mean(dim=1)  # (B, hidden_size)

    def _train_step_standard(self, micro_batches, accelerator, optimizers, ema, global_batch_size, global_step):
        K = len(micro_batches)
        device = accelerator.device
        mb_data = self._prep_mb_data(micro_batches, device)

        # Pre-compute corrupted conditions and negative encoder embeddings once.
        # Negative z_H/z_L are stepped through the supervision loop alongside
        # positives (one thinker step per iteration), avoiding the n_sup-step
        # burst that _thinker_forward would otherwise create at the final step.
        # Build corrupted conditions and init neg carry states.
        # enc_emb_neg is NOT cached here — it is recomputed fresh at each
        # supervision step so the encoder activations are freed by that step's
        # backward instead of being held for the entire n_sup loop.
        corrupted_conds = []
        neg_states = []
        for i, mb in enumerate(micro_batches):
            d = mb_data[i]
            given_mask = mb.get("given_mask")
            if given_mask is not None:
                corrupted = self._corrupt_condition(d["condition"], d["solution"], given_mask.to(device))
            else:
                corrupted = d["condition"]
            corrupted_conds.append(corrupted)
            B = d["noisy"].shape[0]
            z_H_neg, z_L_neg = self.get_initial_states(B)
            neg_states.append({"z_H": z_H_neg.to(device), "z_L": z_L_neg.to(device)})

        total_diff_loss = 0.0
        total_sudoku_loss = 0.0
        total_verif_loss = 0.0
        lr = 0.0

        for _ in range(self.n_sup):
            for i, d in enumerate(mb_data):
                noise_pred, logits, d["z_H"], d["z_L"] = self.reasoning_step(
                    d["condition"], d["noisy"], d["z_H"], d["z_L"], d["timesteps"], d["puzzle_ids"]
                )
                step_loss, diff_loss, sudoku_loss, clf_loss = self._compute_step_loss(noise_pred, logits, d, device)
                total_diff_loss += diff_loss.item()
                total_sudoku_loss += sudoku_loss.item()

                # Detach z_H_pos before the first backward so it survives as a
                # leaf for the verif head.  Gradient through the positive verif
                # path to the thinker is intentionally dropped: the diff+sudoku
                # loss already fully supervises the thinker on positive examples,
                # and the verif signal is more informative through the negative
                # path (corrupted conditions have no other loss).
                seq_len = self.thinker.inner.config.seq_len
                z_H_pos = d["z_H"][:, :seq_len, :].float().mean(dim=1).detach()
                B = z_H_pos.shape[0]

                # First backward: frees the ~2 GB of pos encoder/thinker/painter
                # activations before we allocate the neg encoder below.
                accelerator.backward(step_loss / (global_batch_size * K))

                # Neg step runs in the memory freed by the first backward.
                # enc_emb_neg is created fresh each step so the encoder activations
                # are freed by the second backward (not held across n_sup steps).
                ns = neg_states[i]
                enc_emb_neg = self._get_enc_emb(corrupted_conds[i], d["noisy"], timesteps=d["timesteps"])
                _, ns["z_H"], ns["z_L"] = self.thinker.reasoning_step(enc_emb_neg, ns["z_H"], ns["z_L"])
                z_H_neg = ns["z_H"][:, :seq_len, :].float().mean(dim=1)

                # Second backward: verif gradients accumulate on top of step 1.
                # Gradient flows through the neg path to the thinker and encoder.
                pos_verif = F.binary_cross_entropy_with_logits(
                    self.verif_head(z_H_pos).squeeze(-1), torch.ones(B, device=device)
                )
                neg_verif = F.binary_cross_entropy_with_logits(
                    self.verif_head(z_H_neg).squeeze(-1), torch.zeros(B, device=device)
                )
                verif_loss = (pos_verif + neg_verif) / 2
                accelerator.backward(self.verif_weight * verif_loss / (global_batch_size * K))
                total_verif_loss += verif_loss.item()

            accelerator.clip_grad_norm_(self.get_thinker_params(), 1.0)
            accelerator.clip_grad_norm_(self.get_painter_params(), 1.0)
            lr = apply_lr_and_step(optimizers, global_step)
            global_step += 1
            if ema is not None:
                ema.update(self)

        n = self.n_sup * K
        return (
            {
                "diff_loss": total_diff_loss / n,
                "sudoku_loss": total_sudoku_loss / n,
                "verif_loss": total_verif_loss / n,
            },
            lr,
            global_step,
        )

    @torch.no_grad()
    def eval_step(self, dataloader, accelerator, **kwargs) -> dict:
        result = super().eval_step(dataloader, accelerator, **kwargs)

        max_batches = kwargs.get("max_batches", 20)
        self.eval()
        correct_pos, correct_neg, total = 0, 0, 0
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            device = accelerator.device
            condition = self._get_condition(batch, device)
            noisy_imgs = batch["images"].to(device)
            given_mask = batch.get("given_mask")
            solution = batch["solution"].to(device)

            noise = torch.randn_like(noisy_imgs)
            timesteps = torch.randint(
                0, self.scheduler.config.num_train_timesteps, (noisy_imgs.shape[0],), device=device
            )
            noisy = self.scheduler.add_noise(noisy_imgs, noise, timesteps)

            z_H_pos = self._thinker_forward(condition, noisy, timesteps)
            if given_mask is not None:
                corrupted = self._corrupt_condition(condition, solution, given_mask.to(device))
                z_H_neg = self._thinker_forward(corrupted, noisy, timesteps)
            else:
                z_H_neg = z_H_pos

            pred_pos = self.verif_head(z_H_pos).squeeze(-1) > 0
            pred_neg = self.verif_head(z_H_neg).squeeze(-1) > 0
            correct_pos += pred_pos.sum().item()
            correct_neg += (~pred_neg).sum().item()
            total += noisy_imgs.shape[0]

        if total > 0:
            result["verif_pos_acc"] = correct_pos / total
            result["verif_neg_acc"] = correct_neg / total
            result["verif_acc"] = (correct_pos + correct_neg) / (2 * total)
        self.train()
        return result


# ── Thinker with frozen ControlNet painter (image-conditioned) ────────────────


class ThinkerWithFrozenPainterControlNet(OriginalTRMRatatouilleV0):
    """
    V0-style thinker (image-conditioned encoder) attached to a frozen
    ControlNet painter loaded from a StandalonePainterControl checkpoint.

    The thinker reasons over the puzzle condition image and produces logits
    that are converted to a spatial control map, which is injected into the
    frozen denoising UNet via ControlNet-style residuals.

    Compared to ThinkerWithFrozenPainterV0:
      - No bridge: replaced by a fresh trainable ConditioningPyramid.
      - Frozen painter: ControlPainterUNet (in_channels=1, no bridge concat).
      - The checkpoint's ConditioningPyramid is NOT reused — it was trained on
        solution tokens, not thinker logits.

    Trainable: thinker, image_encoder, enc_proj, logit_expand (if any),
               control_pyramid.
    Frozen:    ControlPainterUNet.
    """

    def __init__(
        self,
        painter: StandalonePainterControl,
        thinker_cfg: ThinkerModelConfig,
        encoder_cfg: ImageEncoderConfig,
        model_cfg: PainterThinkerConfig,
        train_cfg: TrainConfig,
        eval_cfg: EvalConfig,
        thinker_optim_cfg: ThinkerOptimConfig,
        painter_optim_cfg: PainterOptimConfig,
        scheduler,
    ):
        super().__init__(
            thinker_cfg=thinker_cfg,
            encoder_cfg=encoder_cfg,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            eval_cfg=eval_cfg,
            thinker_optim_cfg=thinker_optim_cfg,
            painter_optim_cfg=painter_optim_cfg,
            scheduler=scheduler,
        )
        self.painter_optim_cfg = painter_optim_cfg

        # Discard the bridge built by the parent (not needed for ControlNet).
        self.bridge = None

        # Freeze the denoising UNet from the checkpoint.
        self.painter = painter.painter
        for p in self.painter.parameters():
            p.requires_grad_(False)

        # Fresh ConditioningPyramid — trained to convert thinker logits to
        # control residuals.  Channel count matches _logits_to_spatial output.
        ctrl_in_ch = (
            encoder_cfg.thinker_out_channels if encoder_cfg.thinker_out_channels is not None else thinker_cfg.vocab_size
        )
        self.control_pyramid = ConditioningPyramid(
            in_channels=ctrl_in_ch,
            block_out_channels=tuple(model_cfg.painter_channels),
            layers_per_block=model_cfg.painter_layers_per_block,
        )

    def _run_painter(
        self,
        noisy: torch.Tensor,
        spatial_cond: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        # spatial_cond: (B, C, 9, 9) from _logits_to_spatial — upsample to painter_size.
        spatial_cond = F.interpolate(
            spatial_cond, size=self.model_cfg.painter_size, mode="bilinear", align_corners=False
        )
        ctx = (
            torch.autocast(device_type=noisy.device.type, dtype=self._painter_dtype)
            if self._painter_dtype is not None
            else torch.autocast(device_type=noisy.device.type, enabled=False)
        )
        with ctx:
            down_res, mid_res = self.control_pyramid(spatial_cond)
            return self.painter(
                noisy,
                timesteps,
                down_block_additional_residuals=down_res,
                mid_block_additional_residual=mid_res,
            ).sample

    def get_painter_params(self) -> list:
        return []

    def _get_encoder_params(self) -> list:
        frozen_ids = {id(p) for p in self.painter.parameters()}
        thinker_ids = {id(p) for p in self.thinker.parameters()}
        return [
            p for p in self.parameters() if id(p) not in frozen_ids and id(p) not in thinker_ids and p.requires_grad
        ]

    def build_optimizers(self, world_size, num_steps) -> list[ScheduledOptimizer]:
        thinker_optims = self.thinker.build_optimizers(world_size, num_steps)
        encoder_params = self._get_encoder_params()
        if not encoder_params:
            return thinker_optims
        enc_optim = torch.optim.AdamW(encoder_params, lr=0, weight_decay=self.painter_optim_cfg.weight_decay)
        enc_scheduled = ScheduledOptimizer(
            enc_optim,
            base_lr=self.painter_optim_cfg.lr,
            warmup_steps=self.painter_optim_cfg.warmup_steps,
            num_steps=num_steps,
            min_ratio=self.painter_optim_cfg.lr_min_ratio,
        )
        return thinker_optims + [enc_scheduled]

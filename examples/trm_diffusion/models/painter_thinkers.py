from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.trm_wrappers import SpatialTRM
from models.utility_models import SpatialEncoder, AttentiveBridge, SpatialBridge, TimestepMLP
import numpy as np
from tqdm.auto import tqdm

from models.painters import make_painter, StandalonePainter, compute_losses_painter
from models.optim_utils import ScheduledOptimizer, apply_lr_and_step
from models.diffusion_utils import apply_noisy_swap
from datasets.sudoku_dataset import IGNORE_LABEL_ID, make_tok_labels
from eval.mnist_eval import (
    sample_grids,
    evaluate_grids,
    load_or_train_classifier,
    make_panel_image,
    plot_thinker_ts_curve,
)

try:
    import wandb as _wandb
except ImportError:
    _wandb = None
from configs.schemas import (
    ThinkerModelConfig,
    PainterThinkerConfig,
    ImageEncoderConfig,
    TimestepCondConfig,
    ThinkerOptimConfig,
    PainterOptimConfig,
    TrainConfig,
    EvalConfig,
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

        self.eval_clf = None
        if eval_cfg.classifier_path is not None:
            self.eval_clf = load_or_train_classifier(eval_cfg.classifier_path, None, model_cfg.cell_size, "cuda")
            for p in self.eval_clf.parameters():
                p.requires_grad_(False)

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
            solution = mb["solution"].to(device)
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
                    "condition": self._get_condition(mb, device),
                    "solution": solution,
                    "ce_labels": self._make_ce_labels(solution),
                    "puzzle_ids": puzzle_ids,
                    "noisy": noisy,
                    "timesteps": timesteps,
                    "target": target,
                    "z_H": z_H.to(device),
                    "z_L": z_L.to(device),
                }
            )
        return mb_data

    def _compute_step_loss(self, noise_pred, logits, d, device, *, include_sudoku=True):
        sudoku_w = self.train_cfg.sudoku_loss_weight
        mse_w = self.train_cfg.mse_loss_weight
        step_loss = torch.tensor(0.0, device=device)
        diff_loss = sudoku_loss = torch.tensor(0.0, device=device)

        if mse_w > 0.0:
            diff_loss = F.mse_loss(noise_pred.float(), d["target"])
            step_loss = step_loss + mse_w * diff_loss

        if include_sudoku and logits is not None and sudoku_w > 0:
            B_, N, C = logits.shape
            sudoku_loss = F.cross_entropy(
                logits.float().reshape(B_ * N, C),
                d["ce_labels"][:, :N].reshape(B_ * N).clamp(min=0),
                ignore_index=IGNORE_LABEL_ID,
            )
            step_loss = step_loss + sudoku_w * sudoku_loss

        return step_loss, diff_loss, sudoku_loss

    def _train_step_standard(self, micro_batches, accelerator, optimizers, ema, global_batch_size, global_step):
        K = len(micro_batches)
        device = accelerator.device
        mb_data = self._prep_mb_data(micro_batches, device)

        total_diff_loss = 0.0
        total_sudoku_loss = 0.0
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
                step_loss, diff_loss, sudoku_loss = self._compute_step_loss(noise_pred, logits, d, device)
                accelerator.backward(step_loss / (global_batch_size * K))
                total_diff_loss += diff_loss.item()
                total_sudoku_loss += sudoku_loss.item()

            accelerator.clip_grad_norm_(self.get_thinker_params(), 1.0)
            accelerator.clip_grad_norm_(self.get_painter_params(), 1.0)
            lr = apply_lr_and_step(optimizers, global_step)
            global_step += 1
            if ema is not None:
                ema.update(self)

        n = self.n_sup * K
        return {"diff_loss": total_diff_loss / n, "sudoku_loss": total_sudoku_loss / n}, lr, global_step

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
                    step_loss, diff_loss, sudoku_loss = self._compute_step_loss(
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
                    step_loss, diff_loss, sudoku_loss = self._compute_step_loss(noise_pred, logits, d, device)
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
        if self.eval_clf is not None and accelerator.is_main_process:
            cell_size = self.model_cfg.cell_size
            painter_size = self.model_cfg.painter_size
            n_total = self.eval_cfg.num_samples
            n_ddim = self.eval_cfg.num_ddim_steps
            n_log = self.eval_cfg.num_log_images
            token_offset = getattr(self, "token_offset", 0)

            all_cell_acc, all_puzzle_acc = [], []
            all_thinker_cell_best, all_thinker_cell_mean = [], []
            all_thinker_puzzle_best, all_thinker_puzzle_mean = [], []
            all_thinker_deviation = []
            all_painter_dev_best, all_painter_dev_mean = [], []
            ts_cell_accs: dict[int, list] = {}
            ts_puzzle_accs: dict[int, list] = {}
            panels: list = []
            n_done = 0

            n_batches = (n_total + self.eval_cfg.batch_size - 1) // self.eval_cfg.batch_size
            for batch in tqdm(dataloader, "Sampling eval", total=n_batches):
                if n_done >= n_total:
                    break
                solutions = batch["solution"]
                given_masks = batch.get("given_mask")
                puzzle_ids = batch.get("puzzle_id")
                if puzzle_ids is not None:
                    puzzle_ids = puzzle_ids.to(accelerator.device)
                B_cur = solutions.shape[0]

                sr = sample_grids(
                    self,
                    self._get_condition(batch, accelerator.device),
                    num_train_timesteps=self.scheduler.config.num_train_timesteps,
                    beta_schedule=self.scheduler.config.beta_schedule,
                    prediction_type=self.scheduler.config.prediction_type,
                    num_steps=n_ddim,
                    device=accelerator.device,
                    puzzle_ids=puzzle_ids,
                    solutions=solutions,
                    painter_size=painter_size,
                    given_masks=given_masks,
                )
                acc = evaluate_grids(sr["generated"], solutions, self.eval_clf, cell_size, given_masks=given_masks)
                all_cell_acc.append(acc["cell_acc"])
                all_puzzle_acc.append(acc["puzzle_acc"])

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

                # Painter deviation from thinker (all CPU: preds are .cpu(), gm must match)
                painter_preds = acc["preds"]
                _gm = given_masks[:B_cur].cpu() if given_masks is not None else None
                for tp_raw, dev_lst in [
                    (sr.get("best_thinker_preds"), all_painter_dev_best),
                    (sr.get("mean_thinker_preds"), all_painter_dev_mean),
                ]:
                    if tp_raw is not None:
                        tp = tp_raw - token_offset
                        N = tp.shape[1]
                        diff = painter_preds[:, :N] != tp
                        if _gm is not None:
                            blank = ~_gm[:, :N]
                            dev = diff[blank].float().mean().item() if blank.any() else diff.float().mean().item()
                        else:
                            dev = diff.float().mean().item()
                        dev_lst.append(dev)

                # Image panels for wandb
                if _wandb is not None and len(panels) < n_log:
                    n_new = min(n_log - len(panels), B_cur)
                    conds_vis = batch.get("conditions", batch.get("puzzle_tokens", None))
                    if conds_vis is not None and conds_vis.dim() == 4:
                        conds_vis = conds_vis.cpu()
                    tp_all = sr.get("best_thinker_preds")
                    tt_all = sr.get("best_thinker_ts")
                    sols_np = solutions.cpu().numpy()
                    for i in range(n_new):
                        tp = (tp_all[i] - token_offset).numpy() if tp_all is not None else None
                        tt = tt_all[i] if tt_all is not None else None
                        cond_img = conds_vis[i] if (conds_vis is not None and conds_vis.dim() == 4) else None
                        panel = make_panel_image(
                            cond_img, sr["generated"][i], sols_np[i], thinker_preds=tp, thinker_t=tt
                        )
                        panels.append(_wandb.Image(panel, caption=f"sample[{n_done + i}]"))

                n_done += B_cur

            result["cell_acc"] = float(np.mean(all_cell_acc))
            result["puzzle_acc"] = float(np.mean(all_puzzle_acc))
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
            if ts_cell_accs:
                result["thinker_vs_timestep"] = _wandb.Image(plot_thinker_ts_curve(ts_cell_accs, ts_puzzle_accs))

            # Real-solution eval
            if self.has_realsolution_eval:
                all_real_cell, all_real_puzzle = [], []
                n_real = 0
                for batch in tqdm(dataloader, "Realsolution eval", total=n_batches):
                    if n_real >= n_total:
                        break
                    solutions = batch["solution"]
                    given_masks = batch.get("given_mask")
                    puzzle_ids = batch.get("puzzle_id")
                    if puzzle_ids is not None:
                        puzzle_ids = puzzle_ids.to(accelerator.device)
                    sr_r = sample_grids(
                        self,
                        self._get_teacher_condition(batch, accelerator.device),
                        num_train_timesteps=self.scheduler.config.num_train_timesteps,
                        beta_schedule=self.scheduler.config.beta_schedule,
                        prediction_type=self.scheduler.config.prediction_type,
                        num_steps=n_ddim,
                        device=accelerator.device,
                        puzzle_ids=puzzle_ids,
                        solutions=solutions,
                        painter_size=painter_size,
                        given_masks=given_masks,
                    )
                    acc_r = evaluate_grids(
                        sr_r["generated"], solutions, self.eval_clf, cell_size, given_masks=given_masks
                    )
                    all_real_cell.append(acc_r["cell_acc"])
                    all_real_puzzle.append(acc_r["puzzle_acc"])
                    n_real += solutions.shape[0]
                result["real_cell_acc"] = float(np.mean(all_real_cell))
                result["real_puzzle_acc"] = float(np.mean(all_real_puzzle))

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
        """(B, C, H, W) → float embeddings (B, 81, hidden_size)"""
        feat = self.image_encoder(x)  # (B, enc_channels, grid, grid)
        proj = self.enc_proj(feat)  # (B, hidden_size, grid, grid)
        return proj.flatten(2).transpose(1, 2)  # (B, 81, hidden_size)

    def _get_enc_emb(
        self,
        condition: torch.Tensor,
        noisy: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """V0: encode condition only, ignore noisy and timesteps.
        V1 overrides to use cat(condition, noisy) and optionally the timestep."""
        return self._encode_image(condition)

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
      - _get_enc_emb concatenates condition + noisy before encoding
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
        )
        # Replace 1-channel encoder with 2-channel (condition + noisy)
        self.image_encoder = SpatialEncoder(
            2,
            encoder_cfg.enc_channels,
            factor=model_cfg.cell_size,
            hidden_channels=tuple(encoder_cfg.enc_hidden_channels),
        )

        # Timestep conditioning.  Both projections are zero-init so the model
        # starts as the no-timestep identity and gradually learns to use t.
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

    def _get_enc_emb(
        self,
        condition: torch.Tensor,
        noisy: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Compute shared timestep embedding once (if any conditioning is active).
        temb = None
        if timesteps is not None and (self.enc_timestep_cond or self.thinker_timestep_cond):
            temb = self.timestep_mlp(timesteps)

        # Encode cat(condition, noisy) with optional encoder FiLM.
        feat = self.image_encoder(torch.cat([condition, noisy], dim=1))
        if temb is not None and self.enc_timestep_cond:
            scale, shift = self.enc_film(temb).chunk(2, dim=1)  # (B, enc_channels) each
            feat = feat * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        proj = self.enc_proj(feat)
        enc_emb = proj.flatten(2).transpose(1, 2)  # (B, 81, hidden_size)

        # T2: broadcast timestep embedding into thinker token space.
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

    def _logits_to_spatial(self, logits: torch.Tensor) -> torch.Tensor:
        if self.logit_projection is not None:
            logits = self.logit_projection(logits)  # (B, 81, adapter_in_channels)
        return super()._logits_to_spatial(logits)

    def _run_painter(self, noisy, spatial_cond, timesteps):
        if self.bridge_input_conv is not None:
            # Apply new trainable first conv, then the frozen second conv.
            spatial_cond = F.interpolate(
                spatial_cond, size=self.bridge.painter_size, mode="bilinear", align_corners=False
            )
            spatial_cond = torch.nn.functional.silu(self.bridge_input_conv(spatial_cond))
            bridge_feat = self.bridge.conv[2](spatial_cond)
            return self.painter(torch.cat([noisy, bridge_feat], dim=1), timesteps).sample
        return super()._run_painter(noisy, spatial_cond, timesteps)

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

        # Pre-compute corrupted conditions for verification
        corrupted_conds = []
        for i, mb in enumerate(micro_batches):
            given_mask = mb.get("given_mask")
            real_cond = mb_data[i]["condition"]
            if given_mask is not None:
                corrupted = self._corrupt_condition(real_cond, mb_data[i]["solution"], given_mask.to(device))
            else:
                corrupted = real_cond
            corrupted_conds.append(corrupted)

        total_diff_loss = 0.0
        total_sudoku_loss = 0.0
        total_verif_loss = 0.0
        lr = 0.0

        for sup_idx in range(self.n_sup):
            for i, d in enumerate(mb_data):
                noise_pred, logits, d["z_H"], d["z_L"] = self.reasoning_step(
                    d["condition"], d["noisy"], d["z_H"], d["z_L"], d["timesteps"], d["puzzle_ids"]
                )
                step_loss, diff_loss, sudoku_loss = self._compute_step_loss(noise_pred, logits, d, device)
                total_diff_loss += diff_loss.item()
                total_sudoku_loss += sudoku_loss.item()

                # Verification loss only at the last supervision step
                if sup_idx == self.n_sup - 1:
                    seq_len = self.thinker.inner.config.seq_len
                    z_H_pos = d["z_H"][:, :seq_len, :].float().mean(dim=1)
                    z_H_neg = self._thinker_forward(corrupted_conds[i], d["noisy"], d["timesteps"])
                    B = z_H_pos.shape[0]
                    verif_logits = torch.cat([self.verif_head(z_H_pos), self.verif_head(z_H_neg)]).squeeze(-1)
                    verif_labels = torch.cat([torch.ones(B), torch.zeros(B)]).to(device)
                    verif_loss = F.binary_cross_entropy_with_logits(verif_logits, verif_labels)
                    step_loss = step_loss + self.verif_weight * verif_loss
                    total_verif_loss += verif_loss.item()

                accelerator.backward(step_loss / (global_batch_size * K))

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
                "verif_loss": total_verif_loss / K,
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

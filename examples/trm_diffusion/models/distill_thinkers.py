# ABOUTME: Thinker trained by distillation from a frozen teacher thinker that reads the solved image, through the
# ABOUTME: teacher's frozen translator: per-cell MSE on the translator's bridge output, with the ACT persistent carry.
import copy
import dataclasses

import torch
import torch.nn.functional as F
from datasets.data_sample import DataSample

from models.optim_utils import all_reduce_grads, apply_lr_and_step
from models.painter_thinkers import ThinkerFrozenPainterACT, _masked_merge_sample
from models.utility_models import strip_compiled_prefix


def _load_prefixed(module: torch.nn.Module, sd: dict, prefix: str) -> None:
    """Load the entries of sd under prefix into module; every parameter and persistent buffer must be present."""
    sub = {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}
    own = module.state_dict()
    sub = {k: v for k, v in sub.items() if not (k in own and own[k].shape != v.shape)}
    missing = [k for k in module.load_state_dict(sub, strict=False).missing_keys if "halt_head" not in k]
    if missing:
        raise ValueError(f"teacher checkpoint lacks {prefix}{missing[:5]}")


class ThinkerFrozenPainterDistillACT(ThinkerFrozenPainterACT):
    """ACT thinker trained only to reproduce a frozen teacher's steering code.

    The teacher is a thinker of the same architecture trained with the solved image as its condition image
    (`_seex0`). At every reasoning step the student's logits go through the teacher's translator bridge (BatchNorm
    on the teacher's stored statistics: what the translator sees) and are pulled to the teacher's bridged logits for
    the same x_t and timestep by a per-cell MSE. The painter is not run in training; the student steers the painter
    through the teacher's translator, frozen, in eval.

    The teacher's output depends only on the sample (solved image, x_t, t), which ACT keeps fixed for a whole
    episode, so it is computed once when a sample is loaded (n_sup full teacher reasoning steps from its initial
    state, as its eval forward) and cached in the slot. The halt head is labeled from the per-sample distillation
    loss instead of the painter's loss.

    teacher_checkpoint: a thinker-stage checkpoint of the teacher run; its EMA weights are used for the teacher's
        condition encoder and thinker, and for the translator (weights and bridge statistics) the student uses.
    """

    def __init__(self, *args, teacher_checkpoint: str, **kwargs):
        super().__init__(*args, **kwargs)
        ck = torch.load(teacher_checkpoint, map_location="cpu", weights_only=False)
        sd = {**strip_compiled_prefix(ck["model_state"]), **strip_compiled_prefix(ck["ema_state"] or {})}
        self.teacher_encoder = copy.deepcopy(self.condition_encoder)
        self.teacher_thinker = copy.deepcopy(self.thinker)
        _load_prefixed(self.teacher_encoder, sd, "condition_encoder.")
        _load_prefixed(self.teacher_thinker, sd, "thinker.")
        _load_prefixed(self.thinker_painter_translator, sd, "thinker_painter_translator.")
        for m in (self.teacher_encoder, self.teacher_thinker, self.thinker_painter_translator):
            m.requires_grad_(False)
        if getattr(self.thinker_painter_translator, "_bridge_mode", None) != "normalized":
            raise ValueError("distillation targets the translator's normalizing bridge (bridge_mode=normalized)")

    def train(self, mode: bool = True):
        """Teacher and translator stay in eval mode: the bridge always uses the teacher's stored statistics."""
        super().train(mode)
        for m in (self.teacher_encoder, self.teacher_thinker, self.thinker_painter_translator):
            m.eval()
        return self

    def get_thinker_params(self) -> list:
        return list(self.thinker.parameters())

    def get_encoder_params(self) -> list:
        """The student's condition encoder only (the teacher and the translator are frozen)."""
        return [p for p in self.condition_encoder.parameters() if p.requires_grad]

    def bridge(self, logits: torch.Tensor) -> torch.Tensor:
        """(B, N, C) logits -> (B, N, C) translator bridge output on the teacher's stored statistics."""
        bn = self.thinker_painter_translator._norm
        out = F.batch_norm(
            logits.float().transpose(1, 2),
            bn.running_mean,
            bn.running_var,
            bn.weight,
            bn.bias,
            training=False,
            eps=bn.eps,
        )
        return out.transpose(1, 2)

    def teacher_sample(self, sample: DataSample) -> DataSample:
        """The sample as the teacher saw it in training: the solved image in [0, 1] as the condition image."""
        return dataclasses.replace(sample, spatial_conditions=self.painter.decode_for_eval(sample.images))

    @torch.no_grad()
    def teacher_target(self, sample: DataSample) -> torch.Tensor:
        """(B, N, C) the teacher's bridged logits after n_sup reasoning steps from its initial state."""
        ts = self.teacher_sample(sample)
        enc = self.teacher_encoder(ts.spatial_conditions, x_noisy=ts.x_noisy, timesteps=ts.timesteps).enc_emb
        z_H, z_L = self.teacher_thinker.get_initial_states(enc.shape[0])
        z_H, z_L = z_H.to(enc.device), z_L.to(enc.device)
        for _ in range(self.n_sup):
            logits, z_H, z_L = self.teacher_thinker.reasoning_step(enc, z_H, z_L, ts.puzzle_id, timesteps=ts.timesteps)
        return self.bridge(logits)

    def distill_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """(B,) per-sample MSE between the student's bridged logits and the target, over cells and channels."""
        assert logits.shape == target.shape, (logits.shape, target.shape)
        return (self.bridge(logits) - target).pow(2).flatten(1).mean(1)

    def train_step(self, micro_batches, accelerator, optimizers, ema, global_step, **kwargs):
        """ThinkerFrozenPainterACT.train_step with the distillation loss in place of the painter's loss."""
        device = accelerator.device
        K = len(micro_batches)
        n_sup = self.n_sup

        if self._act_carry is None:
            self._act_carry = []
            for mb in micro_batches:
                sample = self.painter._prepare_training_sample(mb, device)
                bsz = sample.x_noisy.shape[0]
                z_H, z_L = self.get_initial_states(bsz)
                z_H, z_L = z_H.to(device), z_L.to(device)
                z_H0_init = z_H[:, 0].detach().clone()
                self._act_carry.append(
                    {
                        "sample": sample,
                        "target": self.teacher_target(sample),
                        "z_H": z_H,
                        "z_L": z_L,
                        "steps": torch.zeros(bsz, dtype=torch.long, device=device),
                        "halted": None,
                        "pending_halt": torch.zeros(bsz, dtype=torch.bool, device=device),
                        "z_prev": z_H0_init.clone(),
                        "z_init": z_H0_init.clone(),
                        "hist_z": [[] for _ in range(bsz)],
                        "hist_loss": [[] for _ in range(bsz)],
                    }
                )

        total = 0.0
        completed_histories: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for i, slot in enumerate(self._act_carry):
            prev_halted = slot["halted"]
            if prev_halted is not None and prev_halted.any():
                fresh = self.painter._prepare_training_sample(micro_batches[i], device)
                bsz = fresh.x_noisy.shape[0]
                fresh_z_H, fresh_z_L = self.get_initial_states(bsz)
                fresh_z_H, fresh_z_L = fresh_z_H.to(device), fresh_z_L.to(device)
                slot["sample"] = _masked_merge_sample(slot["sample"], fresh, prev_halted)
                rows = prev_halted.nonzero(as_tuple=True)[0]
                target = slot["target"].clone()
                target[rows] = self.teacher_target(_select_rows(slot["sample"], rows))
                slot["target"] = target
                m = prev_halted.view(-1, 1, 1)
                slot["z_H"] = torch.where(m, fresh_z_H, slot["z_H"])
                slot["z_L"] = torch.where(m, fresh_z_L, slot["z_L"])
                slot["steps"] = torch.where(prev_halted, torch.zeros_like(slot["steps"]), slot["steps"])
                slot["pending_halt"] = slot["pending_halt"] & ~prev_halted
                m2 = prev_halted.view(-1, 1)
                fresh_z_H0 = fresh_z_H[:, 0].detach()
                slot["z_prev"] = torch.where(m2, fresh_z_H0, slot["z_prev"])
                slot["z_init"] = torch.where(m2, fresh_z_H0, slot["z_init"])
                for j in rows.tolist():
                    slot["hist_z"][j] = []
                    slot["hist_loss"][j] = []

            sample = slot["sample"]
            logits, slot["z_H"], slot["z_L"] = self.thinker.reasoning_step(
                self._encode_condition(sample), slot["z_H"], slot["z_L"], sample.puzzle_id, timesteps=sample.timesteps
            )
            per_sample = self.distill_loss(logits, slot["target"])
            accelerator.backward(per_sample.mean() / K)
            total += per_sample.mean().item()
            slot["steps"] = slot["steps"] + 1

            z_H0 = slot["z_H"][:, 0].detach()
            loss_detached = per_sample.detach()
            for j in range(z_H0.shape[0]):
                slot["hist_z"][j].append(z_H0[j])
                slot["hist_loss"][j].append(loss_detached[j])

            pred = self.thinker.predict_halt_value(slot["z_H"], slot["z_prev"])
            slot["z_prev"] = z_H0
            is_last_step = slot["steps"] >= n_sup
            head_wants_halt = pred <= self.halt_threshold
            halted = is_last_step | (slot["pending_halt"] if self.halt_lookahead else head_wants_halt)
            if self.halt_exploration_prob > 0 and n_sup > 1:
                b = halted.shape[0]
                explore = torch.rand(b, device=device) < self.halt_exploration_prob
                min_steps = torch.where(
                    explore,
                    torch.randint(2, n_sup + 1, (b,), device=device),
                    torch.zeros(b, dtype=torch.long, device=device),
                )
                halted = halted & (slot["steps"] >= min_steps)
            slot["pending_halt"] = head_wants_halt & ~halted & self.halt_lookahead
            slot["halted"] = halted
            for j in halted.nonzero(as_tuple=True)[0].tolist():
                completed_histories.append(
                    (slot["z_init"][j], torch.stack(slot["hist_z"][j]), torch.stack(slot["hist_loss"][j]))
                )

        enc_params = self.get_encoder_params()
        all_reduce_grads(self.get_thinker_params() + enc_params)
        accelerator.clip_grad_norm_(self.get_thinker_params(), self.train_cfg.grad_clip)
        if enc_params:
            accelerator.clip_grad_norm_(enc_params, self.train_cfg.grad_clip)
        lr = apply_lr_and_step(optimizers, global_step)
        global_step += 1
        if ema is not None:
            ema.update(self)

        losses = {"distill_loss": total / K}
        halt_loss = self.thinker.train_halt_head_ragged(completed_histories, label_margin=self.halt_label_margin)
        if completed_histories:
            losses["halt_head_loss"] = halt_loss
            losses["act_steps_completed"] = float(len(completed_histories))
        return losses, lr, global_step


def _select_rows(sample: DataSample, rows: torch.Tensor) -> DataSample:
    """The DataSample restricted to the given batch rows (every populated tensor field)."""
    updates = {
        f.name: getattr(sample, f.name)[rows]
        for f in dataclasses.fields(DataSample)
        if isinstance(getattr(sample, f.name), torch.Tensor)
    }
    return dataclasses.replace(sample, **updates)

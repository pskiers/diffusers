# ABOUTME: One-stage teacher/student thinker: each training sample runs through one shared thinker twice, as a teacher
# ABOUTME: that reads the clean target (painter's loss) and a student that reads the usual input (matches the teacher).
import torch
import torch.nn.functional as F
from datasets.data_sample import DataSample

from models.optim_utils import all_reduce_grads, apply_lr_and_step
from models.painter_thinkers import ThinkerFrozenPainterACT, _masked_merge_sample


class ThinkerFrozenPainterPairedACT(ThinkerFrozenPainterACT):
    """ACT thinker trained in one stage as its own teacher.

    Every training sample (x_t, t) is read twice by the same encoder and thinker: the teacher view reads the clean
    target as train_cfg.thinker_sees_x0 says, the student view reads the usual input. The teacher's logits steer the
    painter and get the painter's loss (training thinker, encoder and translator); the student's logits are pulled
    to the teacher's by match_weight * the per-cell MSE between their bridge outputs, with the teacher's side
    detached. The bridge here is the translator's BatchNorm on its running statistics with its affine detached, so
    the matching loss cannot shrink the bridge itself. Teacher and student keep their own ACT states for the same
    sample and halt together when the student's halt head says so; halt labels come from the student's matching
    loss. Eval and sampling read the usual input (the student).

    The training-only steering options of reasoning_step (cfg_prob, steer_grad_clip, steer_grad_t_min) are not
    supported here.
    """

    def __init__(self, *args, match_weight: float, **kwargs):
        super().__init__(*args, **kwargs)
        tc = self.train_cfg
        if tc.thinker_sees_x0 is None or tc.thinker_x0_prob != 1.0:
            raise ValueError("thinker_act_paired needs train.thinker_sees_x0 (the teacher's view), thinker_x0_prob 1")
        if tc.cfg_prob > 0 or tc.steer_grad_clip is not None or tc.steer_grad_t_min is not None:
            raise NotImplementedError("thinker_act_paired does not support cfg_prob / steer_grad_clip / steer_grad_t_min")
        if getattr(self.thinker_painter_translator, "_bridge_mode", None) != "normalized":
            raise ValueError("the matching loss uses the translator's normalizing bridge (bridge_mode=normalized)")
        self.match_weight = match_weight

    def _encode_condition(self, sample: DataSample) -> torch.Tensor:
        """The student's view: eval, sampling and the student pass in training."""
        return self._encode_with(self.condition_encoder, sample, None)

    def bridge(self, logits: torch.Tensor) -> torch.Tensor:
        """(B, N, C) logits -> (B, N, C) the translator's bridge on its running statistics, no gradient to it."""
        bn = self.thinker_painter_translator._norm
        out = F.batch_norm(
            logits.float().transpose(1, 2),
            bn.running_mean,
            bn.running_var,
            bn.weight.detach(),
            bn.bias.detach(),
            training=False,
            eps=bn.eps,
        )
        return out.transpose(1, 2)

    def pair_step(self, sample: DataSample, z_T: tuple, z_S: tuple):
        """One reasoning step of both views on the same sample.

        Returns (painter loss, its loss dict, (B,) matching loss, (B,) painter loss per sample, new teacher
        (z_H, z_L), new student (z_H, z_L)).
        """
        b = sample.x_noisy.shape[0]
        enc = torch.cat(
            [self._encode_with(self.condition_encoder, sample, self.train_cfg.thinker_sees_x0), self._encode_condition(sample)]
        )
        z_H, z_L = torch.cat([z_T[0], z_S[0]]), torch.cat([z_T[1], z_S[1]])
        pid = None if sample.puzzle_id is None else torch.cat([sample.puzzle_id, sample.puzzle_id])
        logits, z_H, z_L = self.thinker.reasoning_step(enc, z_H, z_L, pid, timesteps=torch.cat([sample.timesteps] * 2))
        logits_T, logits_S = logits[:b], logits[b:]
        scale_fn = getattr(self.loss_fn, "scale_logits_for_painter", None)
        noise_pred = self.run_painter(sample, scale_fn(logits_T) if scale_fn is not None else logits_T)
        painter_loss, loss_dict = self.loss_fn(noise_pred, logits_T, sample)
        per_sample_painter = (noise_pred.float() - sample.target.float()).pow(2).flatten(1).mean(1).detach()
        match = (self.bridge(logits_S) - self.bridge(logits_T).detach()).pow(2).flatten(1).mean(1)
        return painter_loss, loss_dict, match, per_sample_painter, (z_H[:b], z_L[:b]), (z_H[b:], z_L[b:])

    def train_step(self, micro_batches, accelerator, optimizers, ema, global_step, **kwargs):
        """ThinkerFrozenPainterACT.train_step on paired teacher / student carries (see the class docstring)."""
        device = accelerator.device
        K = len(micro_batches)
        n_sup = self.n_sup

        if self._act_carry is None:
            self._act_carry = []
            for mb in micro_batches:
                sample = self._prepare_sample(mb, device)
                bsz = sample.x_noisy.shape[0]
                z_H, z_L = (z.to(device) for z in self.get_initial_states(bsz))
                z_H0_init = z_H[:, 0].detach().clone()
                self._act_carry.append(
                    {
                        "sample": sample,
                        "z_T": (z_H, z_L),
                        "z_S": (z_H.clone(), z_L.clone()),
                        "steps": torch.zeros(bsz, dtype=torch.long, device=device),
                        "halted": None,
                        "pending_halt": torch.zeros(bsz, dtype=torch.bool, device=device),
                        "z_prev": z_H0_init.clone(),
                        "z_init": z_H0_init.clone(),
                        "hist_z": [[] for _ in range(bsz)],
                        "hist_loss": [[] for _ in range(bsz)],
                    }
                )

        totals: dict[str, float] = {}
        completed_histories: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for i, slot in enumerate(self._act_carry):
            prev_halted = slot["halted"]
            if prev_halted is not None and prev_halted.any():
                fresh = self._prepare_sample(micro_batches[i], device)
                fresh_z_H, fresh_z_L = (z.to(device) for z in self.get_initial_states(fresh.x_noisy.shape[0]))
                slot["sample"] = _masked_merge_sample(slot["sample"], fresh, prev_halted)
                m = prev_halted.view(-1, 1, 1)
                for view in ("z_T", "z_S"):
                    slot[view] = (torch.where(m, fresh_z_H, slot[view][0]), torch.where(m, fresh_z_L, slot[view][1]))
                slot["steps"] = torch.where(prev_halted, torch.zeros_like(slot["steps"]), slot["steps"])
                slot["pending_halt"] = slot["pending_halt"] & ~prev_halted
                m2 = prev_halted.view(-1, 1)
                fresh_z_H0 = fresh_z_H[:, 0].detach()
                slot["z_prev"] = torch.where(m2, fresh_z_H0, slot["z_prev"])
                slot["z_init"] = torch.where(m2, fresh_z_H0, slot["z_init"])
                for j in prev_halted.nonzero(as_tuple=True)[0].tolist():
                    slot["hist_z"][j] = []
                    slot["hist_loss"][j] = []

            painter_loss, loss_dict, match, per_sample_painter, slot["z_T"], slot["z_S"] = self.pair_step(
                slot["sample"], slot["z_T"], slot["z_S"]
            )
            accelerator.backward((painter_loss + self.match_weight * match.mean()) / K)
            for k, v in {**loss_dict, "match_loss": match.mean().item(),
                         "teacher_painter_loss": per_sample_painter.mean().item()}.items():
                totals[k] = totals.get(k, 0.0) + v
            slot["steps"] = slot["steps"] + 1

            z_H0 = slot["z_S"][0][:, 0].detach()
            match_detached = match.detach()
            for j in range(z_H0.shape[0]):
                slot["hist_z"][j].append(z_H0[j])
                slot["hist_loss"][j].append(match_detached[j])

            pred = self.thinker.predict_halt_value(slot["z_S"][0], slot["z_prev"])
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
        all_reduce_grads(self.get_thinker_params() + enc_params + self.get_painter_params())
        accelerator.clip_grad_norm_(self.get_thinker_params(), self.train_cfg.grad_clip)
        if enc_params:
            accelerator.clip_grad_norm_(enc_params, self.train_cfg.grad_clip)
        lr = apply_lr_and_step(optimizers, global_step)
        global_step += 1
        if ema is not None:
            ema.update(self)

        losses = {k: v / K for k, v in totals.items()}
        halt_loss = self.thinker.train_halt_head_ragged(completed_histories, label_margin=self.halt_label_margin)
        if completed_histories:
            losses["halt_head_loss"] = halt_loss
            losses["act_steps_completed"] = float(len(completed_histories))
        return losses, lr, global_step

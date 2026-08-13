import dataclasses
from typing import Optional

import torch
import torch.nn as nn
from hydra.utils import instantiate
from tqdm.auto import tqdm

from models.eval_callbacks import EvalCallbackBase
from models.optim_utils import ScheduledOptimizer, apply_lr_and_step
from configs.schemas import (
    TrainConfig,
    EvalConfig,
)
from models.base import BaseModel
from models.losses import LossBase, build_loss
from models.translators import ThinkerPainterTranslatorBase
from models.condition_encoders import ConditionEncoderBase
from models.interfaces import TRMOutput, DiffusionPrediction
from datasets.data_sample import DataSample


class ThinkerFrozenPainterBase(BaseModel):
    """
    Base class for thinker models with a frozen pre-trained painter.

    Covers the ControlNet, CrossAttn, and IPAdapter paradigms where a
    thinker reasons over a condition image and its output is translated
    into conditioning signals for a frozen painter (UNet or DiT).

    All submodules (thinker, painter, condition_encoder, loss,
    thinker_painter_translator, eval_callbacks) are instantiated from
    Hydra configs passed to __init__.  The painter is expected to own its
    scheduler, optim_cfg, painter_dtype, condition_keys, and
    _prepare_training_sample.

    Constructor args
    ----------------
    thinker                   : Hydra config → SpatialTRM (owns optim_cfg)
    painter                   : Hydra config → PainterBase (e.g. CrossAttnSteeredDiTPainter)
    train_cfg                 : TrainConfig
    eval_cfg                  : EvalConfig
    condition_encoder         : Hydra config → ConditionEncoderBase subclass
    loss                      : Hydra config → LossBase subclass
    thinker_painter_translator: Hydra config → ThinkerPainterTranslatorBase subclass
    eval_callbacks            : list of Hydra configs → EvalCallbackBase instances
    scheduler                 : DDPMScheduler injected from factory

    train_cfg.force_unconditional_painter: if True, the frozen painter always receives
        its own null_condition_sample() instead of the real sample (see run_painter) —
        the translator/steering still sees the real one. Forces the steering pathway
        to supply all conditioning itself, rather than nudging an already-conditional
        painter. Default False is a no-op, fully backward compatible.
    """

    token_input: bool = False
    has_realsolution_eval: bool = True
    token_offset: int = 0  # image-conditioned models use 0-8 labels, no shift

    def __init__(
        self,
        thinker,
        painter,
        train_cfg: TrainConfig,
        eval_cfg: EvalConfig,
        condition_encoder,
        loss,
        thinker_painter_translator,
        eval_callbacks=None,
        scheduler=None,
        sampling_pipeline=None,
    ):
        super().__init__()

        # ── Store configs ─────────────────────────────────────────────────────
        self.train_cfg = train_cfg
        self.eval_cfg = eval_cfg
        self.sampling_pipeline = instantiate(sampling_pipeline) if sampling_pipeline is not None else None

        # ── Thinker (instantiated from Hydra config; owns optim_cfg) ─────────
        self.thinker = instantiate(thinker)

        # ── Painter (instantiated from Hydra config; owns scheduler/dtype) ───
        # scheduler / train_cfg / eval_cfg are injected so the painter constructor
        # (DiTPainter / UNetPainter) gets everything it needs.
        self.painter = instantiate(
            painter,
            scheduler=scheduler,
            train_cfg=train_cfg,
            eval_cfg=eval_cfg,
        )

        # ── Condition encoder, loss, translator, eval callbacks ───────────────
        self.condition_encoder: ConditionEncoderBase = instantiate(condition_encoder)
        self.condition_encoder.bind_painter(self.painter)
        self.loss_fn: LossBase = build_loss(train_cfg, self.painter.scheduler)
        self.thinker_painter_translator: ThinkerPainterTranslatorBase = instantiate(thinker_painter_translator)
        self.eval_callbacks: list[EvalCallbackBase] = (
            [instantiate(cb) for cb in eval_callbacks] if eval_callbacks else []
        )

    # ── Delegated properties ──────────────────────────────────────────────────

    @property
    def scheduler(self):
        return self.painter.scheduler

    @property
    def noise_shape(self) -> tuple:
        return self.painter.noise_shape

    def decode_for_eval(self, latents: torch.Tensor) -> torch.Tensor:
        return self.painter.decode_for_eval(latents)

    def images_to_log(self, images: torch.Tensor) -> torch.Tensor:
        return self.painter.images_to_log(images)

    @property
    def n_sup(self) -> int:
        return self.thinker.n_sup

    # ── Condition helpers ─────────────────────────────────────────────────────

    def _encode_condition(self, sample: DataSample) -> torch.Tensor:
        """Encode condition fields from a DataSample into thinker token embeddings.

        timesteps is always forwarded, not just when some encoder's
        condition_keys happens to list it: every ConditionEncoderBase
        subclass's forward() accepts a `timesteps` kwarg (defaulting to
        None) for with_timestep_emb / noisy_dropout_p_max / the
        X0PredHintConditionEncoder wrapper's hint mechanism, none of which
        can do anything useful without it.
        """
        enc_keys = self.condition_encoder.condition_keys
        primary = getattr(sample, enc_keys[0])
        extra: dict = {"timesteps": sample.timesteps}
        for k in enc_keys[1:]:
            if k == "x_noisy":
                extra["x_noisy"] = sample.enc_x_noisy if sample.enc_x_noisy is not None else sample.x_noisy
            else:
                extra[k] = getattr(sample, k)
        if getattr(self.condition_encoder, "needs_sample", False):
            extra["sample"] = sample
        return self.condition_encoder(primary, **extra).enc_emb

    # ── Core model methods ────────────────────────────────────────────────────

    def _batch_to_sample(self, batch, device: torch.device) -> DataSample:
        """Build a static-condition DataSample from a raw batch for sampling."""
        _RUNTIME = {"x_noisy", "timesteps", "target", "enc_x_noisy"}
        kwargs: dict = {}
        for f in dataclasses.fields(DataSample):
            if f.name in _RUNTIME:
                continue
            val = batch.get(f.name) if hasattr(batch, "get") else None
            if val is not None:
                kwargs[f.name] = val.to(device) if isinstance(val, torch.Tensor) else val
        return DataSample(**kwargs)

    def get_initial_states(self, bsz: int):
        return self.thinker.get_initial_states(bsz)

    def run_painter(self, sample: DataSample, logits: torch.Tensor) -> torch.Tensor:
        """Translate thinker logits to conditioning, then run the frozen painter.

        translator_extra is always built from the real sample — the translator
        needs real information to produce useful steering. Only the painter's
        own input is optionally blinded (force_unconditional_painter), via its
        own null_condition_sample() so the frozen painter's condition_encoder
        keeps running with the same shapes, just on zeroed data.
        """
        painter_dtype = getattr(self.painter, "painter_dtype", None)
        ctx = (
            torch.autocast(device_type=sample.x_noisy.device.type, dtype=painter_dtype)
            if painter_dtype is not None
            else torch.autocast(device_type=sample.x_noisy.device.type, enabled=False)
        )
        painter_sample = (
            self.painter.null_condition_sample(sample) if self.train_cfg.force_unconditional_painter else sample
        )
        with ctx:
            translator_extra = {k: getattr(sample, k) for k in self.thinker_painter_translator.condition_keys}
            steering = self.thinker_painter_translator(TRMOutput(logits=logits), **translator_extra)
            return self.painter(painter_sample, steering).pred

    def reasoning_step(self, sample: DataSample, z_H: torch.Tensor, z_L: torch.Tensor):
        """One supervision step: encode -> think -> translate -> paint.

        Returns: (noise_pred, logits, z_H_next, z_L_next)
        """
        enc_emb = self._encode_condition(sample)
        logits, z_H_next, z_L_next = self.thinker.reasoning_step(
            enc_emb, z_H, z_L, sample.puzzle_id, timesteps=sample.timesteps
        )

        scale_fn = getattr(self.loss_fn, "scale_logits_for_painter", None)
        logits_for_tpt = scale_fn(logits) if scale_fn is not None else logits

        if self.training and self.train_cfg.cfg_prob > 0:
            drop = torch.rand(logits.shape[0], device=logits.device) < self.train_cfg.cfg_prob
            logits_for_tpt = logits_for_tpt * (~drop[:, None, None]).float()

        noise_pred = self.run_painter(sample, logits_for_tpt)
        return noise_pred, logits, z_H_next, z_L_next

    def forward_with_carry(
        self,
        sample: DataSample,
        z_H: Optional[torch.Tensor] = None,
        z_L: Optional[torch.Tensor] = None,
        n_sup: Optional[int] = None,
        null_steering: bool = False,
        use_halt_head: bool = False,
        halt_threshold: float = 0.0,
        steps_used: Optional[list] = None,
        halt_preds_out: Optional[list] = None,
        halt_steps_out: Optional[list] = None,
    ) -> tuple[DiffusionPrediction, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Like forward(), but exposes the TRM's recurrent carry (z_H, z_L)
        instead of always resetting it, and allows overriding n_sup.

        z_H/z_L=None (the default) reproduces forward()'s behavior exactly:
        a fresh get_initial_states() reset. Passing in a previous call's
        returned carry instead lets a caller thread reasoning state across
        denoising timesteps — out of the training distribution (training
        only ever sees a fresh reset), so this is for inference-time
        ablation only (see experiments/ablate_trm_loop_budget.py), not used
        by forward() itself.

        use_halt_head/halt_threshold: opt-in early exit (requires
        thinker.with_halt_head=True at construction), applied per sample via
        dynamic re-batching rather than one collective batch-mean decision:
        the moment an individual sample's own prediction drops to/below
        halt_threshold, it's removed from the active batch (its final
        (z_H, z_L, logits) are recorded as-is) and every later iteration
        operates on a strictly smaller tensor containing only the
        still-active samples — real, not just nominal, per-sample compute
        savings, since the batch actually shrinks rather than merely
        freezing already-halted samples' outputs in place while still
        paying for them every iteration. Off by default, so existing call
        sites are unaffected.

        steps_used: optional list; if given and use_halt_head=True, the
        *average* number of reasoning steps used per sample this call
        (mean of each sample's own halting step, i.e. real compute cost —
        may be fractional) is appended to it; if use_halt_head=False, the
        (always-n_sup) iteration count is appended instead. See
        experiments/ablate_trm_loop_budget.py's "halt" axis. None by
        default, so existing call sites are unaffected. Not appended to
        under null_steering (no reasoning step is taken).

        halt_preds_out: optional list; if given, predict_halt_value() for
        the currently-active subset — the full per-active-sample tensor,
        which shrinks in step with the active batch once use_halt_head=True
        starts removing samples — is appended to it after every reasoning
        step. Pairs with n_sup=self.n_sup and use_halt_head=False to record
        every individual sample's own prediction across the full,
        untruncated trajectory, with a consistent (B,) shape at every step
        — see experiments/eval_halt_step_distribution.py. None by default,
        so existing call sites are unaffected.

        halt_steps_out: optional list; if given and use_halt_head=True, the
        per-sample (B,) tensor of the reasoning step at which each sample
        individually halted (the actual iteration count performed if it
        never did) is appended to it — the real per-sample halting-step
        distribution, not an estimate. See
        experiments/eval_halt_step_profile.py. None by default, so existing
        call sites are unaffected.

        Returns: (DiffusionPrediction, z_H_next, z_L_next). Under
        null_steering, the carry is passed through unchanged so callers can
        thread state uniformly regardless of null_steering. z_H_next/
        z_L_next/logits are always full (B, ...) tensors regardless of
        use_halt_head — samples removed from the active batch partway
        through have their final state scattered back into their original
        position.
        """
        bsz = sample.x_noisy.shape[0]

        if null_steering:
            logits = torch.zeros(
                bsz, self.thinker.inner.config.seq_len, self.thinker.vocab_size,
                device=sample.x_noisy.device,
            )
        else:
            enc_emb = self._encode_condition(sample)
            puzzle_ids = sample.puzzle_id

            if z_H is None or z_L is None:
                z_H, z_L = self.get_initial_states(bsz)
                z_H, z_L = z_H.to(sample.x_noisy.device), z_L.to(sample.x_noisy.device)

            n_sup = n_sup if n_sup is not None else self.n_sup
            device = sample.x_noisy.device

            if not use_halt_head:
                logits = None
                step_count = 0
                for _ in range(n_sup):
                    logits, z_H, z_L = self.thinker.reasoning_step(
                        enc_emb, z_H, z_L, puzzle_ids, timesteps=sample.timesteps
                    )
                    step_count += 1
                    if halt_preds_out is not None:
                        halt_preds_out.append(self.thinker.predict_halt_value(z_H).detach())
                if steps_used is not None:
                    steps_used.append(step_count)
            else:
                # active_idx: original-batch positions of samples still being
                # reasoned about. cur_* are filtered down to just that subset
                # each time a sample halts; final_* accumulate each sample's
                # last computed state, in original-batch order, as it halts.
                active_idx = torch.arange(bsz, device=device)
                cur_enc_emb, cur_puzzle_ids, cur_timesteps = enc_emb, puzzle_ids, sample.timesteps
                cur_z_H, cur_z_L, cur_logits = z_H, z_L, None

                final_logits = None
                final_z_H = torch.empty_like(z_H)
                final_z_L = torch.empty_like(z_L)
                halt_step = torch.full((bsz,), -1, dtype=torch.long, device=device)

                step_count = 0
                for _ in range(n_sup):
                    new_logits, new_z_H, new_z_L = self.thinker.reasoning_step(
                        cur_enc_emb, cur_z_H, cur_z_L, cur_puzzle_ids, timesteps=cur_timesteps
                    )
                    step_count += 1
                    if final_logits is None:
                        final_logits = torch.empty(
                            bsz, *new_logits.shape[1:], dtype=new_logits.dtype, device=device
                        )

                    pred = self.thinker.predict_halt_value(new_z_H)
                    if halt_preds_out is not None:
                        halt_preds_out.append(pred.detach())
                    halt_now = pred <= halt_threshold

                    halted_orig = active_idx[halt_now]
                    final_logits[halted_orig] = new_logits[halt_now]
                    final_z_H[halted_orig] = new_z_H[halt_now]
                    final_z_L[halted_orig] = new_z_L[halt_now]
                    halt_step[halted_orig] = step_count

                    keep = ~halt_now
                    active_idx = active_idx[keep]
                    if active_idx.numel() == 0:
                        break
                    cur_enc_emb = cur_enc_emb[keep]
                    cur_puzzle_ids = cur_puzzle_ids[keep] if cur_puzzle_ids is not None else None
                    cur_timesteps = cur_timesteps[keep] if cur_timesteps is not None else None
                    cur_z_H, cur_z_L, cur_logits = new_z_H[keep], new_z_L[keep], new_logits[keep]

                if active_idx.numel() > 0:
                    # n_sup exhausted with some samples never individually
                    # crossing the threshold — freeze them at their last
                    # computed (still-active) state.
                    final_logits[active_idx] = cur_logits
                    final_z_H[active_idx] = cur_z_H
                    final_z_L[active_idx] = cur_z_L
                    halt_step[active_idx] = step_count

                logits, z_H, z_L = final_logits, final_z_H, final_z_L
                if steps_used is not None:
                    steps_used.append(float(halt_step.float().mean().item()))
                if halt_steps_out is not None:
                    halt_steps_out.append(halt_step.detach())

        noise_pred = self.run_painter(sample, logits)

        pred = DiffusionPrediction(
            pred=noise_pred,
            pred_type=self.painter.scheduler.config.prediction_type,
            logits=None if null_steering else logits,
        )
        return pred, z_H, z_L

    def forward(
        self,
        sample: DataSample,
        null_steering: bool = False,
        use_halt_head: Optional[bool] = None,
        halt_threshold: Optional[float] = None,
        **kwargs,
    ) -> DiffusionPrediction:
        """Single inference pass: encode → think (n_sup steps) → translate → paint.

        No guidance is applied here.  Use CFGPredictor / NoisyGuidancePredictor
        from models.sampling to add guidance during SamplingPipeline.sample().
        Logits are included in the return value so eval_step can compute CE loss.

        null_steering=True skips condition encoding and thinker reasoning
        entirely and feeds the translator all-zero logits instead. The
        frozen painter still receives the real sample (its own conditioning
        is untouched) — only the thinker's contribution is ablated. This is
        the steering-only CFG null used by SteeringCFGPredictor
        (models/sampling.py), as opposed to null_condition_sample() below,
        which also zeros the painter's own conditioning.

        use_halt_head/halt_threshold default to self.eval_cfg's
        use_halt_head/halt_threshold (settable via +eval.use_halt_head=true
        +eval.halt_threshold=... on the train_trm.py/eval.py command line,
        same pattern as train.force_unconditional_painter) when not passed
        explicitly — so existing call sites in models/sampling.py (which all
        just call model(sample)) transparently pick up the config value
        instead of needing to be modified themselves.
        """
        if use_halt_head is None:
            use_halt_head = self.eval_cfg.use_halt_head
        if halt_threshold is None:
            halt_threshold = self.eval_cfg.halt_threshold
        pred, _, _ = self.forward_with_carry(
            sample, null_steering=null_steering, use_halt_head=use_halt_head, halt_threshold=halt_threshold
        )
        return pred

    def null_condition_sample(self, sample: DataSample) -> DataSample:
        """Return a copy with all condition fields zeroed for the CFG unconditional pass.

        Zeros both thinker condition_encoder keys and painter condition_keys so both
        the thinker encoding and the painter cross-attention receive null input.

        "x_noisy" is special-cased: it's the painter's actual denoising target
        (x_t), not a droppable conditioning field. If a condition encoder reads
        it (e.g. ObjectFeatureEncoderV1's self-conditioning on the noisy latent),
        the null view is given via enc_x_noisy instead — mirroring
        _encode_condition's fallback — so the painter still denoises the real
        x_t while the encoder sees zeros.
        """
        all_keys = set(self.condition_encoder.condition_keys) | set(self.painter.condition_keys)
        needs_null_noisy_view = "x_noisy" in all_keys
        droppable_keys = all_keys - {"x_noisy"}
        updates = {
            k: torch.zeros_like(getattr(sample, k))
            for k in droppable_keys
            if getattr(sample, k, None) is not None
        }
        if needs_null_noisy_view and sample.x_noisy is not None:
            updates["enc_x_noisy"] = torch.zeros_like(sample.x_noisy)
        elif sample.enc_x_noisy is not None:
            updates["enc_x_noisy"] = torch.zeros_like(sample.enc_x_noisy)
        return dataclasses.replace(sample, **updates)

    # ── Parameter groups ──────────────────────────────────────────────────────

    def get_painter_params(self) -> list:
        """Return trainable painter parameters (non-empty for IP-Adapter variants)."""
        return [p for p in self.painter.parameters() if p.requires_grad]

    def get_thinker_params(self) -> list:
        return list(self.thinker.parameters())

    def get_encoder_params(self) -> list:
        """All trainable parameters except the thinker and painter."""
        painter_ids = {id(p) for p in self.painter.parameters()}
        thinker_ids = {id(p) for p in self.thinker.parameters()}
        return [
            p for p in self.parameters()
            if id(p) not in painter_ids and id(p) not in thinker_ids and p.requires_grad
        ]

    # ── Optimizers ────────────────────────────────────────────────────────────

    def build_optimizers(self, world_size, num_steps) -> list[ScheduledOptimizer]:
        thinker_optims = self.thinker.build_optimizers(world_size, num_steps)
        extra_params = self.get_painter_params() + self.get_encoder_params()
        if not extra_params:
            return thinker_optims
        optim_cfg = self.painter.optim_cfg
        enc_optim = torch.optim.AdamW(extra_params, lr=0, weight_decay=optim_cfg.weight_decay)
        enc_scheduled = ScheduledOptimizer(
            enc_optim,
            base_lr=optim_cfg.lr,
            warmup_steps=optim_cfg.warmup_steps,
            num_steps=num_steps,
            min_ratio=optim_cfg.lr_min_ratio,
        )
        return thinker_optims + [enc_scheduled]

    # ── Data preparation ──────────────────────────────────────────────────────

    def prep_mb_data(self, micro_batches, device) -> list[dict]:
        """Prepare all per-microbatch data for a training step.

        Delegates image encoding, noise sampling, noisy_swap, and condition
        field routing to painter._prepare_training_sample.  Also initialises
        the thinker recurrent states so train_step has everything in one place.

        Returns a list of dicts with:
            "sample": DataSample (x_noisy, timesteps, target, all condition fields)
            "z_H":    initial thinker H state on device
            "z_L":    initial thinker L state on device
        """
        result = []
        for mb in micro_batches:
            sample = self.painter._prepare_training_sample(mb, device)
            bsz = sample.x_noisy.shape[0]
            z_H, z_L = self.get_initial_states(bsz)
            result.append({"sample": sample, "z_H": z_H.to(device), "z_L": z_L.to(device)})
        return result

    # ── Training ──────────────────────────────────────────────────────────────

    def train_step(
        self,
        micro_batches,
        accelerator,
        optimizers,
        ema,
        global_batch_size,
        global_step,
        **kwargs,
    ):
        K = len(micro_batches)
        device = accelerator.device
        mb_data = self.prep_mb_data(micro_batches, device)

        total_losses: dict[str, float] = {}
        lr = 0.0

        # Buffers for the adaptive-halting head's auxiliary loss (see
        # SpatialTRM.train_halt_head) — one (z_H[:, 0], per-sample loss) pair
        # per reasoning step, per micro-batch. Only the pooled token is kept
        # (not the full z_H), and everything is detached: this rides on
        # tensors already produced by the main loop, no extra graph/compute.
        use_halt_head = getattr(self.thinker, "with_halt_head", False)
        halt_histories = [{"z_H0": [], "loss": []} for _ in mb_data] if use_halt_head else None

        for _ in range(self.n_sup):
            for i, d in enumerate(mb_data):
                noise_pred, logits, d["z_H"], d["z_L"] = self.reasoning_step(
                    d["sample"], d["z_H"], d["z_L"]
                )
                step_loss, loss_dict = self.loss_fn(noise_pred, logits, d["sample"])
                for k, v in loss_dict.items():
                    total_losses[k] = total_losses.get(k, 0.0) + v
                if step_loss.requires_grad:
                    accelerator.backward(step_loss / (global_batch_size * K))

                if halt_histories is not None:
                    per_sample_loss = (
                        (noise_pred.float() - d["sample"].target.float()).pow(2).flatten(1).mean(1)
                    )
                    halt_histories[i]["z_H0"].append(d["z_H"][:, 0].detach())
                    halt_histories[i]["loss"].append(per_sample_loss.detach())

            accelerator.clip_grad_norm_(self.get_thinker_params(), 1.0)
            enc_params = self.get_encoder_params()
            if enc_params:
                accelerator.clip_grad_norm_(enc_params, 1.0)
            lr = apply_lr_and_step(optimizers, global_step)
            global_step += 1
            if ema is not None:
                ema.update(self)

        n = self.n_sup * K
        losses = {k: v / n for k, v in total_losses.items()}

        if halt_histories is not None:
            halt_loss_sum = sum(
                self.thinker.train_halt_head(h["z_H0"], h["loss"]) for h in halt_histories
            )
            losses["halt_head_loss"] = halt_loss_sum / K

        return losses, lr, global_step

    # ── Evaluation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def eval_step(self, dataloader, accelerator, **kwargs) -> dict:
        max_batches = kwargs.get("max_batches", 100)

        self.train()
        metric_accum: dict[str, float] = {}
        n_batches = 0

        for i, batch in tqdm(enumerate(dataloader), "Eval", total=max_batches):
            if i >= max_batches:
                break
            [d] = self.prep_mb_data([batch], accelerator.device)
            result = self(d["sample"])
            noise_pred, logits = result.pred, result.logits

            _, loss_dict = self.loss_fn(noise_pred, logits, d["sample"])

            if logits is not None and d["sample"].solution is not None:
                B_, N, _ = logits.shape
                if N <= d["sample"].solution.shape[1]:
                    preds = logits.argmax(dim=-1)
                    correct = preds == d["sample"].solution[:B_, :N]
                    loss_dict["thinker_puzzle_acc"] = correct.all(dim=1).float().mean().item()
                    loss_dict["thinker_cell_acc"] = correct.float().mean().item()

            for k, v in loss_dict.items():
                metric_accum[k] = metric_accum.get(k, 0.0) + v
            n_batches += 1

        result = {k: v / n_batches for k, v in metric_accum.items()} if n_batches > 0 else {}

        self.eval()
        for cb in self.eval_callbacks:
            result.update(cb(self, dataloader, accelerator, **kwargs))

        self.train()
        return result

    # ── Compilation ───────────────────────────────────────────────────────────

    def compile_submodules(self):
        self.thinker.inner.L_level = torch.compile(self.thinker.inner.L_level, fullgraph=False)
        if hasattr(self.painter, "compile_submodules"):
            self.painter.compile_submodules()
        if hasattr(self.condition_encoder, "compile_submodules"):
            self.condition_encoder.compile_submodules()
        else:
            self.condition_encoder = torch.compile(self.condition_encoder, fullgraph=False)
        if hasattr(self.thinker_painter_translator, "compile_submodules"):
            self.thinker_painter_translator.compile_submodules()
        else:
            self.thinker_painter_translator = torch.compile(self.thinker_painter_translator, fullgraph=False)


# ── Persistent-carry ACT training (separate from ThinkerFrozenPainterBase —
#    see its own docstring for why) ───────────────────────────────────────────


def _masked_merge_sample(old: DataSample, new: DataSample, mask: torch.Tensor) -> DataSample:
    """Per-row select: mask[j]=True takes new's row j, False keeps old's.

    Assumes old and new agree on which fields are populated (true for any
    two DataSamples built by the same painter._prepare_training_sample
    pathway on the same dataset — collate_data_samples already enforces
    that within one batch, and both call sites here go through the same
    painter/dataset every time)."""
    updates: dict = {}
    for f in dataclasses.fields(DataSample):
        old_v = getattr(old, f.name)
        if old_v is None:
            continue
        new_v = getattr(new, f.name)
        m = mask.view((-1,) + (1,) * (old_v.ndim - 1))
        updates[f.name] = torch.where(m, new_v, old_v)
    return dataclasses.replace(old, **updates)


class ThinkerFrozenPainterACT(ThinkerFrozenPainterBase):
    """ThinkerFrozenPainterBase variant that trains with real ACT-style
    persistent-carry batching, mirroring the actual TRM/HRM training loop
    (see models.trm_wrappers.SpatialTRM.train_halt_head's docstring lineage
    and this class's own commit history for the source research) instead of
    the base class's fixed-n_sup deep-supervision loop. The base class is
    untouched — this only overrides __init__ and train_step; forward,
    forward_with_carry, eval_step, build_optimizers etc. are all inherited
    as-is, since none of them need to change.

    Requires thinker.with_halt_head=True.

    Mechanics: each call to train_step advances exactly ONE reasoning step
    for a persistent batch of "slots" (one per grad-accum micro-batch),
    instead of n_sup steps for a disposable one. Per-sample (not per-slot):
    any sample that halted on the PREVIOUS call gets a fresh example loaded
    from THIS call's micro_batches and its (z_H, z_L, steps) reset before
    this call's step runs; a sample that hasn't halted keeps its in-
    progress state untouched, and this call's freshly-fetched replacement
    candidate at that row is simply discarded — this matches the real
    training loops exactly (they fetch one full fresh batch every
    iteration regardless of how many samples actually need it, and
    torch.where-merge on the halted mask, rather than fetching a variable
    number of replacements). Consequently train_trm.py needs no changes:
    it already fetches grad_accum_steps fresh micro_batches before every
    train_step call.

    global_step advances by 1 per call here, not by n_sup like the base
    class — train.num_steps means something different under this class
    (roughly num_steps_base_class * n_sup for an equivalent total compute
    budget, though not exactly, since compute is reallocated rather than
    uniform).

    halt_exploration_prob: with this probability (per sample, per step),
    force a random minimum step count in [2, n_sup] before that sample is
    allowed to halt, regardless of what the halt head says — ports over
    the real TRM/HRM training recipe's exploration mechanism
    (halt_exploration_prob=0.1 in every real training run in both papers).
    Without this, a sample the head starts (possibly wrongly) halting
    early never generates the longer trajectory that would let the head's
    own training correct that mistake — it only ever sees the short
    trajectories its own decisions produce.

    continue_bias_init: at construction, the halt head's bias is set to
    this value (rather than the with_halt_head default of 0 — fine for
    that case, where halting is just an auxiliary probe riding on an
    otherwise-unaffected loop) so an untrained head starts by confidently
    predicting "there is more to gain, keep going" rather than "ambivalent"
    — mirrors the vendored q_head's own zero-weight + bias=-5 init trick,
    which exists for exactly this reason: under real ACT-driven training,
    halting gates the main training signal itself, so a bad early bias
    risks starving training rather than just slowing the aux loss's own
    convergence.

    halt_threshold: the training-time halting threshold (distinct from
    eval_cfg.halt_threshold, which only governs inference-time forward()/
    forward_with_carry() defaults).
    """

    def __init__(
        self,
        *args,
        halt_threshold: float = 0.0,
        halt_exploration_prob: float = 0.1,
        continue_bias_init: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not getattr(self.thinker, "with_halt_head", False):
            raise ValueError("ThinkerFrozenPainterACT requires thinker.with_halt_head=True.")
        self.halt_threshold = halt_threshold
        self.halt_exploration_prob = halt_exploration_prob
        with torch.no_grad():
            nn.init.constant_(self.thinker.halt_head.bias, continue_bias_init)
        self._act_carry: Optional[list[dict]] = None

    def train_step(
        self,
        micro_batches,
        accelerator,
        optimizers,
        ema,
        global_batch_size,
        global_step,
        **kwargs,
    ):
        device = accelerator.device
        K = len(micro_batches)
        n_sup = self.n_sup

        if self._act_carry is None:
            self._act_carry = []
            for mb in micro_batches:
                sample = self.painter._prepare_training_sample(mb, device)
                bsz = sample.x_noisy.shape[0]
                z_H, z_L = self.get_initial_states(bsz)
                self._act_carry.append({
                    "sample": sample,
                    "z_H": z_H.to(device),
                    "z_L": z_L.to(device),
                    "steps": torch.zeros(bsz, dtype=torch.long, device=device),
                    "halted": None,
                    "hist_z": [[] for _ in range(bsz)],
                    "hist_loss": [[] for _ in range(bsz)],
                })

        total_losses: dict[str, float] = {}
        completed_histories: list[tuple[torch.Tensor, torch.Tensor]] = []

        for i, slot in enumerate(self._act_carry):
            prev_halted = slot["halted"]
            if prev_halted is not None and prev_halted.any():
                fresh_sample = self.painter._prepare_training_sample(micro_batches[i], device)
                bsz = fresh_sample.x_noisy.shape[0]
                fresh_z_H, fresh_z_L = self.get_initial_states(bsz)
                fresh_z_H, fresh_z_L = fresh_z_H.to(device), fresh_z_L.to(device)

                slot["sample"] = _masked_merge_sample(slot["sample"], fresh_sample, prev_halted)
                m = prev_halted.view(-1, 1, 1)
                slot["z_H"] = torch.where(m, fresh_z_H, slot["z_H"])
                slot["z_L"] = torch.where(m, fresh_z_L, slot["z_L"])
                slot["steps"] = torch.where(prev_halted, torch.zeros_like(slot["steps"]), slot["steps"])
                for j in prev_halted.nonzero(as_tuple=True)[0].tolist():
                    slot["hist_z"][j] = []
                    slot["hist_loss"][j] = []

            noise_pred, logits, slot["z_H"], slot["z_L"] = self.reasoning_step(
                slot["sample"], slot["z_H"], slot["z_L"]
            )
            step_loss, loss_dict = self.loss_fn(noise_pred, logits, slot["sample"])
            for k, v in loss_dict.items():
                total_losses[k] = total_losses.get(k, 0.0) + v
            if step_loss.requires_grad:
                accelerator.backward(step_loss / (global_batch_size * K))

            slot["steps"] = slot["steps"] + 1

            per_sample_loss = (noise_pred.float() - slot["sample"].target.float()).pow(2).flatten(1).mean(1)
            z_H0 = slot["z_H"][:, 0].detach()
            loss_detached = per_sample_loss.detach()
            for j in range(z_H0.shape[0]):
                slot["hist_z"][j].append(z_H0[j])
                slot["hist_loss"][j].append(loss_detached[j])

            pred = self.thinker.predict_halt_value(slot["z_H"])
            is_last_step = slot["steps"] >= n_sup
            halted = is_last_step | (pred <= self.halt_threshold)

            if self.halt_exploration_prob > 0 and n_sup > 1:
                bsz = halted.shape[0]
                explore = torch.rand(bsz, device=device) < self.halt_exploration_prob
                min_steps = torch.where(
                    explore,
                    torch.randint(2, n_sup + 1, (bsz,), device=device),
                    torch.zeros(bsz, dtype=torch.long, device=device),
                )
                halted = halted & (slot["steps"] >= min_steps)

            slot["halted"] = halted

            for j in halted.nonzero(as_tuple=True)[0].tolist():
                z_seq = torch.stack(slot["hist_z"][j], dim=0)
                l_seq = torch.stack(slot["hist_loss"][j], dim=0)
                completed_histories.append((z_seq, l_seq))

        accelerator.clip_grad_norm_(self.get_thinker_params(), 1.0)
        enc_params = self.get_encoder_params()
        if enc_params:
            accelerator.clip_grad_norm_(enc_params, 1.0)
        lr = apply_lr_and_step(optimizers, global_step)
        global_step += 1
        if ema is not None:
            ema.update(self)

        losses = {k: v / K for k, v in total_losses.items()}
        if completed_histories:
            losses["halt_head_loss"] = self.thinker.train_halt_head_ragged(completed_histories)
            losses["act_steps_completed"] = float(len(completed_histories))

        return losses, lr, global_step

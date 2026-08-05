import dataclasses
from typing import Optional

import torch
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
        masking rather than one collective batch-mean decision: each sample
        keeps updating its own (z_H, z_L, logits) until ITS OWN prediction
        drops to/below halt_threshold, at which point it's frozen at that
        value while the rest of the batch keeps reasoning. The loop itself
        still runs (and every sample is still computed every iteration —
        masking doesn't skip compute for already-halted samples, it only
        discards their update) until every sample has halted or n_sup is
        reached, so the real compute cost of a call is bounded by its
        slowest/hardest sample, same as before — masking changes what
        result each sample ends up with, not how many iterations the batch
        as a whole takes. Off by default, so existing call sites are
        unaffected.

        steps_used: optional list; if given, the number of reasoning-loop
        iterations actually performed this call (i.e. the real compute cost
        — bounded by the hardest sample in the batch) is appended to it —
        see experiments/ablate_trm_loop_budget.py's "halt" axis. None by
        default, so existing call sites are unaffected.
        Not appended to under null_steering (no reasoning step is taken).

        halt_preds_out: optional list; if given, predict_halt_value(z_H) —
        the full per-sample (B,) tensor at that step's (possibly
        per-sample-frozen) z_H — is appended to it after every reasoning
        step, regardless of use_halt_head. Pairs with n_sup=self.n_sup and
        use_halt_head=False to record every individual sample's own
        prediction across the full, untruncated trajectory — see
        experiments/eval_halt_step_distribution.py. None by default, so
        existing call sites are unaffected.

        halt_steps_out: optional list; if given and use_halt_head=True, the
        per-sample (B,) tensor of the reasoning step at which each sample
        individually halted (n_sup if it never did) is appended to it — the
        real per-sample halting-step distribution, not an estimate. See
        experiments/eval_halt_step_profile.py. None by default, so existing
        call sites are unaffected.

        Returns: (DiffusionPrediction, z_H_next, z_L_next). Under
        null_steering, the carry is passed through unchanged so callers can
        thread state uniformly regardless of null_steering.
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
            logits = None
            step_count = 0
            halted = torch.zeros(bsz, dtype=torch.bool, device=device) if use_halt_head else None
            halt_step = torch.full((bsz,), n_sup, dtype=torch.long, device=device) if use_halt_head else None

            for _ in range(n_sup):
                new_logits, new_z_H, new_z_L = self.thinker.reasoning_step(
                    enc_emb, z_H, z_L, puzzle_ids, timesteps=sample.timesteps
                )
                step_count += 1

                if use_halt_head:
                    active = ~halted  # samples not yet halted still get this step's update
                    active_mask = active.view(-1, 1, 1)
                    z_H = torch.where(active_mask, new_z_H, z_H)
                    z_L = torch.where(active_mask, new_z_L, z_L)
                    logits = new_logits if logits is None else torch.where(active_mask, new_logits, logits)

                    pred = self.thinker.predict_halt_value(z_H)
                    if halt_preds_out is not None:
                        halt_preds_out.append(pred.detach())
                    newly_halted = active & (pred <= halt_threshold)
                    halt_step = torch.where(newly_halted, torch.full_like(halt_step, step_count), halt_step)
                    halted = halted | newly_halted
                    if halted.all():
                        break
                else:
                    logits, z_H, z_L = new_logits, new_z_H, new_z_L
                    if halt_preds_out is not None:
                        halt_preds_out.append(self.thinker.predict_halt_value(z_H).detach())

            if steps_used is not None:
                steps_used.append(step_count)
            if halt_steps_out is not None and use_halt_head:
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

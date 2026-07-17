import dataclasses
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
        """Encode condition fields from a DataSample into thinker token embeddings."""
        enc_keys = self.condition_encoder.condition_keys
        primary = getattr(sample, enc_keys[0])
        extra: dict = {}
        for k in enc_keys[1:]:
            if k == "x_noisy":
                extra["x_noisy"] = sample.enc_x_noisy if sample.enc_x_noisy is not None else sample.x_noisy
            else:
                extra[k] = getattr(sample, k)
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
        """Translate thinker logits to conditioning, then run the frozen painter."""
        painter_dtype = getattr(self.painter, "painter_dtype", None)
        ctx = (
            torch.autocast(device_type=sample.x_noisy.device.type, dtype=painter_dtype)
            if painter_dtype is not None
            else torch.autocast(device_type=sample.x_noisy.device.type, enabled=False)
        )
        with ctx:
            translator_extra = {k: getattr(sample, k) for k in self.thinker_painter_translator.condition_keys}
            steering = self.thinker_painter_translator(TRMOutput(logits=logits), **translator_extra)
            return self.painter(sample, steering).pred

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

    def forward(self, sample: DataSample, **kwargs) -> DiffusionPrediction:
        """Single inference pass: encode → think (n_sup steps) → translate → paint.

        No guidance is applied here.  Use CFGPredictor / NoisyGuidancePredictor
        from models.sampling to add guidance during SamplingPipeline.sample().
        Logits are included in the return value so eval_step can compute CE loss.
        """
        enc_emb = self._encode_condition(sample)
        puzzle_ids = sample.puzzle_id

        bsz = sample.x_noisy.shape[0]
        z_H, z_L = self.get_initial_states(bsz)
        z_H, z_L = z_H.to(sample.x_noisy.device), z_L.to(sample.x_noisy.device)

        logits = None
        for _ in range(self.n_sup):
            logits, z_H, z_L = self.thinker.reasoning_step(
                enc_emb, z_H, z_L, puzzle_ids, timesteps=sample.timesteps
            )

        painter_dtype = getattr(self.painter, "painter_dtype", None)
        ctx = (
            torch.autocast(device_type=sample.x_noisy.device.type, dtype=painter_dtype)
            if painter_dtype is not None
            else torch.autocast(device_type=sample.x_noisy.device.type, enabled=False)
        )
        with ctx:
            translator_extra = {k: getattr(sample, k) for k in self.thinker_painter_translator.condition_keys}
            steering = self.thinker_painter_translator(TRMOutput(logits=logits), **translator_extra)
            noise_pred = self.painter(sample, steering).pred

        return DiffusionPrediction(
            pred=noise_pred,
            pred_type=self.painter.scheduler.config.prediction_type,
            logits=logits,
        )

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

        for _ in range(self.n_sup):
            for d in mb_data:
                noise_pred, logits, d["z_H"], d["z_L"] = self.reasoning_step(
                    d["sample"], d["z_H"], d["z_L"]
                )
                step_loss, loss_dict = self.loss_fn(noise_pred, logits, d["sample"])
                for k, v in loss_dict.items():
                    total_losses[k] = total_losses.get(k, 0.0) + v
                if step_loss.requires_grad:
                    accelerator.backward(step_loss / (global_batch_size * K))

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

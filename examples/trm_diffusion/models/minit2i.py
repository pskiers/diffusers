"""
models/minit2i.py — frozen MiniT2I-B/16 painter (Wang et al. 2026) for thinker
steering on text-to-image, plus its flow-matching scheduler and the thinker's
input encoder.

MiniT2I is a pixel-space MM-DiT (third_party/minit2i/mmdit.py, loaded verbatim
with the released weights): FLAN-T5-Large text tokens and 16x16 image patches
run through 17 joint-attention blocks, and the model predicts the clean image.
It has no timestep or AdaLN pathway (its blocks ignore `vec`), so steering goes
into the token stream instead.

Steering ("read early, steer late"): the first `steer_from_block` blocks run
unsteered; their output (DataSample.painter_hidden) is what the thinker reads,
via MiniT2IHiddenEncoder; PainterResidualTranslator turns the thinker's output
into one residual per remaining block, added to the image tokens after that
block. The early half depends only on (x_t, text), so it runs once per sample
per denoising step — not once per reasoning step.

Objective and sampler follow the model we compare against:
    x_t = t x0 + (1 - t) eps,  eps ~ N(0, noise_scale^2),  t ~ logit-normal(-0.8, 0.8)
    v   = (x0 - x_t) / max(1 - t, 0.05),  v_hat likewise from the predicted x0
    loss = mean (v_hat - v)^2,  text dropped (mask zeroed) for 10% of samples
    Euler from t = 0 to 1 with classifier-free guidance.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from datasets.data_sample import DataSample
from models.condition_encoders import ConditionEncoderBase
from models.interfaces import DiffusionPrediction, ThinkerSteering, TRMInput
from models.painter_base import PainterBase
from third_party.minit2i import mmdit

MINIT2I_REVISION = "5b4da2870d1554992670d1005567e9d55b1b37bb"  # see third_party/minit2i/README.md


# ── Scheduler ─────────────────────────────────────────────────────────────────


class MiniT2IFlowScheduler:
    """Euler sampler for MiniT2I flow matching (t = 0 noise -> t = 1 image),
    exposing the part of the diffusers scheduler interface SamplingPipeline uses
    (config, init_noise_sigma, set_timesteps, timesteps, step).

    The painter returns the velocity with 1 - t clamped at train_velocity_clamp.
    With sample_velocity_clamp equal to it (default 0.05) this is exactly
    looped_dit/diffusion.py: euler_sample — the protocol Looped-DiT is evaluated
    with; 0.001 instead gives MiniT2I's own Hugging Face pipeline. Classifier-free
    guidance commutes with the conversion between the two (it is linear in the
    prediction for fixed x_t and t).
    """

    def __init__(
        self,
        noise_scale: float = 2.0,
        train_velocity_clamp: float = 0.05,
        sample_velocity_clamp: float = 0.05,
        num_train_timesteps: int = 1000,
    ):
        # num_train_timesteps: only read by SamplingPipeline as the `T` its
        # guidance schedules receive; continuous-time training doesn't use it.
        self.config = SimpleNamespace(num_train_timesteps=num_train_timesteps, prediction_type="v_prediction")
        self.init_noise_sigma = noise_scale
        self.train_velocity_clamp = train_velocity_clamp
        self.sample_velocity_clamp = sample_velocity_clamp
        self.timesteps: Optional[torch.Tensor] = None

    def set_timesteps(self, num_inference_steps: int, device=None) -> None:
        self._grid = torch.linspace(0.0, 1.0, num_inference_steps + 1, device=device)
        self.timesteps = self._grid[:-1]
        self._index = 0

    def step(self, model_output: torch.Tensor, timestep: torch.Tensor, sample: torch.Tensor):
        t0, t1 = self._grid[self._index], self._grid[self._index + 1]
        if not torch.isclose(torch.as_tensor(timestep, device=t0.device, dtype=t0.dtype), t0):
            raise ValueError(f"step() called with t={float(timestep)}, expected t={float(t0)} (step {self._index})")
        self._index += 1
        v = model_output.float()
        if self.sample_velocity_clamp != self.train_velocity_clamp:
            x0 = sample + v * (1.0 - t0).clamp_min(self.train_velocity_clamp)
            v = (x0 - sample) / (1.0 - t0).clamp_min(self.sample_velocity_clamp)
        return SimpleNamespace(prev_sample=sample + v * (t1 - t0))


# ── Fused-attention versions of the vendored blocks ───────────────────────────
# Same parameters and math as mmdit.py; only the attention itself goes through
# F.scaled_dot_product_attention instead of materializing the full softmax
# matrix (12 x 1280 x 1280 per sample), which training could not afford.


def _attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """(B, L, H, D) inputs → (B, L, H, D); softmax(q k^T / sqrt(D)) v as in mmdit.py."""
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    return out.transpose(1, 2)


class _FusedPlainTextTransformerBlock(mmdit.PlainTextTransformerBlock):
    def forward(self, txt):
        b, length, _ = txt.shape
        qkv = self.qkv(self.norm1(txt)).reshape(b, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = self.rope(self.q_norm(q))
        k = self.rope(self.k_norm(k))
        out = _attention(q, k, v).reshape(b, length, -1)
        txt = txt + self.attn_proj(out)
        txt = txt + self.mlp(self.norm2(txt))
        return txt


class _FusedDoubleStreamDiTBlock(mmdit.DoubleStreamDiTBlock):
    def forward(self, x, txt, vec=None):
        b, li, _ = x.shape
        lt = txt.shape[1]
        x_norm = self.img_norm1(x)
        txt_norm = self.txt_norm1(txt)
        qkv_i = self.img_qkv(x_norm).reshape(b, li, 3, self.num_heads, self.head_dim)
        qkv_t = self.txt_qkv(txt_norm).reshape(b, lt, 3, self.num_heads, self.head_dim)
        q_i, k_i, v_i = qkv_i[:, :, 0], qkv_i[:, :, 1], qkv_i[:, :, 2]
        q_t, k_t, v_t = qkv_t[:, :, 0], qkv_t[:, :, 1], qkv_t[:, :, 2]
        q_i, k_i = self.q_norm(q_i), self.k_norm(k_i)
        q_t, k_t = self.q_norm(q_t), self.k_norm(k_t)
        q = self.rope(torch.cat([q_t, q_i], dim=1), txt_len=lt)
        k = self.rope(torch.cat([k_t, k_i], dim=1), txt_len=lt)
        v = torch.cat([v_t, v_i], dim=1)
        out = _attention(q, k, v)
        x = x + self.img_attn_proj(out[:, lt:].reshape(b, li, -1))
        txt = txt + self.txt_attn_proj(out[:, :lt].reshape(b, lt, -1))
        x = x + self.img_mlp(self.img_norm2(x))
        txt = txt + self.txt_mlp(self.txt_norm2(txt))
        return x, txt


def load_minit2i(repo: str = "MiniT2I/MiniT2I", subfolder: str = "minit2i-b-16", revision: str = MINIT2I_REVISION):
    """The released MiniT2I backbone as a vendored mmdit.MMJiT, frozen, with
    fused attention."""

    config = json.loads(open(hf_hub_download(repo, f"{subfolder}/transformer/config.json", revision=revision)).read())
    names = {f.name for f in dataclasses.fields(mmdit.MMJiTConfig)}
    cfg = mmdit.MMJiTConfig(**{k: (tuple(v) if k == "cfg_interval" else v) for k, v in config.items() if k in names})
    net = mmdit.MMJiT(cfg)
    weights = load_file(hf_hub_download(repo, f"{subfolder}/transformer/diffusion_pytorch_model.safetensors", revision=revision))
    net.load_state_dict({k.removeprefix("model.net."): v for k, v in weights.items()}, strict=True)
    for block in net.txt_preamble_blocks:
        block.__class__ = _FusedPlainTextTransformerBlock
    for block in net.double_blocks:
        block.__class__ = _FusedDoubleStreamDiTBlock
    return net.eval().requires_grad_(False)


# ── Painter ───────────────────────────────────────────────────────────────────


class MiniT2IPainter(PainterBase):
    """Frozen MiniT2I with its frozen FLAN-T5 text encoder, steerable after
    block `steer_from_block` (see module docstring).

    Reads DataSample.token_conditions / token_mask (T5 token ids and attention
    mask, e.g. from Std120KDataset); images and x_t are in [-1, 1]. Predicts the
    velocity v_hat (pred_type "v_prediction") consumed by MiniT2IFlowScheduler
    and trained against DataSample.target by plain MSE.

    Args:
        scheduler, train_cfg, eval_cfg: injected by the owning model.
        optim_cfg: optimizer for the trainable parts outside the thinker
            (condition encoder, translator) — see ThinkerFrozenPainterBase.
        repo, subfolder, revision: which released checkpoint.
        text_encoder: tokenizer/encoder MiniT2I was trained with.
        steer_from_block: blocks [0, steer_from_block) form the read-only early
            half; the rest get steering residuals.
        painter_dtype: autocast dtype (Looped-DiT trains and samples in bf16).
        noise_scale, t_logit_mean, t_logit_std, velocity_clamp,
        label_drop_rate: looped_dit/diffusion.py: training_loss defaults.
    """

    def __init__(
        self,
        scheduler,
        train_cfg,
        eval_cfg,
        optim_cfg,
        repo: str = "MiniT2I/MiniT2I",
        subfolder: str = "minit2i-b-16",
        revision: str = MINIT2I_REVISION,
        text_encoder: str = "google/flan-t5-large",
        steer_from_block: int = 6,
        painter_dtype: Optional[str] = "bfloat16",
        noise_scale: float = 2.0,
        t_logit_mean: float = -0.8,
        t_logit_std: float = 0.8,
        velocity_clamp: float = 0.05,
        label_drop_rate: float = 0.1,
        eval_callbacks=None,
        sampling_pipeline=None,
    ):
        super().__init__()
        from transformers import T5EncoderModel

        self.scheduler = scheduler
        self.train_cfg = train_cfg
        self.eval_cfg = eval_cfg
        self.optim_cfg = optim_cfg
        self.eval_callbacks = [instantiate(cb) for cb in eval_callbacks] if eval_callbacks else []
        self.sampling_pipeline = instantiate(sampling_pipeline) if sampling_pipeline is not None else None
        self.net = load_minit2i(repo, subfolder, revision)
        self.text_encoder = T5EncoderModel.from_pretrained(text_encoder).eval().requires_grad_(False)
        if not 0 < steer_from_block < len(self.net.double_blocks):
            raise ValueError(f"steer_from_block must be in [1, {len(self.net.double_blocks) - 1}]")
        self.steer_from_block = steer_from_block
        self.painter_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[painter_dtype] if painter_dtype else None
        self.noise_scale = noise_scale
        self.t_logit_mean = t_logit_mean
        self.t_logit_std = t_logit_std
        self.velocity_clamp = velocity_clamp
        self.label_drop_rate = label_drop_rate
        self.condition_keys = ["token_conditions", "token_mask"]

    # Frozen modules stay in eval mode (T5 has dropout) whatever the owner sets.
    def train(self, mode: bool = True):
        super().train(mode)
        self.net.eval()
        self.text_encoder.eval()
        return self

    @property
    def num_steered_blocks(self) -> int:
        return len(self.net.double_blocks) - self.steer_from_block

    @property
    def noise_shape(self) -> tuple:
        s = self.net.cfg.image_size
        return (self.net.cfg.in_channels, s, s)

    def decode_for_eval(self, x: torch.Tensor) -> torch.Tensor:
        return ((x.float().clamp(-1.0, 1.0) + 1.0) / 2.0).clamp(0.0, 1.0)

    def images_to_log(self, images: torch.Tensor) -> torch.Tensor:
        return self.decode_for_eval(images)

    def _autocast(self, device: torch.device):
        if self.painter_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=device.type, dtype=self.painter_dtype)

    # ── The two halves of mmdit.MMJiT.forward ────────────────────────────────

    @torch.no_grad()
    def encode_head(self, x_t: torch.Tensor, token_ids: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        """Text encoding + embeddings + blocks [0, steer_from_block): the joint
        (B, L_txt + L_img, D) token stream, text first (MMJiT's own order)."""
        net = self.net
        with self._autocast(x_t.device):
            # looped_dit/pipeline.py: TextEncoder.encode
            context = self.text_encoder(input_ids=token_ids, attention_mask=token_mask).last_hidden_state
            # mmdit.MMJiT.forward up to its block loop (`vec` is computed there but
            # never used by any block, so it is skipped here).
            mask = token_mask.to(device=context.device)
            context = torch.where(mask[:, :, None] > 0.5, context, net.mask_token.to(dtype=context.dtype))
            x = net.img_embedder(x_t)
            x = x + mmdit.get_2d_sincos_pos_embed(net.cfg.hidden_size, net.latent_img_size, x.device, x.dtype)[None]
            txt = net.txt_embedder(context.to(dtype=net.txt_embedder.weight.dtype))
            for block in net.txt_preamble_blocks:
                txt = block(txt)
            for block in net.double_blocks[: self.steer_from_block]:
                x, txt = block(x, txt, None)
        return torch.cat([txt, x], dim=1)

    def decode_tail(self, hidden: torch.Tensor, block_residuals: Optional[list] = None) -> torch.Tensor:
        """Blocks [steer_from_block, 17) + output head → predicted clean image.
        block_residuals[j] is added to the image tokens after the j-th of them."""
        net = self.net
        n_img = net.latent_img_size ** 2
        txt, x = hidden[:, :-n_img], hidden[:, -n_img:]
        with self._autocast(hidden.device):
            for j, block in enumerate(net.double_blocks[self.steer_from_block :]):
                x, txt = block(x, txt, None)
                if block_residuals is not None:
                    x = x + block_residuals[j].to(x.dtype)
            # MMJiT applies final_layer to cat([txt, x]) and keeps the image part;
            # it acts per token, so applying it to x alone is the same.
            return net.unpatchify(net.final_layer(x))

    def fill_runtime_fields(self, sample: DataSample) -> DataSample:
        """Compute painter_hidden for the sample's x_t and text (see
        ThinkerFrozenPainterBase.forward_with_carry)."""
        hidden = self.encode_head(sample.x_noisy, sample.token_conditions, sample.token_mask)
        return dataclasses.replace(sample, painter_hidden=hidden)

    def velocity(self, x0_hat: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        scale = (1.0 - t.float()).view(-1, 1, 1, 1).clamp_min(self.velocity_clamp)
        return (x0_hat.float() - x_t.float()) / scale

    def forward(self, sample: DataSample, steering: Optional[ThinkerSteering] = None) -> DiffusionPrediction:
        hidden = sample.painter_hidden
        if hidden is None:
            hidden = self.encode_head(sample.x_noisy, sample.token_conditions, sample.token_mask)
        residuals = steering.to_painter_kwargs()["block_residuals"] if steering is not None else None
        if residuals is not None and len(residuals) != self.num_steered_blocks:
            raise ValueError(
                f"{len(residuals)} steering residuals for {self.num_steered_blocks} steered blocks "
                f"(steer_from_block={self.steer_from_block}) — set translator.num_blocks to match"
            )
        x0_hat = self.decode_tail(hidden, residuals)
        return DiffusionPrediction(
            pred=self.velocity(x0_hat, sample.x_noisy, sample.timesteps), pred_type="v_prediction"
        )

    def _prepare_training_sample(self, mb: DataSample, device: torch.device) -> DataSample:
        """x_t, t and the velocity target, drawn exactly as
        looped_dit/diffusion.py: training_loss (same draws, same order), then
        painter_hidden for that x_t."""
        images = mb.images.to(device).float()
        token_ids = mb.token_conditions.to(device)
        token_mask = mb.token_mask.to(device)
        b = images.shape[0]
        if self.label_drop_rate > 0:
            drop = torch.rand(b, device=device) < self.label_drop_rate
            token_mask = torch.where(drop[:, None], torch.zeros_like(token_mask), token_mask)
        t = torch.sigmoid(torch.randn(b, device=device) * self.t_logit_std + self.t_logit_mean).clamp(1e-5, 1 - 1e-5)
        t4 = t[:, None, None, None]
        noise = torch.randn_like(images) * self.noise_scale
        x_t = images * t4 + noise * (1.0 - t4)
        target = (images - x_t) / (1.0 - t4).clamp_min(self.velocity_clamp)
        sample = DataSample(
            images=images, token_conditions=token_ids, token_mask=token_mask, x_noisy=x_t, timesteps=t, target=target
        )
        return self.fill_runtime_fields(sample)


# ── Thinker input ─────────────────────────────────────────────────────────────


class MiniT2IHiddenEncoder(ConditionEncoderBase):
    """Thinker tokens from DataSample.painter_hidden: the image tokens average-
    pooled `pool` x `pool` (32x32 → 16x16 for B/16), then the text tokens, each
    normalized and projected to the thinker width. Order [image grid; text] —
    PainterResidualTranslator reads the image grid back from the front.

    The projections are initialized so tokens come out at the thinker's own
    token-embedding scale (std 1/sqrt(output_dim)), which the TRM then
    multiplies by sqrt(output_dim).
    """

    condition_keys = ["painter_hidden"]

    def __init__(self, output_dim: int, painter_dim: int = 768, img_grid: int = 32, pool: int = 2):
        super().__init__()
        self.img_grid = img_grid
        self.pool = pool
        self.img_norm = nn.LayerNorm(painter_dim, elementwise_affine=False)
        self.txt_norm = nn.LayerNorm(painter_dim, elementwise_affine=False)
        self.img_proj = nn.Linear(painter_dim, output_dim)
        self.txt_proj = nn.Linear(painter_dim, output_dim)
        for proj in (self.img_proj, self.txt_proj):
            nn.init.normal_(proj.weight, std=1.0 / (painter_dim * output_dim) ** 0.5)
            nn.init.zeros_(proj.bias)

    @property
    def num_img_tokens(self) -> int:
        return (self.img_grid // self.pool) ** 2

    def forward(self, painter_hidden: torch.Tensor, timesteps=None, **_) -> TRMInput:
        h = painter_hidden.float()
        b, _, d = h.shape
        n_img = self.img_grid**2
        txt, img = h[:, :-n_img], h[:, -n_img:]
        img = img.transpose(1, 2).reshape(b, d, self.img_grid, self.img_grid)
        img = F.avg_pool2d(img, self.pool).flatten(2).transpose(1, 2)
        tokens = torch.cat([self.img_proj(self.img_norm(img)), self.txt_proj(self.txt_norm(txt))], dim=1)
        return TRMInput(enc_emb=tokens)

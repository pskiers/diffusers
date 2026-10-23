"""
models/mmdit_painter.py — generic MM-DiT painter (the MiniT2I architecture at any
size, on any conditioning), its flow-matching scheduler, and the thinker's input
encoder over its hidden states.

The network is MiniT2I's (third_party/minit2i/mmdit.py: MMJiT): two token streams
— image patches and condition tokens — each with its own weights, attending
jointly in every double-stream block; no timestep pathway; predicts the clean
image. What it conditions on is a CondTokenizer:

    T5TextTokenizer         frozen T5 over token_conditions/token_mask (MiniT2I)
    PatchCondTokenizer      spatial_conditions cut into patches (2D RoPE)
    EmbeddingCondTokenizer  embedding_conditions/embedding_mask, one token each

Positions whose mask is 0 are replaced by the learned mask_token, as MiniT2I does
for text padding; dropping the condition for CFG zeroes every condition field
(PainterBase.null_condition_sample), which is also how training drops it.

Steering ("read early, steer late", see models/minit2i.py): blocks
[0, steer_from_block) run unsteered and their output, DataSample.painter_hidden
([cond; image] tokens), is what PainterHiddenEncoder hands the thinker;
PainterResidualTranslator turns the thinker's output into one residual per later
block for the image stream and, optionally, the condition stream.

Objective and sampler (MiniT2I / Looped-DiT):
    x_t = t x0 + (1 - t) eps,  eps ~ N(0, noise_scale^2),  t ~ logit-normal
    v   = (x0 - x_t) / max(1 - t, velocity_clamp),  v_hat likewise from the predicted x0
    loss = mean (v_hat - v)^2;  Euler from t = 0 (noise) to 1 (image)
The model works in model_range ([-1, 1] as MiniT2I by default); pixel_range says what the
dataset's images are in.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import math
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from tqdm.auto import tqdm

from datasets.data_sample import DataSample
from models.base import BaseModel
from models.condition_encoders import ConditionEncoderBase
from models.interfaces import DiffusionPrediction, ThinkerSteering, TRMInput
from models.optim_utils import ScheduledOptimizer, all_reduce_grads, apply_lr_and_step
from models.painter_base import PainterBase, _load_frozen_checkpoint
from third_party.minit2i import mmdit

_RUNTIME_FIELDS = {"x_noisy", "timesteps", "target", "enc_x_noisy", "painter_hidden", "loss_weight"}


# ── Scheduler ─────────────────────────────────────────────────────────────────


class FlowMatchingScheduler:
    """Euler sampler for MiniT2I flow matching (t = 0 noise -> t = 1 image),
    exposing the part of the diffusers scheduler interface SamplingPipeline uses
    (config, init_noise_sigma, set_timesteps, timesteps, step).

    The painter returns the velocity with 1 - t clamped at train_velocity_clamp.
    With sample_velocity_clamp equal to it (default 0.05) this is exactly
    looped_dit/diffusion.py: euler_sample — the protocol Looped-DiT is evaluated
    with; 0.001 instead gives MiniT2I's own Hugging Face pipeline. Classifier-free
    guidance commutes with the conversion between the two (it is linear in the
    prediction for fixed x_t and t).

    The painter takes its training noise scale and velocity clamp from here, so
    training and sampling cannot disagree on them.
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
# matrix (12 x 1280 x 1280 per sample for MiniT2I), which training could not
# afford. cond_rope picks the condition stream's rotary embedding: "1d"
# (mmdit.py's own, for text), "2d" (the image stream's, for patch tokens on a
# square grid) or "none" (unordered sets, e.g. objects).


def _attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """(B, L, H, D) inputs → (B, L, H, D); softmax(q k^T / sqrt(D)) v as in mmdit.py."""
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    return out.transpose(1, 2)


# Rotary embeddings: mmdit.py's TextRotaryEmbedding1D / VisionRotaryEmbeddingFast
# rebuild their cos/sin tables in every forward (the 2D one from int(sqrt(length))),
# which torch.compile (inductor, torch 2.4.1) fails to code-generate. Same formulas,
# op for op, but the tables are built once per shape outside the compiled graph and
# cached; applying them is unchanged, so outputs are bit-identical to mmdit.py's.


@torch.compiler.disable
@functools.lru_cache(maxsize=64)
def _rope_tables(kind: str, length: int, d: int, theta: float, device: torch.device, dtype: torch.dtype):
    with torch.inference_mode(False), torch.no_grad():
        if kind == "text":  # TextRotaryEmbedding1D.forward
            inv = 1.0 / (theta ** (torch.arange(0, d, 2, device=device, dtype=torch.float32) / d))
            pos = torch.arange(length, device=device, dtype=torch.float32)
            angles = torch.einsum("l,f->lf", pos, inv)
            angles = torch.cat([angles, angles], dim=-1)
        else:  # VisionRotaryEmbeddingFast.forward, with dim = head_dim // 2
            dim = d // 2
            side = int(math.sqrt(length))
            if side * side != length:
                raise ValueError(f"image token length must be square, got {length}")
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32)[: dim // 2] / dim))
            t = torch.arange(side, device=device, dtype=torch.float32)
            base = torch.einsum("l,f->lf", t, freqs)
            f_h, f_w = torch.broadcast_tensors(base[:, None, :], base[None, :, :])
            angles = torch.cat([f_h, f_w], dim=-1)
            angles = torch.cat([angles, angles], dim=-1).reshape(length, -1)
        return angles.cos().to(dtype=dtype), angles.sin().to(dtype=dtype)


def _apply_rope(x: torch.Tensor, kind: str, theta: float = 10000.0) -> torch.Tensor:
    """(B, L, H, D) → rotated, as mmdit.py's rotary embeddings."""
    cos, sin = _rope_tables(kind, x.shape[1], x.shape[-1], float(theta), x.device, x.dtype)
    return x * cos[None, :, None, :] + mmdit.rotate_half(x) * sin[None, :, None, :]


class _FusedPlainTextTransformerBlock(mmdit.PlainTextTransformerBlock):
    cond_rope: str = "1d"

    def _rope(self, x):
        if self.cond_rope == "1d":
            return _apply_rope(x, "text", self.rope.theta)
        if self.cond_rope == "2d":
            return _apply_rope(x, "vision")
        return x

    def forward(self, txt):
        b, length, _ = txt.shape
        qkv = self.qkv(self.norm1(txt)).reshape(b, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = self._rope(self.q_norm(q))
        k = self._rope(self.k_norm(k))
        out = _attention(q, k, v).reshape(b, length, -1)
        txt = txt + self.attn_proj(out)
        txt = txt + self.mlp(self.norm2(txt))
        return txt


class _FusedDoubleStreamDiTBlock(mmdit.DoubleStreamDiTBlock):
    cond_rope: str = "1d"

    def _rope(self, q_t, q_i):
        q_i = _apply_rope(q_i, "vision", self.rope.vision_rope.theta)
        if self.cond_rope == "1d":
            q_t = _apply_rope(q_t, "text", self.rope.text_rope.theta)
        elif self.cond_rope == "2d":
            q_t = _apply_rope(q_t, "vision", self.rope.vision_rope.theta)
        return torch.cat([q_t, q_i], dim=1)

    def forward(self, x, txt, vec=None):
        adaln = getattr(self, "adaln", None)
        if adaln is not None and vec is not None:
            return self._forward_adaln(x, txt, adaln(vec).chunk(12, dim=-1))
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
        q = self._rope(q_t, q_i)
        k = self._rope(k_t, k_i)
        v = torch.cat([v_t, v_i], dim=1)
        out = _attention(q, k, v)
        x = x + self.img_attn_proj(out[:, lt:].reshape(b, li, self.num_heads * self.head_dim))
        txt = txt + self.txt_attn_proj(out[:, :lt].reshape(b, lt, self.num_heads * self.head_dim))
        x = x + self.img_mlp(self.img_norm2(x))
        txt = txt + self.txt_mlp(self.txt_norm2(txt))
        return x, txt

    def _forward_adaln(self, x, txt, mods):
        """The block with DiT's adaLN-Zero timestep modulation on both streams
        (shift/scale before attention and MLP, gates on their outputs; see
        build_mmdit(time_cond=True))."""
        sh_ai, sc_ai, g_ai, sh_mi, sc_mi, g_mi, sh_at, sc_at, g_at, sh_mt, sc_mt, g_mt = mods
        b, li, _ = x.shape
        lt = txt.shape[1]
        x_norm = mmdit.modulate(self.img_norm1(x), sh_ai, sc_ai)
        txt_norm = mmdit.modulate(self.txt_norm1(txt), sh_at, sc_at)
        qkv_i = self.img_qkv(x_norm).reshape(b, li, 3, self.num_heads, self.head_dim)
        qkv_t = self.txt_qkv(txt_norm).reshape(b, lt, 3, self.num_heads, self.head_dim)
        q_i, k_i, v_i = qkv_i[:, :, 0], qkv_i[:, :, 1], qkv_i[:, :, 2]
        q_t, k_t, v_t = qkv_t[:, :, 0], qkv_t[:, :, 1], qkv_t[:, :, 2]
        q_i, k_i = self.q_norm(q_i), self.k_norm(k_i)
        q_t, k_t = self.q_norm(q_t), self.k_norm(k_t)
        out = _attention(self._rope(q_t, q_i), self._rope(k_t, k_i), torch.cat([v_t, v_i], dim=1))
        x = x + g_ai[:, None] * self.img_attn_proj(out[:, lt:].reshape(b, li, self.num_heads * self.head_dim))
        txt = txt + g_at[:, None] * self.txt_attn_proj(out[:, :lt].reshape(b, lt, self.num_heads * self.head_dim))
        x = x + g_mi[:, None] * self.img_mlp(mmdit.modulate(self.img_norm2(x), sh_mi, sc_mi))
        txt = txt + g_mt[:, None] * self.txt_mlp(mmdit.modulate(self.txt_norm2(txt), sh_mt, sc_mt))
        return x, txt


def fuse_attention(net: mmdit.MMJiT, cond_rope: str = "1d") -> mmdit.MMJiT:
    """Swap an MMJiT's blocks to the fused-attention versions above, in place."""
    if cond_rope not in ("1d", "2d", "none"):
        raise ValueError(f"cond_rope must be '1d', '2d' or 'none', got {cond_rope!r}")
    for block in net.txt_preamble_blocks:
        block.__class__ = _FusedPlainTextTransformerBlock
        block.cond_rope = cond_rope
    for block in net.double_blocks:
        block.__class__ = _FusedDoubleStreamDiTBlock
        block.cond_rope = cond_rope
    return net


class _PlainPatchEmbed(nn.Module):
    """DiT's patch embedding: one conv from patch_size x patch_size pixels straight to
    the width (MMJiT's BottleneckPatchEmbed goes through pca_channels first)."""

    def __init__(self, patch_size: int, in_channels: int, hidden_size: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size, bias=True)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


def build_mmdit(
    cond_dim: int,
    image_size: int,
    patch_size: int,
    in_channels: int,
    hidden_size: int = 384,
    depth: int = 8,
    preamble_depth: int = 1,
    num_heads: int = 6,
    head_dim: int = 64,
    mlp_ratio: float = 2.6667,
    pca_channels: int = 128,
    cond_rope: str = "1d",
    time_cond: bool = False,
    norm: str = "rms",
    qk_norm: bool = True,
    patch_embed: str = "bottleneck",
    zero_init_output: bool = True,
) -> mmdit.MMJiT:
    """A freshly initialized MMJiT of the given size, with fused attention.

    Both streams have width hidden_size. MMJiT's pooled-text embedder is dropped,
    and so is its timestep embedder unless time_cond: no block reads them.
    Initialization as DiT / JiT: xavier-uniform linears and patch embeddings,
    zero biases, zero output layer (so the model starts by predicting x0 = 0).

    time_cond: timestep conditioning as DiT's adaLN-Zero (which MiniT2I lacks):
    MMJiT's TimestepEmbedder gives `vec`; every double-stream block gets a
    zero-initialized SiLU+Linear producing shift/scale/gate for attention and MLP
    of both streams (block.adaln), and the output layer a shift/scale
    (net.final_adaln). Zero-initialized, so every block starts as the identity.

    DiT-style alternatives to MMJiT's design choices (for ablations):
    norm: "rms" (MMJiT's RMSNorm) or "layer" (LayerNorm, eps 1e-6) for every
        pre-norm and the output norm; qk_norm: False drops the RMSNorm on q and k;
    patch_embed: "bottleneck" (MMJiT's) or "plain" (_PlainPatchEmbed);
    zero_init_output: False keeps the output layer's xavier init.
    """
    if norm not in ("rms", "layer") or patch_embed not in ("bottleneck", "plain"):
        raise ValueError(f"norm must be 'rms'/'layer' and patch_embed 'bottleneck'/'plain', got {norm!r}, {patch_embed!r}")
    if image_size % patch_size:
        raise ValueError(f"image_size {image_size} is not a multiple of patch_size {patch_size}")
    cfg = mmdit.MMJiTConfig(
        image_size=image_size,
        patch_size=patch_size,
        in_channels=in_channels,
        txt_input_size=cond_dim,
        hidden_size=hidden_size,
        txt_hidden_size=hidden_size,
        cond_vec_size=hidden_size,
        depth_double=depth,
        txt_preamble_depth=preamble_depth,
        num_heads=num_heads,
        head_dim=head_dim,
        mlp_ratio=mlp_ratio,
        pca_channels=pca_channels,
    )
    net = mmdit.MMJiT(cfg)
    del net.pooled_embedder
    if patch_embed == "plain":
        net.img_embedder = _PlainPatchEmbed(patch_size, in_channels, hidden_size)
    blocks = [*net.txt_preamble_blocks, *net.double_blocks]
    if norm == "layer":
        names = ["norm1", "norm2", "img_norm1", "img_norm2", "txt_norm1", "txt_norm2"]
        for block in blocks:
            for name in names:
                if hasattr(block, name):
                    setattr(block, name, nn.LayerNorm(getattr(block, name).weight.shape[0], eps=1e-6))
        net.final_layer.norm_final = nn.LayerNorm(hidden_size, eps=1e-6)
    if not qk_norm:
        for block in blocks:
            block.q_norm, block.k_norm = nn.Identity(), nn.Identity()
    if not time_cond:
        del net.t_embedder
    for m in net.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight.view(m.weight.shape[0], -1))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    if zero_init_output:
        nn.init.zeros_(net.final_layer.linear.weight)
    net.time_cond = time_cond
    if time_cond:
        for block in net.double_blocks:
            block.adaln = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 12 * hidden_size))
        net.final_adaln = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        for mod in [*(b.adaln for b in net.double_blocks), net.final_adaln]:
            nn.init.zeros_(mod[1].weight)
            nn.init.zeros_(mod[1].bias)
    nn.init.zeros_(net.final_layer.linear.bias)
    return fuse_attention(net, cond_rope)


# ── Condition tokenizers ──────────────────────────────────────────────────────


class CondTokenizerBase(nn.Module):
    """DataSample fields → condition-stream input.

    forward(*fields) takes the fields named in condition_keys, in order, and
    returns (tokens (B, L, out_dim), mask (B, L)); positions with mask 0 become
    the painter's mask_token. rope: the default cond_rope for these tokens.
    """

    condition_keys: list[str] = []
    rope: str = "1d"
    out_dim: int

    def forward(self, *fields):
        raise NotImplementedError


class T5TextTokenizer(CondTokenizerBase):
    """Frozen T5 encoder over caption token ids (MiniT2I's text pathway)."""

    condition_keys = ["token_conditions", "token_mask"]
    rope = "1d"

    def __init__(self, text_encoder: str = "google/flan-t5-large"):
        super().__init__()
        from transformers import T5EncoderModel

        self.encoder = T5EncoderModel.from_pretrained(text_encoder).eval().requires_grad_(False)
        self.out_dim = self.encoder.config.d_model

    def train(self, mode: bool = True):  # frozen: T5's dropout stays off
        return super().train(False)

    @torch.no_grad()
    def forward(self, token_ids, token_mask):
        # looped_dit/pipeline.py: TextEncoder.encode
        return self.encoder(input_ids=token_ids, attention_mask=token_mask).last_hidden_state, token_mask


class PatchCondTokenizer(CondTokenizerBase):
    """spatial_conditions (B, C, H, W) cut into patch_size patches, one token per
    patch (raw pixels; the painter's linear condition embedder projects them).
    With the same patch grid as the image stream and cond_rope "2d", condition
    and image tokens of the same patch share their rotary position."""

    condition_keys = ["spatial_conditions"]
    rope = "2d"

    def __init__(self, in_channels: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.out_dim = in_channels * patch_size**2

    def forward(self, cond):
        tokens = F.unfold(cond.float(), kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
        return tokens, torch.ones(tokens.shape[:2], device=cond.device)


class NullCondTokenizer(CondTokenizerBase):
    """No condition stream: zero condition tokens, so the MM-DiT's joint attention
    is plain self-attention over the image tokens (an unconditional, effectively
    single-stream DiT; the condition-stream weights are never used). Use with
    net.preamble_depth 0. Accepts and ignores other tokenizers' arguments, so a
    config can switch to it by overriding _target_ alone."""

    condition_keys = []
    rope = "none"
    out_dim = 1

    def __init__(self, **_ignored):
        super().__init__()


class EmbeddingCondTokenizer(CondTokenizerBase):
    """embedding_conditions (B, N, in_dim) with embedding_mask (B, N): one token
    per entry (e.g. per CLEVR object), padding entries become mask tokens. An
    unordered set by default (rope "none")."""

    condition_keys = ["embedding_conditions", "embedding_mask"]
    rope = "none"

    def __init__(self, in_dim: int):
        super().__init__()
        self.out_dim = in_dim

    def forward(self, feats, mask):
        return feats.float(), mask.reshape(mask.shape[0], -1)


# ── Painter ───────────────────────────────────────────────────────────────────


class MMDiTPainter(PainterBase, BaseModel):
    """MM-DiT painter: trainable on its own (mode=painter_base), or frozen and
    steered by a thinker (mode=thinker_base) after block `steer_from_block`.

    Args:
        scheduler, train_cfg, eval_cfg: injected by the factory / owning model.
        optim_cfg: optimizer for this painter's training; when frozen, for the
            trainable parts outside the thinker (condition encoder, translator)
            — see ThinkerFrozenPainterBase.build_optimizers.
        cond_tokenizer: CondTokenizerBase instance or Hydra config.
        net: build_mmdit kwargs (except cond_dim, cond_rope), or a ready MMJiT.
        cond_rope: condition-stream rotary embedding; None = the tokenizer's.
        steer_from_block: blocks [0, steer_from_block) form the read-only head
            (0: the thinker reads the embedded tokens and every block is steered).
        painter_dtype: autocast dtype for the painter's forward (None = fp32 /
            whatever the caller's autocast says).
        pixel_range: range of the dataset's images, "[0,1]" or "[-1,1]".
        model_range: range the model works in, "[-1,1]" (MiniT2I) or "[0,1]"
            (the working puzzle painters); images are mapped from pixel_range.
        noise_scale, velocity_clamp: training noise std and velocity clamp;
            None = the scheduler's (they must agree with it).
        t_logit_mean, t_logit_std: logit-normal training timestep distribution.
        label_drop_rate: fraction of training samples whose condition is
            dropped (for classifier-free guidance), in both training stages.
        objective: "flow" (MiniT2I's flow matching: logit-normal t in [0, 1],
            velocity output for FlowMatchingScheduler) or "ddpm" (the DDPM
            training of the working puzzle painters: integer t uniform over the
            scheduler's steps, x_t from scheduler.add_noise, the x0 prediction
            itself as output and MSE target; needs a DDPMScheduler with
            prediction_type "sample").
        time_cond_scale: the net's timestep embedding (net.time_cond) reads
            t * time_cond_scale — 1000 for flow t in [0, 1], 1 for DDPM steps.
        loss_weighting: "velocity" (the velocity MSE, MiniT2I's objective) or
            "x0" (the x0 MSE: the velocity MSE weighted per sample by
            max(1 - t, velocity_clamp)^2, via DataSample.loss_weight). Only the
            loss changes; the prediction and sampling stay the same.
        freeze: freeze every parameter (the thinker stage).
        checkpoint: train_trm.py checkpoint of this painter to load (stage 1's).
    """

    def __init__(
        self,
        scheduler,
        train_cfg,
        eval_cfg,
        optim_cfg,
        cond_tokenizer,
        net,
        cond_rope: Optional[str] = None,
        steer_from_block: int = 1,
        painter_dtype: Optional[str] = None,
        pixel_range: str = "[-1,1]",
        model_range: str = "[-1,1]",
        noise_scale: Optional[float] = None,
        velocity_clamp: Optional[float] = None,
        t_logit_mean: float = -0.8,
        t_logit_std: float = 0.8,
        label_drop_rate: float = 0.1,
        loss_weighting: str = "velocity",
        objective: str = "flow",
        time_cond_scale: float = 1000.0,
        freeze: bool = False,
        checkpoint: Optional[str] = None,
        eval_callbacks=None,
        sampling_pipeline=None,
    ):
        super().__init__()
        self.scheduler = scheduler
        self.train_cfg = train_cfg
        self.eval_cfg = eval_cfg
        self.optim_cfg = optim_cfg
        self.eval_callbacks = [instantiate(cb) for cb in eval_callbacks] if eval_callbacks else []
        self.sampling_pipeline = instantiate(sampling_pipeline) if sampling_pipeline is not None else None

        self.cond_tokenizer = cond_tokenizer if isinstance(cond_tokenizer, nn.Module) else instantiate(cond_tokenizer)
        cond_rope = cond_rope or self.cond_tokenizer.rope
        self.net = net if isinstance(net, nn.Module) else build_mmdit(
            cond_dim=self.cond_tokenizer.out_dim, cond_rope=cond_rope, **net
        )
        self.condition_keys = list(self.cond_tokenizer.condition_keys)

        if not 0 <= steer_from_block < len(self.net.double_blocks):
            raise ValueError(f"steer_from_block must be in [0, {len(self.net.double_blocks) - 1}]")
        if pixel_range not in ("[0,1]", "[-1,1]") or model_range not in ("[0,1]", "[-1,1]"):
            raise ValueError(f"pixel_range / model_range must be '[0,1]' or '[-1,1]', got {pixel_range!r}, {model_range!r}")
        self.model_range = model_range
        self.steer_from_block = steer_from_block
        self.painter_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[painter_dtype] if painter_dtype else None
        self.pixel_range = pixel_range
        self.noise_scale = self._from_scheduler(noise_scale, "init_noise_sigma", 2.0, "noise_scale")
        self.velocity_clamp = self._from_scheduler(velocity_clamp, "train_velocity_clamp", 0.05, "velocity_clamp")
        self.t_logit_mean = t_logit_mean
        self.t_logit_std = t_logit_std
        self.label_drop_rate = label_drop_rate
        if loss_weighting not in ("velocity", "x0"):
            raise ValueError(f"loss_weighting must be 'velocity' or 'x0', got {loss_weighting!r}")
        self.loss_weighting = loss_weighting
        if objective not in ("flow", "ddpm"):
            raise ValueError(f"objective must be 'flow' or 'ddpm', got {objective!r}")
        if objective == "ddpm" and (getattr(scheduler.config, "prediction_type", None) != "sample" or loss_weighting != "velocity"):
            raise ValueError("objective 'ddpm' needs a DDPM scheduler with prediction_type 'sample' (and no loss_weighting)")
        self.objective = objective
        self.time_cond_scale = time_cond_scale

        if checkpoint is not None:
            _load_frozen_checkpoint(self, checkpoint)
        self.frozen = freeze
        if freeze:
            self.requires_grad_(False)

    def _from_scheduler(self, value, attr: str, default: float, name: str) -> float:
        sched_value = getattr(self.scheduler, attr, None)
        if value is None:
            return default if sched_value is None else sched_value
        if sched_value is not None and value != sched_value:
            raise ValueError(f"painter {name}={value} but the scheduler's {attr} is {sched_value}")
        return value

    # ── Shapes / ranges ──────────────────────────────────────────────────────

    @property
    def num_steered_blocks(self) -> int:
        return len(self.net.double_blocks) - self.steer_from_block

    @property
    def img_grid(self) -> int:
        return self.net.latent_img_size

    @property
    def noise_shape(self) -> tuple:
        s = self.net.cfg.image_size
        return (self.net.cfg.in_channels, s, s)

    def to_model_range(self, images: torch.Tensor) -> torch.Tensor:
        images = images.float()
        if self.pixel_range == self.model_range:
            return images
        return images * 2.0 - 1.0 if self.pixel_range == "[0,1]" else (images + 1.0) / 2.0

    def decode_for_eval(self, x: torch.Tensor) -> torch.Tensor:
        """Model-range output → [0, 1] for evaluation."""
        if self.model_range == "[0,1]":
            return x.float().clamp(0.0, 1.0)
        return ((x.float().clamp(-1.0, 1.0) + 1.0) / 2.0).clamp(0.0, 1.0)

    def images_to_log(self, images: torch.Tensor) -> torch.Tensor:
        return self.decode_for_eval(self.to_model_range(images))

    def _autocast(self, device: torch.device):
        if self.painter_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=device.type, dtype=self.painter_dtype)

    # ── The two halves of mmdit.MMJiT.forward ────────────────────────────────

    def _vec(self, t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """The blocks' timestep conditioning (None for nets without it, e.g. MiniT2I)."""
        if not getattr(self.net, "time_cond", False):
            return None
        if t is None:
            raise ValueError("this painter is timestep-conditioned (net.time_cond): pass t")
        return self.net.t_embedder(t.float() * self.time_cond_scale)

    def encode_head(self, x_t: torch.Tensor, *cond: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Condition tokens + embeddings + blocks [0, steer_from_block): the joint
        (B, L_cond + L_img, D) token stream, condition first (MMJiT's own order).
        cond: the fields in self.condition_keys, in order; t: the timesteps (only
        read by a timestep-conditioned net)."""
        net = self.net
        grad = torch.is_grad_enabled() and not self.frozen
        with torch.set_grad_enabled(grad), self._autocast(x_t.device):
            vec = self._vec(t)
            if self.condition_keys:
                context, mask = self.cond_tokenizer(*cond)
            else:  # NullCondTokenizer: no condition tokens
                context = x_t.new_zeros(x_t.shape[0], 0, self.cond_tokenizer.out_dim, dtype=torch.float32)
                mask = x_t.new_zeros(x_t.shape[0], 0)
            # mmdit.MMJiT.forward up to its block loop (`vec` is computed there but
            # never used by any block, so it is skipped here).
            mask = mask.to(device=context.device)
            context = torch.where(mask[:, :, None] > 0.5, context, net.mask_token.to(dtype=context.dtype))
            x = net.img_embedder(x_t)
            x = x + mmdit.get_2d_sincos_pos_embed(net.cfg.hidden_size, net.latent_img_size, x.device, x.dtype)[None]
            txt = net.txt_embedder(context.to(dtype=net.txt_embedder.weight.dtype))
            for block in net.txt_preamble_blocks:
                txt = block(txt)
            for block in net.double_blocks[: self.steer_from_block]:
                x, txt = block(x, txt, vec)
        return torch.cat([txt, x], dim=1)

    def decode_tail(
        self,
        hidden: torch.Tensor,
        block_residuals: Optional[list] = None,
        cond_block_residuals: Optional[list] = None,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Blocks [steer_from_block, depth) + output head → predicted clean image.
        block_residuals[j] is added to the image tokens after the j-th of them,
        cond_block_residuals[j] to the condition tokens (only the first
        num_steered_blocks - 1 can matter: the output head reads image tokens)."""
        net = self.net
        n_img = net.latent_img_size**2
        txt, x = hidden[:, :-n_img], hidden[:, -n_img:]
        if cond_block_residuals is not None and len(cond_block_residuals) > self.num_steered_blocks - 1:
            raise ValueError(f"{len(cond_block_residuals)} condition residuals for {self.num_steered_blocks} steered blocks")
        with self._autocast(hidden.device):
            vec = self._vec(t)
            for j, block in enumerate(net.double_blocks[self.steer_from_block :]):
                x, txt = block(x, txt, vec)
                if block_residuals is not None:
                    x = x + block_residuals[j].to(x.dtype)
                if cond_block_residuals is not None and j < len(cond_block_residuals):
                    txt = txt + cond_block_residuals[j].to(txt.dtype)
            # MMJiT applies final_layer to cat([txt, x]) and keeps the image part;
            # it acts per token, so applying it to x alone is the same.
            if vec is not None:
                shift, scale = net.final_adaln(vec).chunk(2, dim=-1)
                return net.unpatchify(net.final_layer.linear(mmdit.modulate(net.final_layer.norm_final(x), shift, scale)))
            return net.unpatchify(net.final_layer(x))

    def _cond(self, sample: DataSample) -> list:
        return [getattr(sample, k) for k in self.condition_keys]

    def fill_runtime_fields(self, sample: DataSample) -> DataSample:
        """Compute painter_hidden for the sample's x_t and condition (see
        ThinkerFrozenPainterBase.forward_with_carry)."""
        hidden = self.encode_head(sample.x_noisy, *self._cond(sample), t=sample.timesteps)
        return dataclasses.replace(sample, painter_hidden=hidden)

    def null_condition_sample(self, sample: DataSample) -> DataSample:
        """Condition fields zeroed, and painter_hidden dropped so forward recomputes
        it from the null condition — painter_hidden was computed from the real one.
        (ThinkerFrozenPainterBase's force_unconditional_painter: the thinker keeps
        reading the real sample's painter_hidden, the painter paints from null.)"""
        return dataclasses.replace(super().null_condition_sample(sample), painter_hidden=None)

    def velocity(self, x0_hat: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        scale = (1.0 - t.float()).view(-1, 1, 1, 1).clamp_min(self.velocity_clamp)
        return (x0_hat.float() - x_t.float()) / scale

    def forward(self, sample: DataSample, steering: Optional[ThinkerSteering] = None) -> DiffusionPrediction:
        hidden = sample.painter_hidden
        if hidden is None:
            hidden = self.encode_head(sample.x_noisy, *self._cond(sample), t=sample.timesteps)
        kwargs = steering.to_painter_kwargs() if steering is not None else {}
        residuals = kwargs.get("block_residuals")
        if residuals is not None and len(residuals) != self.num_steered_blocks:
            raise ValueError(
                f"{len(residuals)} steering residuals for {self.num_steered_blocks} steered blocks "
                f"(steer_from_block={self.steer_from_block}) — set translator.num_blocks to match"
            )
        x0_hat = self.decode_tail(hidden, residuals, kwargs.get("cond_block_residuals"), t=sample.timesteps)
        if self.objective == "ddpm":
            return DiffusionPrediction(pred=x0_hat.float(), pred_type="sample")
        return DiffusionPrediction(
            pred=self.velocity(x0_hat, sample.x_noisy, sample.timesteps), pred_type="v_prediction"
        )

    # ── Training ─────────────────────────────────────────────────────────────

    def _prepare_training_sample(self, mb: DataSample, device: torch.device) -> DataSample:
        """x_t, t and the velocity target, drawn as looped_dit/diffusion.py:
        training_loss (same draws, same order), with the condition dropped for
        label_drop_rate of the samples; painter_hidden for that x_t when frozen
        (the thinker stage reads it; a trainable painter computes it in forward,
        with gradients)."""
        fields = {
            f.name: getattr(mb, f.name).to(device)
            for f in dataclasses.fields(DataSample)
            if f.name not in _RUNTIME_FIELDS and isinstance(getattr(mb, f.name), torch.Tensor)
        }
        images = self.to_model_range(fields["images"])
        b = images.shape[0]
        if self.label_drop_rate > 0:
            drop = torch.rand(b, device=device) < self.label_drop_rate
            for k in self.condition_keys:
                v = fields[k]
                fields[k] = torch.where(drop.view(-1, *[1] * (v.dim() - 1)), torch.zeros_like(v), v)
        if self.objective == "ddpm":
            t = torch.randint(0, self.scheduler.config.num_train_timesteps, (b,), device=device, dtype=torch.long)
            x_t = self.scheduler.add_noise(images, torch.randn_like(images), t)
            target, weight = images, None
        else:
            t = torch.sigmoid(torch.randn(b, device=device) * self.t_logit_std + self.t_logit_mean).clamp(1e-5, 1 - 1e-5)
            t4 = t[:, None, None, None]
            noise = torch.randn_like(images) * self.noise_scale
            x_t = images * t4 + noise * (1.0 - t4)
            target = (images - x_t) / (1.0 - t4).clamp_min(self.velocity_clamp)
            weight = (1.0 - t).clamp_min(self.velocity_clamp) ** 2 if self.loss_weighting == "x0" else None
        sample = DataSample(
            **{**fields, "images": images, "x_noisy": x_t, "timesteps": t, "target": target, "loss_weight": weight}
        )
        return self.fill_runtime_fields(sample) if self.frozen else sample

    def build_optimizers(self, world_size, num_steps) -> list[ScheduledOptimizer]:
        trainable = [p for p in self.parameters() if p.requires_grad]
        oc = self.optim_cfg  # a DictConfig under _recursive_: false
        optim = torch.optim.AdamW(
            trainable,
            lr=0,
            betas=tuple(getattr(oc, "betas", (0.9, 0.999))),
            weight_decay=oc.weight_decay,
            eps=getattr(oc, "eps", 1e-8),
        )
        return [
            ScheduledOptimizer(
                optim,
                base_lr=oc.lr,
                warmup_steps=oc.warmup_steps,
                num_steps=num_steps,
                min_ratio=getattr(oc, "lr_min_ratio", 0.0),
            )
        ]

    def _loss(self, sample: DataSample) -> torch.Tensor:
        per_sample = (self(sample).pred - sample.target.float()).pow(2).flatten(1).mean(1)
        return (per_sample * sample.loss_weight).mean() if sample.loss_weight is not None else per_sample.mean()

    def train_step(self, micro_batches, accelerator, optimizers, ema, global_step, **kwargs):
        K = len(micro_batches)
        loss_sum = 0.0
        for mb in micro_batches:
            loss = self._loss(self._prepare_training_sample(mb, accelerator.device))
            accelerator.backward(loss / K)
            loss_sum += loss.item()
        params = [p for p in self.parameters() if p.requires_grad]
        all_reduce_grads(params)
        accelerator.clip_grad_norm_(params, self.train_cfg.grad_clip)
        lr = apply_lr_and_step(optimizers, global_step)
        if ema is not None:
            ema.update(self)
        return {"diff_loss": loss_sum / K}, lr, global_step + 1

    def compile_submodules(self):
        """torch.compile every transformer block in place (nn.Module.compile keeps
        parameter names, so checkpoints are unaffected); the loops around them
        and the embeddings stay eager."""
        for block in [*self.net.txt_preamble_blocks, *self.net.double_blocks]:
            block.compile(fullgraph=False, dynamic=False)

    @torch.no_grad()
    def eval_step(self, dataloader, accelerator, **kwargs) -> dict:
        max_batches = kwargs.get("max_batches", 100)
        # loss_seed: the same noise/timesteps/condition dropout at every eval (see
        # ThinkerFrozenPainterBase.eval_step); None = global RNG.
        loss_seed = kwargs.get("loss_seed")
        device = accelerator.device
        self.eval()
        losses = []
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else [], enabled=loss_seed is not None):
            if loss_seed is not None:
                torch.random.default_generator.manual_seed(loss_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed(loss_seed)
            for i, batch in tqdm(enumerate(dataloader), "Eval", total=max_batches):
                if i >= max_batches:
                    break
                losses.append(self._loss(self._prepare_training_sample(batch, device)).item())
        metrics = {"diff_loss": sum(losses) / len(losses)} if losses else {}
        for cb in self.eval_callbacks:
            metrics.update(cb(self, dataloader, accelerator, **kwargs))
        self.train()
        return metrics


# ── Thinker input ─────────────────────────────────────────────────────────────


class PainterHiddenEncoder(ConditionEncoderBase):
    """Thinker tokens from DataSample.painter_hidden ([cond; image] tokens, see
    MMDiTPainter.encode_head): the image tokens average-pooled `pool` x `pool`,
    then the condition tokens, each normalized and projected to the thinker
    width. Order [image grid; cond] — PainterResidualTranslator reads the image
    grid back from the front and the condition tokens after it.

    The projections are initialized so tokens come out at the thinker's own
    token-embedding scale (std 1/sqrt(output_dim)), which the TRM then
    multiplies by sqrt(output_dim).
    """

    condition_keys = ["painter_hidden"]

    def __init__(self, output_dim: int, painter_dim: int = 768, img_grid: int = 32, pool: int = 2):
        super().__init__()
        if img_grid % pool:
            raise ValueError(f"img_grid {img_grid} is not a multiple of pool {pool}")
        self.painter_dim = painter_dim
        self.img_grid = img_grid
        self.pool = pool
        # txt_*: the condition stream (named for MiniT2I's text, kept so
        # existing thinker checkpoints load).
        self.img_norm = nn.LayerNorm(painter_dim, elementwise_affine=False)
        self.txt_norm = nn.LayerNorm(painter_dim, elementwise_affine=False)
        self.img_proj = nn.Linear(painter_dim, output_dim)
        self.txt_proj = nn.Linear(painter_dim, output_dim)
        for proj in (self.img_proj, self.txt_proj):
            nn.init.normal_(proj.weight, std=1.0 / (painter_dim * output_dim) ** 0.5)
            nn.init.zeros_(proj.bias)

    def bind_painter(self, painter) -> None:
        net = painter.net
        if net.cfg.hidden_size != self.painter_dim or net.cfg.txt_hidden_size != self.painter_dim:
            raise ValueError(f"condition_encoder.painter_dim {self.painter_dim} != painter width {net.cfg.hidden_size}")
        if net.latent_img_size != self.img_grid:
            raise ValueError(f"condition_encoder.img_grid {self.img_grid} != painter token grid {net.latent_img_size}")

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

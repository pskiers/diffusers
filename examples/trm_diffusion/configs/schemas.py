from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ── Training loop sub-configs ──────────────────────────────────────────────────


@dataclass
class NoisySwapConfig:
    prob: float = 0.0
    t_min: int = 80
    t_max: int = 100


@dataclass
class ClassifierLossConfig:
    weight: float = 0.0
    t_max: int = 200
    target: str = "x0_pred"
    loss_type: str = "ce"
    noisy_classifier: bool = False
    classifier_path: Optional[str] = None


@dataclass
class PainterStageConfig:
    n_sup: int = -1  # -1 = use model.n_sup
    freeze_thinker: bool = False
    every: int = 1
    H_cycles: Optional[int] = None
    L_cycles: Optional[int] = None


@dataclass
class ThinkerStageConfig:
    n_sup: int = -1  # -1 = use model.n_sup
    every: int = 1


@dataclass
class TwoStageConfig:
    painter: PainterStageConfig = field(default_factory=PainterStageConfig)
    thinker: ThinkerStageConfig = field(default_factory=ThinkerStageConfig)


@dataclass
class TrainConfig:
    seed: int
    batch_size: int
    num_steps: int
    compile: bool = True
    gradient_accumulation_steps: int = 1
    sudoku_loss_weight: float = 1.0
    mse_loss_weight: float = 1.0
    cfg_prob: float = 0.0
    noisy_swap: Optional[NoisySwapConfig] = None
    classifier_loss: Optional[ClassifierLossConfig] = None
    two_stage: Optional[TwoStageConfig] = None
    # Noisy channel dropout: zeroes the x_t input to the V1 encoder with probability
    # p(t) = noisy_dropout_p_max * (1 - t/T).  Highest dropout at t=0 (clean, shortcut
    # regime), zero dropout at t=T (pure noise, x_t uninformative anyway).
    noisy_dropout_p_max: float = 0.0
    # Min-SNR loss weighting (Hang et al. 2023).  When set, per-sample MSE is multiplied
    # by min(SNR(t), gamma)/SNR(t) for x0-prediction or min(SNR(t), gamma) for
    # epsilon-prediction, upweighting high-noise steps relative to easy low-noise ones.
    minsnr_gamma: Optional[float] = None


@dataclass
class EvalConfig:
    eval_every: int
    save_every: int
    log_every: int
    cfg_scale: float = 1.0
    num_samples: int = 7680
    batch_size: int = 384
    num_ddim_steps: int = 20
    num_log_images: int = 10
    classifier_path: Optional[str] = None


# ── Optimizer configs ──────────────────────────────────────────────────────────


@dataclass
class PuzzleEmbOptimConfig:
    lr: float
    weight_decay: float
    warmup_steps: int
    lr_min_ratio: float = 0.0


@dataclass
class ThinkerOptimConfig:
    lr: float
    weight_decay: float
    beta1: float
    beta2: float
    warmup_steps: int
    lr_min_ratio: float = 0.0
    puzzle_emb: Optional[PuzzleEmbOptimConfig] = None


@dataclass
class PainterOptimConfig:
    lr: float
    weight_decay: float
    warmup_steps: int
    lr_min_ratio: float = 0.0


# ── Latent DiT configs ────────────────────────────────────────────────────────


@dataclass
class LatentDiTConfig:
    vae_checkpoint: str  # path to trained VAE .pt checkpoint
    latent_channels: int = 4
    latent_size: int = 36  # spatial size after VAE encoding (painter_size // 4)
    patch_size: int = 4  # patch_size=4 on 36×36 → 81 patches (one per cell)
    num_attention_heads: int = 8
    attention_head_dim: int = 64
    num_layers: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    vocab_size: int = 11  # 0=null, 1=blank, 2-10=digits 1-9
    cond_embed_dim: int = 256
    cell_size: int = 16
    painter_size: int = 144


@dataclass
class LatentDiTOptimConfig:
    lr: float = 1e-4
    weight_decay: float = 0.0
    warmup_steps: int = 1000
    lr_min_ratio: float = 0.1


# ── Model architecture configs ─────────────────────────────────────────────────


@dataclass
class ImageEncoderConfig:
    enc_channels: int = 32
    enc_hidden_channels: tuple = field(default_factory=lambda: (16, 32))
    thinker_out_channels: Optional[int] = None  # if set and != vocab_size, adds logit_expand


@dataclass
class TimestepCondConfig:
    enc_timestep_cond: bool = False
    thinker_timestep_cond: bool = False
    decoder_timestep_cond: bool = False
    temb_dim: int = 256


@dataclass
class PainterConfig:
    vocab_size: int = 11
    painter_size: int = 144
    cell_size: int = 16
    bridge_channels: int = 16
    painter_channels: tuple = field(default_factory=lambda: (32, 64, 64))
    painter_layers_per_block: int = 2
    painter_dtype: Optional[str] = None


@dataclass
class PainterThinkerConfig:
    painter_size: int = 144
    cell_size: int = 16
    bridge_channels: int = 16
    painter_channels: tuple = field(default_factory=lambda: (32, 64, 128))
    painter_layers_per_block: int = 1
    diff_thinker_weight: float = 1.0
    thinker_bridge_mode: str = "logits"
    painter_dtype: Optional[str] = None
    # If set, TRM operates on a thinker_grid_size×thinker_grid_size grid instead of
    # painter_size//cell_size.  Encoder output is adaptively pooled to this size.
    thinker_grid_size: Optional[int] = None


@dataclass
class ThinkerModelConfig:
    vocab_size: int
    seq_len: int
    hidden_size: int
    n_heads: int
    L_layers: int
    L_cycles: int
    H_cycles: int
    n_sup: int
    batch_size: int
    forward_dtype: str
    expansion: float = 4.0
    pos_encodings: str = "rope"
    mlp_t: bool = False
    puzzle_emb_ndim: int = 0
    puzzle_emb_len: int = 16
    num_puzzle_identifiers: int = 1
    halt_exploration_prob: float = 0.0
    freeze_weights: bool = False

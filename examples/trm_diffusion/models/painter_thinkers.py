from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.trm_wrappers import OriginalTRMSudoku
from models.utility_models import SpatialEncoder, AttentiveBridge, SpatialBridge, TimestepMLP
from models.painters import make_painter, StandalonePainter


class OriginalTRMRatatouilleV0Tok(nn.Module):
    """
    Painter-thinker model using the original TRM as the thinker (token input).

    Thinker: OriginalTRMSudoku(vocab_size) — receives puzzle tokens directly,
             outputs (B, 81, vocab_size) logits over sudoku token IDs.
    Bridge:  SpatialBridge — bilinear upsample + 2 convs:
             (B, vocab_size, 9, 9) → (B, bridge_channels, painter_size, painter_size).
    Painter: UNet2DModel — denoises cat([noisy, bridge_feat], dim=1).

    The thinker logits are reshaped to (B, vocab_size, 9, 9) before the bridge:
      logits (B,81,V) → transpose → (B,V,81) → reshape → (B,V,9,9)

    token_input=True tells train_trm.py to use batch["puzzle_tokens"] as
    the condition, not batch["conditions"].

    Sudoku CE loss: labels must be in token format (2-10 for correct digit, -100
    for ignored). train_trm.py converts solution (0-8 digit classes) → (2-10)
    automatically.  See train_trm.py for details.

    diff_thinker_weight: scale of diffusion-loss gradient back into the thinker.
      0.0  → thinker trained only on sudoku CE loss (bridge sees detached spatial).
      1.0  → full diffusion gradient reaches thinker.
    """

    token_input: bool = True
    has_realsolution_eval: bool = True   # eval with full solution tokens as condition

    def __init__(
        self,
        # --- painter geometry ---
        painter_size: int = 144,
        cell_size: int = 16,
        # --- thinker ---
        vocab_size: int = 11,
        seq_len: int = 81,
        hidden_size: int = 512,
        n_heads: int = 8,
        L_layers: int = 2,
        L_cycles: int = 6,
        H_cycles: int = 3,
        n_sup: int = 16,
        expansion: float = 4.0,
        forward_dtype: str = "bfloat16",
        mlp_t: bool = False,
        pos_encodings: str = "rope",
        puzzle_emb_ndim: int = 0,
        puzzle_emb_len: int = 16,
        num_puzzle_identifiers: int = 1,
        halt_exploration_prob: float = 0.0,
        batch_size: int = 1,
        freeze_weights: bool = False,
        # --- bridge & painter ---
        bridge_channels: int = 16,
        painter_channels: tuple = (32, 64, 128),
        painter_layers_per_block: int = 1,
        diff_thinker_weight: float = 1.0,
        # How thinker logits are converted to spatial conditioning at inference time.
        #   "logits"  – raw logits (default, matches training with raw logit spatial)
        #   "onehot"  – argmax → one-hot (matches painter trained on real one-hot solutions)
        #   "softmax" – softmax probabilities (soft version of onehot)
        thinker_bridge_mode: str = "logits",
        # Classifier-free guidance.  cfg_prob > 0 randomly zeros the spatial
        # conditioning during training.  cfg_scale > 1.0 at inference enables
        # the double-forward CFG pass in forward().
        cfg_prob: float = 0.0,
        cfg_scale: float = 1.0,
        # Autocast dtype for the bridge + painter UNet.  None = no autocast
        # (painter runs in whatever dtype the tensors arrive in).
        # "bfloat16" is the safe default: same exponent range as float32, so no
        # GradScaler needed.  "float16" also works but requires a GradScaler
        # (use accelerate mixed_precision="fp16" which handles this automatically).
        # The TRM thinker is NOT affected — it manages its own dtype via forward_dtype.
        painter_dtype: Optional[str] = None,
    ):
        super().__init__()
        self.diff_thinker_weight = diff_thinker_weight
        self.thinker_bridge_mode = thinker_bridge_mode
        self.cfg_prob  = cfg_prob
        self.cfg_scale = cfg_scale
        self._grid = painter_size // cell_size   # e.g. 144//16 = 9
        # Thinker vocab uses 0=PAD, 1=blank, 2-10=digits 1-9.
        # sample_grids compares argmax predictions against raw solution labels (0-8),
        # so it needs to know to shift by this offset when comparing.
        self.token_offset = 2
        self._painter_dtype: Optional[torch.dtype] = (
            {"bfloat16": torch.bfloat16, "float16": torch.float16}[painter_dtype]
            if painter_dtype is not None else None
        )

        self.thinker = OriginalTRMSudoku(
            vocab_size=vocab_size,
            seq_len=seq_len,
            hidden_size=hidden_size,
            n_heads=n_heads,
            L_layers=L_layers,
            L_cycles=L_cycles,
            H_cycles=H_cycles,
            n_sup=n_sup,
            expansion=expansion,
            forward_dtype=forward_dtype,
            mlp_t=mlp_t,
            pos_encodings=pos_encodings,
            puzzle_emb_ndim=puzzle_emb_ndim,
            puzzle_emb_len=puzzle_emb_len,
            num_puzzle_identifiers=num_puzzle_identifiers,
            halt_exploration_prob=halt_exploration_prob,
            batch_size=batch_size,
            freeze_weights=freeze_weights,
        )
        self.bridge = SpatialBridge(
            in_channels=vocab_size,
            out_channels=bridge_channels,
            painter_size=painter_size,
        )
        self.painter = make_painter(
            painter_size=painter_size,
            bridge_channels=bridge_channels,
            painter_channels=tuple(painter_channels),
            layers_per_block=painter_layers_per_block,
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

    def _logits_to_spatial(self, logits: torch.Tensor) -> torch.Tensor:
        """(B, N, C) logits → (B, C, grid, grid) spatial conditioning.

        Conversion respects self.thinker_bridge_mode:
          "logits"  – raw float logits (default)
          "onehot"  – argmax → one-hot
          "softmax" – softmax probabilities
        """
        B, _, C = logits.shape
        mode = getattr(self, "thinker_bridge_mode", "logits")
        if mode == "onehot":
            # Straight-through estimator: hard one-hot in the forward pass,
            # softmax gradient in the backward pass so thinker weights can update.
            soft   = logits.float().softmax(dim=-1)
            hard   = F.one_hot(logits.argmax(dim=-1), num_classes=C).float()
            onehot = hard - soft.detach() + soft   # forward≈hard, grad flows via soft
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
                self.diff_thinker_weight * spatial_cond
                + (1.0 - self.diff_thinker_weight) * spatial_cond.detach()
            )
        else:
            sc_for_painter = spatial_cond

        # CFG training dropout: randomly zero conditioning per sample.
        if self.training and self.cfg_prob > 0:
            drop = torch.rand(sc_for_painter.shape[0], 1, 1, 1, device=sc_for_painter.device) < self.cfg_prob
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
            logits, z_H, z_L = self.thinker.reasoning_step(
                puzzle_tokens, z_H, z_L, puzzle_ids
            )

        spatial_cond = self._logits_to_spatial(logits.float())

        if not self.training and self.cfg_scale > 1.0:
            null = torch.zeros_like(spatial_cond)
            pred_cond   = self._run_painter(noisy, spatial_cond, timesteps)
            pred_uncond = self._run_painter(noisy, null, timesteps)
            noise_pred  = pred_uncond + self.cfg_scale * (pred_cond - pred_uncond)
        else:
            noise_pred = self._run_painter(noisy, spatial_cond, timesteps)
        return noise_pred, logits


# ── Painter-thinker (V0: image-conditioned) ───────────────────────────────────

class OriginalTRMRatatouilleV0(OriginalTRMRatatouilleV0Tok):
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
    has_realsolution_eval: bool = True   # eval with full MNIST image as condition

    def __init__(
        self,
        # --- painter geometry ---
        painter_size: int = 144,
        cell_size: int = 16,
        # --- thinker ---
        num_classes: int = 9,
        seq_len: int = 81,
        hidden_size: int = 512,
        n_heads: int = 8,
        L_layers: int = 2,
        L_cycles: int = 6,
        H_cycles: int = 3,
        n_sup: int = 16,
        expansion: float = 4.0,
        forward_dtype: str = "bfloat16",
        mlp_t: bool = False,
        pos_encodings: str = "rope",
        halt_exploration_prob: float = 0.0,
        batch_size: int = 1,
        freeze_weights: bool = False,
        # --- image encoder ---
        enc_channels: int = 32,
        enc_hidden_channels: tuple = (16, 32),  # intermediate widths in SpatialEncoder
        # --- bridge & painter ---
        thinker_out_channels: int = None,   # if > num_classes, expands logits before bridge
        bridge_channels: int = 16,
        painter_channels: tuple = (32, 64, 128),
        painter_layers_per_block: int = 1,
        diff_thinker_weight: float = 1.0,
        thinker_bridge_mode: str = "logits",
        cfg_prob: float = 0.0,
        cfg_scale: float = 1.0,
        painter_dtype: Optional[str] = None,
    ):
        _toc = thinker_out_channels if thinker_out_channels is not None else num_classes
        super().__init__(
            painter_size=painter_size,
            cell_size=cell_size,
            vocab_size=num_classes,
            seq_len=seq_len,
            hidden_size=hidden_size,
            n_heads=n_heads,
            L_layers=L_layers,
            L_cycles=L_cycles,
            H_cycles=H_cycles,
            n_sup=n_sup,
            expansion=expansion,
            forward_dtype=forward_dtype,
            mlp_t=mlp_t,
            pos_encodings=pos_encodings,
            puzzle_emb_ndim=0,
            halt_exploration_prob=halt_exploration_prob,
            batch_size=batch_size,
            freeze_weights=freeze_weights,
            bridge_channels=bridge_channels,
            painter_channels=painter_channels,
            painter_layers_per_block=painter_layers_per_block,
            diff_thinker_weight=diff_thinker_weight,
            thinker_bridge_mode=thinker_bridge_mode,
            cfg_prob=cfg_prob,
            cfg_scale=cfg_scale,
            painter_dtype=painter_dtype,
        )
        self.token_offset = 0

        # condition image (B,1,H,W) → (B, enc_channels, grid, grid)
        self.image_encoder = SpatialEncoder(1, enc_channels, factor=cell_size,
                                            hidden_channels=tuple(enc_hidden_channels))
        # project enc_channels → hidden_size per cell
        std = 1.0 / (math.sqrt(hidden_size) * math.sqrt(enc_channels))
        self.enc_proj = nn.Conv2d(enc_channels, hidden_size, 1)
        nn.init.normal_(self.enc_proj.weight, std=std)
        nn.init.zeros_(self.enc_proj.bias)

        # Optional expansion: project num_classes → thinker_out_channels before bridge.
        # CE loss still uses raw num_classes logits; only the bridge sees the expanded map.
        if _toc != num_classes:
            self.logit_expand = nn.Linear(num_classes, _toc, bias=False)
            self.bridge = SpatialBridge(
                in_channels=_toc,
                out_channels=bridge_channels,
                painter_size=painter_size,
            )
        else:
            self.logit_expand = None

    def _logits_to_spatial(self, logits: torch.Tensor) -> torch.Tensor:
        if self.logit_expand is not None:
            logits = self.logit_expand(logits.float())
        return super()._logits_to_spatial(logits)

    def _encode_image(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → float embeddings (B, 81, hidden_size)"""
        feat = self.image_encoder(x)               # (B, enc_channels, grid, grid)
        proj = self.enc_proj(feat)                 # (B, hidden_size, grid, grid)
        return proj.flatten(2).transpose(1, 2)     # (B, 81, hidden_size)

    def _get_enc_emb(
        self, condition: torch.Tensor, noisy: torch.Tensor,
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
            enc_emb, noisy, z_H, z_L, timesteps,
            puzzle_ids=puzzle_ids, H_cycles=H_cycles, L_cycles=L_cycles,
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
            logits, z_H, z_L = self.thinker.reasoning_step(
                enc_emb, z_H, z_L, puzzle_ids
            )

        spatial_cond = self._logits_to_spatial(logits.float())
        if not self.training and self.cfg_scale > 1.0:
            null = torch.zeros_like(spatial_cond)
            pred_cond   = self._run_painter(noisy, spatial_cond, timesteps)
            pred_uncond = self._run_painter(noisy, null, timesteps)
            noise_pred  = pred_uncond + self.cfg_scale * (pred_cond - pred_uncond)
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
        # --- painter geometry ---
        painter_size: int = 144,
        cell_size: int = 16,
        # --- thinker ---
        num_classes: int = 9,
        seq_len: int = 81,
        hidden_size: int = 512,
        n_heads: int = 8,
        L_layers: int = 2,
        L_cycles: int = 6,
        H_cycles: int = 3,
        n_sup: int = 16,
        expansion: float = 4.0,
        forward_dtype: str = "bfloat16",
        mlp_t: bool = False,
        pos_encodings: str = "rope",
        halt_exploration_prob: float = 0.0,
        batch_size: int = 1,
        freeze_weights: bool = False,
        # --- image encoder ---
        enc_channels: int = 32,
        enc_hidden_channels: tuple = (16, 32),
        thinker_out_channels: int = None,
        # --- timestep conditioning (V1-specific) ---
        enc_timestep_cond: bool = False,     # FiLM scale+shift on encoder features
        thinker_timestep_cond: bool = False, # T2: broadcast temb added to thinker input tokens
        temb_dim: int = 256,                 # output dim of the shared TimestepMLP
        # --- bridge & painter ---
        bridge_channels: int = 16,
        painter_channels: tuple = (32, 64, 128),
        painter_layers_per_block: int = 1,
        diff_thinker_weight: float = 1.0,
        thinker_bridge_mode: str = "logits",
        cfg_prob: float = 0.0,
        cfg_scale: float = 1.0,
        painter_dtype: Optional[str] = None,
    ):
        super().__init__(
            painter_size=painter_size,
            cell_size=cell_size,
            num_classes=num_classes,
            thinker_out_channels=thinker_out_channels,
            enc_hidden_channels=enc_hidden_channels,
            seq_len=seq_len,
            hidden_size=hidden_size,
            n_heads=n_heads,
            L_layers=L_layers,
            L_cycles=L_cycles,
            H_cycles=H_cycles,
            n_sup=n_sup,
            expansion=expansion,
            forward_dtype=forward_dtype,
            mlp_t=mlp_t,
            pos_encodings=pos_encodings,
            halt_exploration_prob=halt_exploration_prob,
            batch_size=batch_size,
            freeze_weights=freeze_weights,
            enc_channels=enc_channels,
            bridge_channels=bridge_channels,
            painter_channels=painter_channels,
            painter_layers_per_block=painter_layers_per_block,
            diff_thinker_weight=diff_thinker_weight,
            thinker_bridge_mode=thinker_bridge_mode,
            cfg_prob=cfg_prob,
            cfg_scale=cfg_scale,
            painter_dtype=painter_dtype,
        )
        # Replace 1-channel encoder with 2-channel (condition + noisy)
        self.image_encoder = SpatialEncoder(2, enc_channels, factor=cell_size,
                                            hidden_channels=tuple(enc_hidden_channels))

        # Timestep conditioning.  Both projections are zero-init so the model
        # starts as the no-timestep identity and gradually learns to use t.
        self.enc_timestep_cond     = enc_timestep_cond
        self.thinker_timestep_cond = thinker_timestep_cond
        if enc_timestep_cond or thinker_timestep_cond:
            self.timestep_mlp = TimestepMLP(sin_dim=128, out_dim=temb_dim)
        if enc_timestep_cond:
            self.enc_film = nn.Linear(temb_dim, 2 * enc_channels)
            nn.init.zeros_(self.enc_film.weight)
            nn.init.zeros_(self.enc_film.bias)
        if thinker_timestep_cond:
            self.thinker_temb_proj = nn.Linear(temb_dim, hidden_size)
            nn.init.zeros_(self.thinker_temb_proj.weight)
            nn.init.zeros_(self.thinker_temb_proj.bias)

    def _get_enc_emb(
        self, condition: torch.Tensor, noisy: torch.Tensor,
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
        proj    = self.enc_proj(feat)
        enc_emb = proj.flatten(2).transpose(1, 2)          # (B, 81, hidden_size)

        # T2: broadcast timestep embedding into thinker token space.
        if temb is not None and self.thinker_timestep_cond:
            enc_emb = enc_emb + self.thinker_temb_proj(temb).unsqueeze(1)

        return enc_emb


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

    has_realsolution_eval: bool = False   # latent thinker; no digit-level solution eval

    def __init__(
        self,
        # --- painter geometry ---
        painter_size: int = 144,
        cell_size: int = 16,
        # --- thinker ---
        thinker_out_channels: int = 16,
        seq_len: int = 81,
        hidden_size: int = 512,
        n_heads: int = 8,
        L_layers: int = 2,
        L_cycles: int = 6,
        H_cycles: int = 3,
        n_sup: int = 16,
        expansion: float = 4.0,
        forward_dtype: str = "bfloat16",
        mlp_t: bool = False,
        pos_encodings: str = "rope",
        halt_exploration_prob: float = 0.0,
        batch_size: int = 1,
        freeze_weights: bool = False,
        # --- image encoder ---
        enc_channels: int = 32,
        enc_hidden_channels: tuple = (16, 32),
        # --- bridge & painter ---
        bridge_channels: int = 16,
        painter_channels: tuple = (32, 64, 128),
        painter_layers_per_block: int = 1,
        diff_thinker_weight: float = 1.0,
        thinker_bridge_mode: str = "logits",
        cfg_prob: float = 0.0,
        cfg_scale: float = 1.0,
        painter_dtype: Optional[str] = None,
    ):
        super().__init__(
            painter_size=painter_size,
            cell_size=cell_size,
            num_classes=thinker_out_channels,
            seq_len=seq_len,
            hidden_size=hidden_size,
            n_heads=n_heads,
            L_layers=L_layers,
            L_cycles=L_cycles,
            H_cycles=H_cycles,
            n_sup=n_sup,
            expansion=expansion,
            forward_dtype=forward_dtype,
            mlp_t=mlp_t,
            pos_encodings=pos_encodings,
            halt_exploration_prob=halt_exploration_prob,
            batch_size=batch_size,
            freeze_weights=freeze_weights,
            enc_channels=enc_channels,
            enc_hidden_channels=enc_hidden_channels,
            bridge_channels=bridge_channels,
            painter_channels=painter_channels,
            painter_layers_per_block=painter_layers_per_block,
            diff_thinker_weight=diff_thinker_weight,
            thinker_bridge_mode=thinker_bridge_mode,
            cfg_prob=cfg_prob,
            cfg_scale=cfg_scale,
            painter_dtype=painter_dtype,
        )

    def reasoning_step(self, condition, noisy, z_H, z_L, timesteps,
                       puzzle_ids=None, H_cycles=None, L_cycles=None):
        noise_pred, _logits, z_H_next, z_L_next = super().reasoning_step(
            condition, noisy, z_H, z_L, timesteps,
            puzzle_ids=puzzle_ids, H_cycles=H_cycles, L_cycles=L_cycles,
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

    def __init__(self, thinker_out_channels: int = 64, **kwargs):
        super().__init__(thinker_out_channels=thinker_out_channels, **kwargs)


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
        # --- painter geometry ---
        painter_size: int = 144,
        cell_size: int = 16,            # only for _run_painter noisy input shape
        compression_factor: int = 16,   # encoder + thinker grid factor
        # --- thinker ---
        thinker_out_channels: int = 64,
        hidden_size: int = 512,
        n_heads: int = 8,
        L_layers: int = 2,
        L_cycles: int = 6,
        H_cycles: int = 3,
        n_sup: int = 16,
        expansion: float = 4.0,
        forward_dtype: str = "bfloat16",
        mlp_t: bool = False,
        pos_encodings: str = "rope",
        halt_exploration_prob: float = 0.0,
        batch_size: int = 1,
        freeze_weights: bool = False,
        # --- image encoder ---
        enc_channels: int = 32,
        enc_hidden_channels: tuple = (16, 32),
        # --- bridge & painter ---
        bridge_channels: int = 16,
        bridge_num_heads: int = 4,
        painter_channels: tuple = (32, 64, 128),
        painter_layers_per_block: int = 1,
        diff_thinker_weight: float = 1.0,
        thinker_bridge_mode: str = "logits",
        cfg_prob: float = 0.0,
        cfg_scale: float = 1.0,
        painter_dtype: Optional[str] = None,
    ):
        grid_size = painter_size // compression_factor
        seq_len   = grid_size * grid_size

        super().__init__(
            painter_size=painter_size,
            cell_size=cell_size,
            thinker_out_channels=thinker_out_channels,
            seq_len=seq_len,
            hidden_size=hidden_size,
            n_heads=n_heads,
            L_layers=L_layers,
            L_cycles=L_cycles,
            H_cycles=H_cycles,
            n_sup=n_sup,
            expansion=expansion,
            forward_dtype=forward_dtype,
            mlp_t=mlp_t,
            pos_encodings=pos_encodings,
            halt_exploration_prob=halt_exploration_prob,
            batch_size=batch_size,
            freeze_weights=freeze_weights,
            enc_channels=enc_channels,
            enc_hidden_channels=enc_hidden_channels,
            bridge_channels=bridge_channels,
            painter_channels=painter_channels,
            painter_layers_per_block=painter_layers_per_block,
            diff_thinker_weight=diff_thinker_weight,
            thinker_bridge_mode=thinker_bridge_mode,
            cfg_prob=cfg_prob,
            cfg_scale=cfg_scale,
            painter_dtype=painter_dtype,
        )

        # Thinker grid is compression_factor-based, not cell_size-based
        self._grid = grid_size

        # Replace 1→2 channel encoder (set by V1) with compression_factor version
        self.image_encoder = SpatialEncoder(2, enc_channels, factor=compression_factor,
                                            hidden_channels=tuple(enc_hidden_channels))

        # Replace SpatialBridge with AttentiveBridge
        self.bridge = AttentiveBridge(
            in_channels=thinker_out_channels,
            out_channels=bridge_channels,
            out_resolution=painter_size,
            factor=compression_factor,
            num_heads=bridge_num_heads,
        )


# ── Thinker with frozen painter ───────────────────────────────────────────────

class ThinkerWithFrozenPainter(OriginalTRMRatatouilleV0Tok):
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

    def __init__(self, painter: StandalonePainter, adapter_in_channels: int = 0, **thinker_kwargs):
        super().__init__(**thinker_kwargs)
        # Replace the freshly-built bridge+painter with the pretrained frozen ones.
        self.bridge  = painter.bridge
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
        native_in = painter.bridge.conv[0].in_channels          # vocab_size the bridge was trained with
        self.logit_projection   = None
        self.bridge_input_conv  = None
        if adapter_in_channels > 0 and adapter_in_channels != native_in:
            bridge_channels = painter.bridge.conv[0].out_channels
            # Per-cell linear projection on thinker logits: (B,81,vocab) → (B,81,adapter_in)
            self.logit_projection  = nn.Linear(self.vocab_size, adapter_in_channels)
            # Replace first bridge conv; second conv (bridge_ch→bridge_ch) stays frozen.
            self.bridge_input_conv = nn.Conv2d(adapter_in_channels, bridge_channels, kernel_size=3, padding=1)

    def _logits_to_spatial(self, logits: torch.Tensor) -> torch.Tensor:
        if self.logit_projection is not None:
            logits = self.logit_projection(logits)   # (B, 81, adapter_in_channels)
        return super()._logits_to_spatial(logits)

    def _run_painter(self, noisy, spatial_cond, timesteps):
        if self.bridge_input_conv is not None:
            # Apply new trainable first conv, then the frozen second conv.
            spatial_cond = F.interpolate(spatial_cond, size=self.bridge.painter_size,
                                         mode="bilinear", align_corners=False)
            spatial_cond = torch.nn.functional.silu(self.bridge_input_conv(spatial_cond))
            bridge_feat  = self.bridge.conv[2](spatial_cond)
            return self.painter(torch.cat([noisy, bridge_feat], dim=1), timesteps).sample
        return super()._run_painter(noisy, spatial_cond, timesteps)

    def get_painter_params(self) -> list:
        return []  # frozen — excluded from all optimizers

    def get_thinker_params(self) -> list:
        params = super().get_thinker_params()
        if self.logit_projection is not None:
            params = params + list(self.logit_projection.parameters())
        if self.bridge_input_conv is not None:
            params = params + list(self.bridge_input_conv.parameters())
        return params

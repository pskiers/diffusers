"""
Embedded-TRM Bit-Diffusion for Sokoban board generation.

A standard DiT in which EACH transformer layer is augmented with a TRM recursion block
that refines (changes) that layer's output.

  1. DEEP SUPERVISION: decode an x0 prediction after every layer and apply the diffusion
     loss to each (final weight 1.0, earlier weight aux_loss_weight). The DiT layers act as
     TRM supervision steps: every step is trained to produce a better answer.
  2. INPUT GROUNDING: the patchified noisy image is re-injected at every z-update
     (z = norm_z(f_z(x + inj + y + z))), exactly like TRM's constant input injection.
  3. PARAMETER SHARING: each block is reused n_inner*T times per step (huge effective depth
     at tiny param count). Two OPTIONAL sharing axes:
     - shared_stack - z and y reuse one block stack, like original TRM's single L_level.
     - weight_tied - one whole DiT+TRM layer reused for every step, best only at higher depth
       on tiny data where layer specialization would overfit.
  4. fp32 CARRY + bf16 COMPUTE: the (y, z) recurrent state is kept in fp32 between iterations
     while the heavy block matmuls run in the autocast dtype. Use bf16-mixed (NOT fp16).

Architecture:
    board 12x12x3 -> patchify -> inj (144 x D), grounding; y = inj, z = z_init
    for each of num_layers layers:
        x = DiT_layer(y, timestep, class)            # conditioned injection (AdaLN-Zero, residual)
        y_trm, z = TRM_block(x, inj, y, z)           # plain-transformer TRM recursion:
            N-loop : z = norm_z(f_z(x + inj + y + z))   (n_inner iterations)
            y-step : y_trm = norm_y(f_y(y + z))
            T-loop : T-1 no-grad iterations + 1 with-grad iteration (1-step gradient)
        y = x + y_trm                                # residual refine (refine_residual=True):
                                                     # keeps a DiT highway across layers instead
                                                     # of overwriting the stream every step
        x0_step = head(y)                            # deep supervision target
    prediction = x0_step of the last layer

The TRM carry (y, z) is preserved across layers WITH gradient.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import hydra
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf

from diffusers import DDPMScheduler
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.embeddings import CombinedTimestepLabelEmbeddings, PatchEmbed
from sokoban.bit_diffusion.train_std import EMACallback, SokobanBitDataModule, SokobanBitDiffusion


# Plain-transformer TRM recursion (matches original TRM blocks exactly except for the input grounding inj)
class TRMRefiner(nn.Module):
    """TRM recursion over plain transformer blocks, threading a (y, z) carry.

    States (B, L, D):
      - y : high-level "answer" (also the DiT residual stream)
      - z : low-level "scratch"
    Blocks are diffusers ``BasicTransformerBlock(norm_type="layer_norm")`` -> plain self-attn
    + FFN, NO AdaLN/timestep conditioning (exactly like the original TRM reasoning blocks; the
    diffusion timestep/class is injected only by the surrounding DiT layer via ``x``).
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        ffn_mult: int,
        activation_fn: str,
        dropout: float,
        n_inner: int,
        T: int,
        num_inner_layers: int = 1,
        shared_stack: bool = False, # try the same or different networks for the z and y
        isolate_transform: bool = False, # add only the block's own transform (out - in) to the carry
    ) -> None:
        super().__init__()
        self.n_inner = n_inner
        self.T = T
        self.isolate_transform = isolate_transform

        def _make_stack() -> nn.ModuleList:
            return nn.ModuleList([
                BasicTransformerBlock(
                    dim=dim,
                    num_attention_heads=num_heads,
                    attention_head_dim=head_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    norm_type="layer_norm",       # plain block: no AdaLN, no conditioning
                    cross_attention_dim=None,
                    ff_inner_dim=dim * ffn_mult,
                )
                for _ in range(num_inner_layers)
            ])

        self.z_layers = _make_stack()
        self.y_layers = self.z_layers if shared_stack else _make_stack()
        self.norm_z = nn.LayerNorm(dim)
        self.norm_y = nn.LayerNorm(dim)

    @staticmethod
    def _run(layers: nn.ModuleList, h: torch.Tensor) -> torch.Tensor:
        for layer in layers:
            h = layer(h)
        return h

    def _recur(
        self, x: torch.Tensor, inj: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One latent recursion: n_inner z-updates (N-loop) then one y-update.

        The patchified noisy image is re-added at every z-update for constant input
        grounding (prevents drift).

        isolate_transform toggles how each block output updates the carry:
          - False (default): the carry is REPLACED by the (normed) block output. The block has
            its own internal residual, so the previous carry survives through it once, but the
            context terms (x, inj, y) are re-summed into the carry on every iteration.
          - True: only the block's own transformation (out - in) is isolated and ADDED to the
            carry (z = norm_z(z + (out - in))), keeping x/inj/y as pure context that is not
            re-accumulated into the state. This is the canonical TRM residual update.
        """
        for _ in range(self.n_inner):
            h_in_z = x + inj + y + z
            h_out_z = self._run(self.z_layers, h_in_z)
            z_new = z + (h_out_z - h_in_z) if self.isolate_transform else h_out_z
            z = self.norm_z(z_new).float()  # float keeps the carry in fp32 across iterations
        h_in_y = y + z
        h_out_y = self._run(self.y_layers, h_in_y)
        y_new = y + (h_out_y - h_in_y) if self.isolate_transform else h_out_y
        y = self.norm_y(y_new).float()
        return y, z

    def forward(
        self, x: torch.Tensor, inj: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Deep recursion: T-1 no-grad iterations + 1 with-grad (1-step gradient).

        T-1 warm-up iterations build NO autograd graph at all. Only the final recur is differentiated, so each layer's backward graph stays shallow (~n_inner+1 block applications) regardless of T.

        The carry is deliberately NOT detached between layers (full cross-
        layer BPTT) - fine at the small num_layers used here.
        """
        with torch.no_grad():
            for _ in range(self.T - 1):
                y, z = self._recur(x, inj, y, z)
        y, z = self._recur(x, inj, y, z)
        return y, z


class TRMDiTLayer(nn.Module):
    """Standard conditioned DiT layer whose output is refined by a TRM recursion. It is a DiT block (AdaLN-Zero, conditioned) followed by a plain-transformer TRM refinement.

    - The DiT block applies the timestep/class conditioning and produces the per-layer
    injection ``x`` for the TRM block.
    - The TRM block then refines the running answer ``y`` (= the DiT residual
    stream) using ``x``, carrying the scratch ``z`` forward to the next layer.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        ffn_mult: int,
        activation_fn: str,
        dropout: float,
        num_embeds_ada_norm: int,
        n_inner: int,
        T: int,
        num_inner_layers: int = 1,
        shared_stack: bool = False,
        refine_residual: bool = True,
        isolate_transform: bool = False,
    ) -> None:
        super().__init__()
        self.refine_residual = refine_residual
        self.dit = BasicTransformerBlock(
            dim=dim,
            num_attention_heads=num_heads,
            attention_head_dim=head_dim,
            dropout=dropout,
            activation_fn=activation_fn,
            num_embeds_ada_norm=num_embeds_ada_norm,
            attention_bias=True,
            norm_type="ada_norm_zero",
            ff_inner_dim=dim * ffn_mult,
        )
        self.trm = TRMRefiner(
            dim, num_heads, head_dim, ffn_mult, activation_fn, dropout,
            n_inner, T, num_inner_layers, shared_stack, isolate_transform,
        )

    def forward(
        self, y: torch.Tensor, z: torch.Tensor, inj: torch.Tensor,
        timestep: torch.Tensor, class_labels: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.dit(y, timestep=timestep, class_labels=class_labels)  # conditioned injection (residual)
        y_trm, z = self.trm(x, inj, y, z)                              # TRM refines, grounded by inj
        # Residual refine: ADD the TRM correction to the DiT output instead of overwriting the
        # stream.
        y = x + y_trm if self.refine_residual else y_trm
        return y, z


# Full denoiser: standard DiT (patchify -> [DiT layer + TRM refine] x N -> head)
class EmbeddedTRMDiffusion(nn.Module):
    """Standard DiT denoiser with a plain-transformer TRM recursion inside every layer.
    The TRM carry (y, z) is created once and threaded through all layers.
    """

    def __init__(
        self,
        resolution: int,
        in_channels: int,
        out_channels: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_embeds_ada_norm: int,
        num_layers: int = 2,
        n_inner: int = 6,
        T: int = 3,
        num_inner_layers: int = 1,
        shared_stack: bool = False,
        weight_tied: bool = False,
        refine_residual: bool = True,
        isolate_transform: bool = False,
        ffn_mult: int = 4,
        activation_fn: str = "gelu-approximate",
        dropout: float = 0.0,
        patch_size: int = 1,
        use_grid_pos_embed: bool = True,
    ) -> None:
        super().__init__()
        dim = num_attention_heads * attention_head_dim
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.h_p = resolution // patch_size
        self.w_p = resolution // patch_size
        self.num_layers = num_layers
        self.weight_tied = weight_tied
        seq_len = self.h_p * self.w_p

        # Patchify (Conv2d + fixed 2D sincos positional embedding).
        self.patch = PatchEmbed(
            height=resolution, width=resolution, patch_size=patch_size,
            in_channels=in_channels, embed_dim=dim, pos_embed_type="sincos",
        )
        # Optional learnable grid positional embedding (helps structured-grid tasks).
        self.grid_pos = nn.Parameter(torch.zeros(1, seq_len, dim)) if use_grid_pos_embed else None
        if self.grid_pos is not None:
            nn.init.trunc_normal_(self.grid_pos, std=0.02)

        # DiT layers, each augmented with a TRM refinement block. weight_tied=True builds ONE
        # layer reused for all steps (full TRM parameter tying); else num_layers distinct layers.
        n_distinct = 1 if weight_tied else num_layers
        self.layers = nn.ModuleList([
            TRMDiTLayer(
                dim, num_attention_heads, attention_head_dim, ffn_mult, activation_fn, dropout,
                num_embeds_ada_norm, n_inner, T, num_inner_layers, shared_stack, refine_residual,
                isolate_transform,
            )
            for _ in range(n_distinct)
        ])
        # AdaLN-"Zero": zero-init each DiT block's modulation Linear -> identity start.
        for layer in self.layers:
            nn.init.zeros_(layer.dit.norm1.linear.weight)
            nn.init.zeros_(layer.dit.norm1.linear.bias)

        # Fixed (non-trained) initial scratch state z; the answer y starts from the grounded input.
        self.register_buffer("z_init", torch.randn(1, seq_len, dim))

        # Output head: AdaLN-Zero modulation (timestep+class) -> unpatchify. Shared across the
        # per-layer deep-supervision decodes (like TRM's single lm_head).
        self.cond_embed = CombinedTimestepLabelEmbeddings(num_embeds_ada_norm, dim, class_dropout_prob=0.0)
        self.proj_out_1 = nn.Linear(dim, 2 * dim)
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_2 = nn.Linear(dim, patch_size * patch_size * out_channels)

    def _iter_layers(self):
        """Yield the layer for each of num_layers steps (one shared layer if weight_tied)."""
        if self.weight_tied:
            for _ in range(self.num_layers):
                yield self.layers[0]
        else:
            yield from self.layers

    def _unpatchify(self, h: torch.Tensor) -> torch.Tensor:
        p, c, hp, wp = self.patch_size, self.out_channels, self.h_p, self.w_p
        h = h.reshape(-1, hp, wp, p, p, c)
        h = torch.einsum("nhwpqc->nchpwq", h)
        return h.reshape(-1, c, hp * p, wp * p)

    def _head(self, y: torch.Tensor, timestep: torch.Tensor, class_labels: Optional[torch.Tensor]) -> torch.Tensor:
        """Decode an answer state y -> (B, out_channels, H, W) via conditioned AdaLN output head."""
        cond = self.cond_embed(timestep, class_labels, hidden_dtype=y.dtype)
        shift, scale = self.proj_out_1(F.silu(cond)).chunk(2, dim=1)
        out = self.norm_out(y) * (1 + scale[:, None]) + shift[:, None]
        out = self.proj_out_2(out)
        return self._unpatchify(out)

    def forward(
        self, hidden_states: torch.Tensor, timestep: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None, return_all_steps: bool = False,
    ) -> "_Output":
        inj = self.patch(hidden_states)              # (B, L, D): patchified noisy image = grounding
        if self.grid_pos is not None:
            inj = inj + self.grid_pos
        inj = inj.float()
        y = inj                                      # answer starts at the grounded input
        z = self.z_init.expand(inj.shape[0], -1, -1).contiguous().float()  # scratch carry

        all_samples: list = []
        for layer in self._iter_layers():
            y, z = layer(y, z, inj, timestep, class_labels)  # carry (y, z) preserved across layers
            if return_all_steps:
                all_samples.append(self._head(y, timestep, class_labels))  # deep supervision

        if return_all_steps:
            return _Output(all_samples[-1], all_samples)
        return _Output(self._head(y, timestep, class_labels))


class _Output:
    """Minimal output container matching the Transformer2DModel interface"""
    __slots__ = ("sample", "all_samples")

    def __init__(self, sample: torch.Tensor, all_samples=None) -> None:
        self.sample = sample
        self.all_samples = all_samples


class SokobanEmbeddedTRMDiffusion(SokobanBitDiffusion):
    """Embedded-TRM denoiser trainer.

    The model exposes the standard denoiser interface, so sampling/eval/EMA are inherited from
    SokobanBitDiffusion unchanged. The ONLY override is the loss: a DEEP-SUPERVISION objective
    that applies the diffusion loss to the x0 prediction of every layer (final weight 1.0,
    earlier layers weight ``aux_loss_weight``), training each refinement step to improve.
    """

    def __init__(self, *args, aux_loss_weight: float = 0.3, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.aux_loss_weight = aux_loss_weight
        self.hparams["aux_loss_weight"] = aux_loss_weight

    def _single_diffusion_loss(
        self, model_output: torch.Tensor, x_bits: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """SNR-weighted MSE (x0 prediction) or plain MSE (epsilon) for one prediction."""
        prediction_type = self.noise_scheduler.config.prediction_type
        if prediction_type == "sample":
            alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(x_bits.device)
            alpha_t = self._extract_into_tensor(alphas_cumprod, timesteps, x_bits.shape)
            snr = alpha_t / (1.0 - alpha_t).clamp(min=1e-5)
            snr_weights = snr.clamp(max=5.0)
            return (snr_weights * F.mse_loss(model_output.float(), x_bits.float(), reduction="none")).mean()
        elif prediction_type == "epsilon":
            return F.mse_loss(model_output.float(), noise.float())
        raise ValueError(f"Unsupported prediction_type: {prediction_type}")

    def _compute_loss(self, x_bits, class_labels=None, cond_board=None):
        """Deep-supervision diffusion loss: every layer's x0 prediction is supervised.

        Reuses the base diffusion setup (timestep/noise sampling, CFG dropout, self-conditioning)
        but runs the model with return_all_steps=True and sums per-layer losses with weights
        [aux_loss_weight, ..., aux_loss_weight, 1.0], normalized by the weight sum so the loss
        scale is independent of num_layers.
        """
        device = x_bits.device
        B = x_bits.shape[0]

        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (B,), device=device
        ).long()
        noise = torch.randn_like(x_bits)
        x_t = self.noise_scheduler.add_noise(x_bits, noise, timesteps)
        labels_tr, cond_tr = self._apply_cfg_dropout(class_labels, cond_board)  # type: ignore[arg-type]

        # Self-conditioning: 50% chance to seed with the model's own (final) prediction.
        x_pred = None
        if self.self_cond and torch.rand(1).item() > 0.5:
            with torch.no_grad():
                mi0 = self._build_model_input(x_t, None, cond_tr)
                x_pred = self._model_forward(mi0, timesteps, labels_tr).detach().clamp(-1.0, 1.0)

        model_input = self._build_model_input(x_t, x_pred, cond_tr)
        out = self.model(
            hidden_states=model_input, timestep=timesteps, class_labels=labels_tr, return_all_steps=True
        )
        preds = out.all_samples  # list of (B, C, H, W), last = final prediction

        n = len(preds)
        total = x_bits.new_zeros(())
        for i, pred in enumerate(preds):
            weight = 1.0 if i == n - 1 else self.aux_loss_weight
            total = total + weight * self._single_diffusion_loss(pred, x_bits, noise, timesteps)
        weight_sum = 1.0 + self.aux_loss_weight * (n - 1)
        return total / weight_sum



@hydra.main(version_base=None, config_path="config", config_name="embedded_trm_diffusion")
def main(cfg: DictConfig) -> None:
    L.seed_everything(cfg.get("seed", 42), workers=True)

    num_bits = cfg.num_bits
    k_values = cfg.dataset.get("k_values", [3, 8, 10])
    num_classes = len(k_values) if cfg.conditioning == "k_steps" else cfg.num_classes
    use_self_cond = cfg.get("self_cond", True)
    self_cond_mult = 2 if use_self_cond else 1
    if cfg.conditioning == "k_steps":
        in_channels = num_bits * self_cond_mult + num_bits
    else:
        in_channels = num_bits * self_cond_mult

    model = EmbeddedTRMDiffusion(
        resolution=cfg.resolution,
        in_channels=in_channels,
        out_channels=num_bits,
        num_attention_heads=cfg.model.num_attention_heads,
        attention_head_dim=cfg.model.attention_head_dim,
        num_embeds_ada_norm=num_classes + 1,
        num_layers=cfg.model.num_layers,
        n_inner=cfg.trm.n_inner,
        T=cfg.trm.T,
        num_inner_layers=cfg.trm.get("num_inner_layers", 1),
        shared_stack=cfg.trm.get("shared_stack", False),
        weight_tied=cfg.trm.get("weight_tied", False),
        refine_residual=cfg.trm.get("refine_residual", True),
        isolate_transform=cfg.trm.get("isolate_transform", False),
        ffn_mult=cfg.model.get("ffn_mult", 4),
        activation_fn=cfg.model.get("activation_fn", "gelu-approximate"),
        dropout=cfg.model.get("dropout", 0.0),
        patch_size=cfg.model.get("patch_size", 1),
        use_grid_pos_embed=cfg.trm.get("use_grid_pos_embed", True),
    )

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=cfg.get("ddpm_num_train_timesteps", 1000),
        beta_schedule=cfg.get("beta_schedule", "squaredcos_cap_v2"),
        prediction_type=cfg.get("prediction_type", "sample"),
        rescale_betas_zero_snr=cfg.get("rescale_betas_zero_snr", True),
        clip_sample=True,
        clip_sample_range=1.0,
    )

    lit_model = SokobanEmbeddedTRMDiffusion(
        model=model,
        noise_scheduler=noise_scheduler,
        conditioning=cfg.conditioning,
        num_classes=num_classes,
        num_bits=num_bits,
        resolution=cfg.resolution,
        inference_steps=cfg.get("inference_steps", 300),
        self_cond=use_self_cond,
        cfg_drop_rate=cfg.cfg_drop_rate,
        guidance_scale=cfg.guidance_scale,
        time_shift_xi=cfg.get("time_shift_xi", 0.0),
        lr=cfg.lr,
        betas=tuple(cfg.get("adam_betas", [0.9, 0.999])),
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.get("warmup_steps", 500),
        num_epochs=cfg.num_epochs,
        eval_every_n_epochs=cfg.eval_every_n_epochs,
        num_eval_samples=cfg.num_eval_samples,
        k_values=k_values if cfg.conditioning == "k_steps" else None,
        aux_loss_weight=cfg.get("aux_loss_weight", 0.3),
    )

    data_module = SokobanBitDataModule(
        data_path=cfg.dataset.data_path,
        val_data_path=cfg.dataset.get("val_data_path", None),
        conditioning=cfg.conditioning,
        total_train_size=cfg.dataset.total_train_size,
        total_eval_size=cfg.dataset.total_eval_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        num_bits=num_bits,
        k_values=cfg.dataset.get("k_values", [3, 8, 10]),
        use_dihedral_aug=cfg.dataset.get("use_dihedral_aug", True),
        bot_removal_prob=cfg.dataset.get("bot_removal_prob", 0.75),
    )

    run_name = cfg.get("run_name", None)
    wandb_logger = WandbLogger(
        project=cfg.get("wandb_project", "Sokoban-EmbeddedTRM-BitDiffusion"),
        group=cfg.get("wandb_group", None),
        name=run_name,
        save_dir=cfg.output_dir,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    callbacks = [
        EMACallback(decay=cfg.get("ema_decay", 0.9999), inv_gamma=1.0, power=0.75),
        ModelCheckpoint(
            dirpath=Path(cfg.output_dir) / "checkpoints",
            filename="best-{epoch}-{step}", monitor="val/loss", mode="min", save_top_k=1, save_last=True, verbose=True,
        ),
        ModelCheckpoint(
            dirpath=Path(cfg.output_dir) / "checkpoints",
            filename="periodic-{epoch}", every_n_epochs=cfg.save_every_n_epochs, save_top_k=-1,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = L.Trainer(
        max_epochs=cfg.num_epochs,
        accelerator="auto",
        devices="auto",
        precision=cfg.get("precision", "16-mixed"),
        logger=wandb_logger,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        accumulate_grad_batches=cfg.get("gradient_accumulation_steps", 1),
        log_every_n_steps=10,
        val_check_interval=1.0,
    )

    # Persist the W&B run id next to the checkpoints so sample.py can resume this same experiment and log test metrics onto these training charts.
    if rank_zero_only.rank == 0:
        try:
            ckpt_dir = Path(cfg.output_dir) / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            (ckpt_dir / "wandb_run_id.txt").write_text(str(wandb_logger.experiment.id))
        except Exception as e:
            print(f"Could not save W&B run id: {e}")

    trainer.fit(lit_model, datamodule=data_module, ckpt_path=cfg.get("resume_from_checkpoint", None))


if __name__ == "__main__":
    sys.argv = [a for a in sys.argv if not a.startswith("--")]
    main()

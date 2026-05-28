"""
factory.py — Build models, datasets, and schedulers from a Hydra DictConfig.

Each builder reads only from the config; callers should not need to unpack
individual fields before calling these functions.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
from diffusers import DDPMScheduler
from omegaconf import DictConfig
from torch.utils.data import Dataset, random_split

from configs.schemas import (
    ClassifierLossConfig,
    EvalConfig,
    ImageEncoderConfig,
    LatentDiTConfig,
    LatentDiTOptimConfig,
    NoisySwapConfig,
    PainterConfig,
    PainterOptimConfig,
    PainterStageConfig,
    PainterThinkerConfig,
    PuzzleEmbOptimConfig,
    ThinkerModelConfig,
    ThinkerOptimConfig,
    ThinkerStageConfig,
    TimestepCondConfig,
    TrainConfig,
    TwoStageConfig,
)
from datasets.mnist_sudoku_dataset import MNISTSudokuDataset
from datasets.sudoku_dataset import SudokuDataset
from models.base import BaseModel
from models.painter_thinkers import (
    OriginalTRMRatatouilleV0,
    OriginalTRMRatatouilleV1,
    OriginalTRMRatatouilleV2,
    OriginalTRMRatatouilleV3,
    OriginalTRMRatatouilleV4,
    PainterThinkerV0Tok,
    ThinkerWithFrozenPainter,
    ThinkerWithFrozenPainterV0,
    ThinkerWithFrozenPainterV1,
    ThinkerWithFrozenPainterV1Verif,
    ThinkerWithFrozenPainterControlNet,
)
from models.painters import (
    StandalonePainter,
    StandalonePainterControl,
    StandalonePainterSPADE,
)
from models.trm_wrappers import SpatialTRM
from models.utility_models import strip_compiled_prefix

# ── Sub-config builders ────────────────────────────────────────────────────────


def _noisy_swap_cfg(cfg: DictConfig) -> Optional[NoisySwapConfig]:
    ns = cfg.train.get("noisy_swap", None)
    if ns is None:
        return None
    return NoisySwapConfig(
        prob=float(ns.get("prob", 0.0)),
        t_min=int(ns.get("t_min", 80)),
        t_max=int(ns.get("t_max", 100)),
    )


def _classifier_loss_cfg(cfg: DictConfig) -> Optional[ClassifierLossConfig]:
    cl = cfg.train.get("classifier_loss", None)
    if cl is None:
        return None
    return ClassifierLossConfig(
        weight=float(cl.get("weight", 0.0)),
        t_max=int(cl.get("t_max", 200)),
        target=str(cl.get("target", "x0_pred")),
        loss_type=str(cl.get("loss_type", "ce")),
        noisy_classifier=bool(cl.get("noisy_classifier", False)),
        classifier_path=cl.get("classifier_path", None),
    )


def _two_stage_cfg(cfg: DictConfig) -> Optional[TwoStageConfig]:
    ts = cfg.train.get("two_stage", None)
    if ts is None:
        return None
    ps = ts.get("painter", {})
    tp = ts.get("thinker", {})
    return TwoStageConfig(
        painter=PainterStageConfig(
            n_sup=int(ps.get("n_sup", -1)),
            freeze_thinker=bool(ps.get("freeze_thinker", False)),
            every=int(ps.get("every", 1)),
            H_cycles=ps.get("H_cycles", None),
            L_cycles=ps.get("L_cycles", None),
        ),
        thinker=ThinkerStageConfig(
            n_sup=int(tp.get("n_sup", -1)),
            every=int(tp.get("every", 1)),
        ),
    )


def _train_cfg(cfg: DictConfig) -> TrainConfig:
    tr = cfg.train
    return TrainConfig(
        seed=int(tr.seed),
        batch_size=int(tr.batch_size),
        num_steps=int(tr.num_steps),
        compile=bool(tr.get("compile", True)),
        gradient_accumulation_steps=int(tr.get("gradient_accumulation_steps", 1)),
        sudoku_loss_weight=float(tr.get("sudoku_loss_weight", 1.0)),
        mse_loss_weight=float(tr.get("mse_loss_weight", 1.0)),
        cfg_prob=float(tr.get("cfg_prob", 0.0)),
        noisy_swap=_noisy_swap_cfg(cfg),
        classifier_loss=_classifier_loss_cfg(cfg),
        two_stage=_two_stage_cfg(cfg),
    )


def _eval_cfg(cfg: DictConfig) -> EvalConfig:
    ev = cfg.eval
    return EvalConfig(
        eval_every=int(ev.eval_every),
        save_every=int(ev.save_every),
        log_every=int(ev.log_every),
        cfg_scale=float(ev.get("cfg_scale", 1.0)),
        num_samples=int(ev.get("num_samples", 7680)),
        batch_size=int(ev.get("batch_size", 384)),
        num_ddim_steps=int(ev.get("num_ddim_steps", 20)),
        num_log_images=int(ev.get("num_log_images", 10)),
        classifier_path=ev.get("classifier_path", None),
    )


def _thinker_optim_cfg(cfg: DictConfig) -> ThinkerOptimConfig:
    ot = cfg.thinker.optim
    puzzle_emb = None
    pe = ot.get("puzzle_emb", None)
    if pe is not None:
        puzzle_emb = PuzzleEmbOptimConfig(
            lr=float(pe.lr),
            weight_decay=float(pe.weight_decay),
            warmup_steps=int(pe.warmup_steps),
            lr_min_ratio=float(pe.get("lr_min_ratio", 0.0)),
        )
    return ThinkerOptimConfig(
        lr=float(ot.lr),
        weight_decay=float(ot.weight_decay),
        beta1=float(ot.get("beta1", 0.9)),
        beta2=float(ot.get("beta2", 0.95)),
        warmup_steps=int(ot.warmup_steps),
        lr_min_ratio=float(ot.get("lr_min_ratio", 0.0)),
        puzzle_emb=puzzle_emb,
    )


def _painter_optim_cfg(cfg: DictConfig) -> PainterOptimConfig:
    op = cfg.painter.optim
    return PainterOptimConfig(
        lr=float(op.lr),
        weight_decay=float(op.weight_decay),
        warmup_steps=int(op.get("warmup_steps", cfg.thinker.optim.warmup_steps)),
        lr_min_ratio=float(op.get("lr_min_ratio", 0.0)),
    )


def _thinker_model_cfg(cfg: DictConfig, vocab_size: int) -> ThinkerModelConfig:
    t = cfg.thinker
    return ThinkerModelConfig(
        vocab_size=vocab_size,
        seq_len=int(t.seq_len),
        hidden_size=int(t.hidden_size),
        n_heads=int(t.n_heads),
        L_layers=int(t.L_layers),
        L_cycles=int(t.L_cycles),
        H_cycles=int(t.H_cycles),
        n_sup=int(t.n_sup),
        batch_size=int(cfg.train.batch_size),
        forward_dtype=str(cfg.precision.thinker_dtype),
        expansion=float(t.expansion),
        pos_encodings=str(t.pos_encodings),
        mlp_t=bool(t.mlp_t),
        puzzle_emb_ndim=int(t.get("puzzle_emb_ndim", 0)),
        puzzle_emb_len=int(t.get("puzzle_emb_len", 16)),
        num_puzzle_identifiers=int(t.get("num_puzzle_identifiers", 1)),
        halt_exploration_prob=float(t.get("halt_exploration_prob", 0.0)),
        freeze_weights=bool(t.get("freeze_weights", False)),
    )


def _painter_cfg(cfg: DictConfig) -> PainterConfig:
    p = cfg.painter
    cell_size = int(cfg.data.cell_size)
    return PainterConfig(
        vocab_size=int(cfg.data.vocab_size),
        painter_size=9 * cell_size,
        cell_size=cell_size,
        bridge_channels=int(p.bridge_channels),
        painter_channels=tuple(p.painter_channels),
        painter_layers_per_block=int(p.painter_layers_per_block),
        painter_dtype=cfg.precision.get("painter_dtype", None),
    )


def _painter_thinker_cfg(cfg: DictConfig) -> PainterThinkerConfig:
    p = cfg.painter
    cell_size = int(cfg.data.cell_size)
    return PainterThinkerConfig(
        painter_size=9 * cell_size,
        cell_size=cell_size,
        bridge_channels=int(p.bridge_channels),
        painter_channels=tuple(p.painter_channels),
        painter_layers_per_block=int(p.painter_layers_per_block),
        diff_thinker_weight=float(p.get("diff_thinker_weight", 1.0)),
        thinker_bridge_mode=str(p.get("thinker_bridge_mode", "logits")),
        painter_dtype=cfg.precision.get("painter_dtype", None),
    )


def _image_encoder_cfg(cfg: DictConfig) -> ImageEncoderConfig:
    p = cfg.painter
    return ImageEncoderConfig(
        enc_channels=int(p.get("enc_channels", 32)),
        enc_hidden_channels=tuple(p.get("enc_hidden_channels", [16, 32])),
        thinker_out_channels=p.get("thinker_out_channels", None),
    )


def _timestep_cond_cfg(cfg: DictConfig) -> Optional[TimestepCondConfig]:
    p = cfg.painter
    enc_t = bool(p.get("enc_timestep_cond", False))
    thinker_t = bool(p.get("thinker_timestep_cond", False))
    decoder_t = bool(p.get("decoder_timestep_cond", False))
    if not enc_t and not thinker_t and not decoder_t:
        return None
    return TimestepCondConfig(
        enc_timestep_cond=enc_t,
        thinker_timestep_cond=thinker_t,
        decoder_timestep_cond=decoder_t,
        temb_dim=int(p.get("temb_dim", 256)),
    )


# ── Latent DiT sub-config builders ────────────────────────────────────────────


def _latent_dit_cfg(cfg: DictConfig) -> LatentDiTConfig:
    d = cfg.latent_dit
    cell_size = int(cfg.data.cell_size)
    return LatentDiTConfig(
        vae_checkpoint=str(d.vae_checkpoint),
        latent_channels=int(d.get("latent_channels", 4)),
        latent_size=int(d.get("latent_size", 36)),
        patch_size=int(d.get("patch_size", 4)),
        num_attention_heads=int(d.get("num_attention_heads", 8)),
        attention_head_dim=int(d.get("attention_head_dim", 64)),
        num_layers=int(d.get("num_layers", 6)),
        mlp_ratio=float(d.get("mlp_ratio", 4.0)),
        dropout=float(d.get("dropout", 0.0)),
        vocab_size=int(d.get("vocab_size", 11)),
        cond_embed_dim=int(d.get("cond_embed_dim", 256)),
        cell_size=cell_size,
        painter_size=cell_size * 9,
    )


def _latent_dit_optim_cfg(cfg: DictConfig) -> LatentDiTOptimConfig:
    o = cfg.latent_dit.optim
    return LatentDiTOptimConfig(
        lr=float(o.lr),
        weight_decay=float(o.weight_decay),
        warmup_steps=int(o.warmup_steps),
        lr_min_ratio=float(o.get("lr_min_ratio", 0.1)),
    )


def _build_vae(cfg: DictConfig):
    from diffusers import AutoencoderKL

    v = cfg.latent_dit.vae_arch
    return AutoencoderKL(
        in_channels=int(v.in_channels),
        out_channels=int(v.out_channels),
        latent_channels=int(v.latent_channels),
        down_block_types=list(v.down_block_types),
        up_block_types=list(v.up_block_types),
        block_out_channels=list(v.block_out_channels),
        layers_per_block=int(v.layers_per_block),
        norm_num_groups=int(v.norm_num_groups),
        act_fn=str(v.act_fn),
    )


# ── Public builders ────────────────────────────────────────────────────────────


def build_scheduler(cfg: DictConfig) -> Optional[DDPMScheduler]:
    if cfg.mode == "sudoku":
        return None
    return DDPMScheduler(
        num_train_timesteps=int(cfg.diffusion.num_train_timesteps),
        beta_schedule=str(cfg.diffusion.beta_schedule),
        prediction_type=str(cfg.diffusion.prediction_type),
    )


def build_datasets(cfg: DictConfig) -> tuple[Dataset, Dataset]:
    if cfg.mode == "sudoku":
        train_dir = os.path.join(cfg.data.sudoku_dir, "train")
        test_dir = os.path.join(cfg.data.sudoku_dir, "test")
        train_ds = SudokuDataset(train_dir, mask_given=True)
        if os.path.isdir(test_dir):
            return train_ds, SudokuDataset(test_dir, mask_given=True)
        n_val = max(1, int(0.1 * len(train_ds)))
        return random_split(train_ds, [len(train_ds) - n_val, n_val])

    cell_size = int(cfg.data.cell_size)
    train_dir = os.path.join(cfg.data.sudoku_dir, "train")
    test_dir = os.path.join(cfg.data.sudoku_dir, "test")
    sudoku_eval_dir = test_dir if os.path.isdir(test_dir) else train_dir
    num_givens = cfg.data.get("num_givens", None)
    if num_givens is not None:
        num_givens = int(num_givens)
    train_ds = MNISTSudokuDataset(
        sudoku_dir=train_dir,
        mnist_root=cfg.data.mnist_root,
        cell_size=cell_size,
        mnist_split="train",
        mask_given=True,
        num_givens=num_givens,
    )
    eval_ds = MNISTSudokuDataset(
        sudoku_dir=sudoku_eval_dir,
        mnist_root=cfg.data.mnist_root,
        cell_size=cell_size,
        mnist_split="test",
        mask_given=True,
        num_givens=num_givens,
    )
    return train_ds, eval_ds


_PAINTER_THINKER_VARIANTS: dict[str, type] = {
    "v0tok": PainterThinkerV0Tok,
    "v0": OriginalTRMRatatouilleV0,
    "v1": OriginalTRMRatatouilleV1,
    "v2": OriginalTRMRatatouilleV2,
    "v3": OriginalTRMRatatouilleV3,
    "v4": OriginalTRMRatatouilleV4,
}

_STANDALONE_PAINTER_MODES: dict[str, type] = {
    "standalone_painter": StandalonePainter,
    "standalone_painter_spade": StandalonePainterSPADE,
    "standalone_painter_control": StandalonePainterControl,
}


def build_model(cfg: DictConfig, scheduler) -> BaseModel:
    mode = str(cfg.mode)
    train_cfg = _train_cfg(cfg)
    eval_cfg = _eval_cfg(cfg)

    if mode == "latent_dit":
        from models.latent_dit import LatentDiT

        dit_cfg = _latent_dit_cfg(cfg)
        vae_ckpt_path = dit_cfg.vae_checkpoint

        # Try to load scaling factor saved by train_vae.py next to the checkpoint.
        ckpt_dir = os.path.dirname(vae_ckpt_path)
        sf_path = os.path.join(ckpt_dir, "scaling_factor.pt")
        scaling_factor = (
            torch.load(sf_path, map_location="cpu", weights_only=True)["scaling_factor"]
            if os.path.exists(sf_path)
            else 1.0
        )

        ckpt = torch.load(vae_ckpt_path, map_location="cpu", weights_only=False)
        vae = _build_vae(cfg)
        vae.load_state_dict(ckpt["model_state"])
        vae.eval()

        return LatentDiT(
            model_cfg=dit_cfg,
            optim_cfg=_latent_dit_optim_cfg(cfg),
            train_cfg=train_cfg,
            eval_cfg=eval_cfg,
            scheduler=scheduler,
            vae=vae,
            scaling_factor=scaling_factor,
        )

    if mode == "sudoku":
        thinker_cfg = _thinker_model_cfg(cfg, vocab_size=int(cfg.data.vocab_size))
        return SpatialTRM(
            optim_cfg=_thinker_optim_cfg(cfg),
            vocab_size=thinker_cfg.vocab_size,
            seq_len=thinker_cfg.seq_len,
            hidden_size=thinker_cfg.hidden_size,
            n_heads=thinker_cfg.n_heads,
            L_layers=thinker_cfg.L_layers,
            L_cycles=thinker_cfg.L_cycles,
            H_cycles=thinker_cfg.H_cycles,
            n_sup=thinker_cfg.n_sup,
            expansion=thinker_cfg.expansion,
            forward_dtype=thinker_cfg.forward_dtype,
            mlp_t=thinker_cfg.mlp_t,
            pos_encodings=thinker_cfg.pos_encodings,
            puzzle_emb_ndim=thinker_cfg.puzzle_emb_ndim,
            puzzle_emb_len=thinker_cfg.puzzle_emb_len,
            num_puzzle_identifiers=thinker_cfg.num_puzzle_identifiers,
            halt_exploration_prob=thinker_cfg.halt_exploration_prob,
            batch_size=thinker_cfg.batch_size,
            freeze_weights=thinker_cfg.freeze_weights,
        )

    if mode in _STANDALONE_PAINTER_MODES:
        cls = _STANDALONE_PAINTER_MODES[mode]
        return cls(
            model_cfg=_painter_cfg(cfg),
            optim_cfg=_painter_optim_cfg(cfg),
            train_cfg=train_cfg,
            eval_cfg=eval_cfg,
            scheduler=scheduler,
        )

    if mode == "thinker_frozen_painter":
        painter_ckpt_path = cfg.painter.get("painter_checkpoint", None)
        if painter_ckpt_path is None:
            raise ValueError("thinker_frozen_painter requires painter.painter_checkpoint to be set.")
        # Use classifier_path=None so the frozen backbone doesn't load eval_clf.
        _frozen_eval_cfg = EvalConfig(
            eval_every=eval_cfg.eval_every,
            save_every=eval_cfg.save_every,
            log_every=eval_cfg.log_every,
            classifier_path=None,
        )
        # Strip classifier_loss so the frozen backbone doesn't create a train_clf submodule,
        # which would cause state_dict key mismatch when loading the pre-trained checkpoint.
        # The outer ThinkerWithFrozenPainter still receives the full train_cfg and loads train_clf itself.
        from dataclasses import replace as _dc_replace

        _frozen_train_cfg = _dc_replace(train_cfg, classifier_loss=None)

        painter_variant = str(cfg.get("painter_variant", "v0tok"))

        if painter_variant == "controlnet":
            frozen_painter = StandalonePainterControl(
                model_cfg=_painter_cfg(cfg),
                optim_cfg=_painter_optim_cfg(cfg),
                train_cfg=_frozen_train_cfg,
                eval_cfg=_frozen_eval_cfg,
                scheduler=scheduler,
            )
        else:
            frozen_painter = StandalonePainter(
                model_cfg=_painter_cfg(cfg),
                optim_cfg=_painter_optim_cfg(cfg),
                train_cfg=_frozen_train_cfg,
                eval_cfg=_frozen_eval_cfg,
                scheduler=scheduler,
            )
        ckpt = torch.load(painter_ckpt_path, map_location="cpu", weights_only=False)
        frozen_painter.load_state_dict(strip_compiled_prefix(ckpt["model_state"]))
        thinker_optim_cfg = _thinker_optim_cfg(cfg)
        painter_optim_cfg = _painter_optim_cfg(cfg)
        model_cfg = _painter_thinker_cfg(cfg)

        if painter_variant == "controlnet":
            thinker_cfg = _thinker_model_cfg(cfg, vocab_size=int(cfg.data.vocab_size))
            thinker_cfg.puzzle_emb_ndim = 0
            thinker_cfg.puzzle_emb_len = 0
            return ThinkerWithFrozenPainterControlNet(
                painter=frozen_painter,
                thinker_cfg=thinker_cfg,
                encoder_cfg=_image_encoder_cfg(cfg),
                model_cfg=model_cfg,
                train_cfg=train_cfg,
                eval_cfg=eval_cfg,
                thinker_optim_cfg=thinker_optim_cfg,
                painter_optim_cfg=painter_optim_cfg,
                scheduler=scheduler,
            )

        if painter_variant == "v0":
            thinker_cfg = _thinker_model_cfg(cfg, vocab_size=int(cfg.data.vocab_size))
            thinker_cfg.puzzle_emb_ndim = 0
            thinker_cfg.puzzle_emb_len = 0
            return ThinkerWithFrozenPainterV0(
                painter=frozen_painter,
                thinker_cfg=thinker_cfg,
                encoder_cfg=_image_encoder_cfg(cfg),
                model_cfg=model_cfg,
                train_cfg=train_cfg,
                eval_cfg=eval_cfg,
                thinker_optim_cfg=thinker_optim_cfg,
                painter_optim_cfg=painter_optim_cfg,
                scheduler=scheduler,
            )

        if painter_variant in ("v1", "v1_verif"):
            thinker_cfg = _thinker_model_cfg(cfg, vocab_size=int(cfg.data.vocab_size))
            thinker_cfg.puzzle_emb_ndim = 0
            thinker_cfg.puzzle_emb_len = 0
            cls = ThinkerWithFrozenPainterV1Verif if painter_variant == "v1_verif" else ThinkerWithFrozenPainterV1
            extra = {}
            if painter_variant == "v1_verif":
                extra["verif_weight"] = float(cfg.painter.get("verif_weight", 0.1))
                extra["verif_max_corruptions"] = int(cfg.painter.get("verif_max_corruptions", 5))
            return cls(
                painter=frozen_painter,
                thinker_cfg=thinker_cfg,
                encoder_cfg=_image_encoder_cfg(cfg),
                model_cfg=model_cfg,
                train_cfg=train_cfg,
                eval_cfg=eval_cfg,
                thinker_optim_cfg=thinker_optim_cfg,
                painter_optim_cfg=painter_optim_cfg,
                scheduler=scheduler,
                timestep_cfg=_timestep_cond_cfg(cfg),
                **extra,
            )

        return ThinkerWithFrozenPainter(
            painter=frozen_painter,
            thinker_cfg=_thinker_model_cfg(cfg, vocab_size=int(cfg.data.vocab_size)),
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            eval_cfg=eval_cfg,
            thinker_optim_cfg=thinker_optim_cfg,
            painter_optim_cfg=painter_optim_cfg,
            scheduler=scheduler,
            adapter_in_channels=int(cfg.painter.get("adapter_in_channels", 0)),
        )

    # Painter-thinker variants (mode == "painter")
    painter_variant = str(cfg.get("painter_variant", "v0tok"))
    cls = _PAINTER_THINKER_VARIANTS.get(painter_variant)
    if cls is None:
        raise ValueError(
            f"Unknown painter_variant: {painter_variant!r}. " f"Choose from: {list(_PAINTER_THINKER_VARIANTS)}"
        )

    model_cfg = _painter_thinker_cfg(cfg)
    thinker_optim_cfg = _thinker_optim_cfg(cfg)
    painter_optim_cfg = _painter_optim_cfg(cfg)

    if painter_variant == "v0tok":
        return cls(
            thinker_cfg=_thinker_model_cfg(cfg, vocab_size=int(cfg.data.vocab_size)),
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            eval_cfg=eval_cfg,
            thinker_optim_cfg=thinker_optim_cfg,
            painter_optim_cfg=painter_optim_cfg,
            scheduler=scheduler,
        )

    # Image-conditioned variants (V0–V4).
    # V2/V3/V4: vocab_size = thinker_out_channels (latent, no CE supervision).
    # V0/V1:    vocab_size = num_classes (digit classifier output, default 9).
    if painter_variant in ("v2", "v3", "v4"):
        vocab_size = int(cfg.painter.get("thinker_out_channels", 16))
    else:
        vocab_size = int(cfg.thinker.get("num_classes", 9))

    thinker_cfg = _thinker_model_cfg(cfg, vocab_size=vocab_size)
    # Image-conditioned variants (V0–V4) don't use puzzle identifiers.
    # Force to 0 so test overrides of thinker.puzzle_emb_ndim don't bleed in.
    thinker_cfg.puzzle_emb_ndim = 0
    thinker_cfg.puzzle_emb_len = 0
    encoder_cfg = _image_encoder_cfg(cfg)

    if painter_variant == "v4":
        return cls(
            thinker_cfg=thinker_cfg,
            encoder_cfg=encoder_cfg,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            eval_cfg=eval_cfg,
            thinker_optim_cfg=thinker_optim_cfg,
            painter_optim_cfg=painter_optim_cfg,
            scheduler=scheduler,
            compression_factor=int(cfg.painter.get("compression_factor", 16)),
            bridge_num_heads=int(cfg.painter.get("bridge_num_heads", 4)),
            timestep_cfg=_timestep_cond_cfg(cfg),
        )

    # V0, V1, V2, V3 — V1+ accept timestep_cfg (None = disabled)
    timestep_cfg = _timestep_cond_cfg(cfg) if painter_variant != "v0" else None
    return cls(
        thinker_cfg=thinker_cfg,
        encoder_cfg=encoder_cfg,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        eval_cfg=eval_cfg,
        thinker_optim_cfg=thinker_optim_cfg,
        painter_optim_cfg=painter_optim_cfg,
        scheduler=scheduler,
        **({"timestep_cfg": timestep_cfg} if painter_variant != "v0" else {}),
    )

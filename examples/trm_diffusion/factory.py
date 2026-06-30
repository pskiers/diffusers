"""
factory.py — Build models, datasets, and schedulers from a Hydra DictConfig.

Supports two training modes (set via cfg.mode in the experiment config):

  painter_base  — Standalone painter (UNetPainter / DiTPainter).
                  The full model is declared under cfg.painter with a Hydra
                  _target_; scheduler / train_cfg / eval_cfg are injected here.

  thinker_base  — ThinkerFrozenPainterBase: TRM thinker + frozen pre-trained
                  painter.  Sub-configs (thinker, painter, condition_encoder,
                  translator, loss, eval_callbacks) are separate Hydra config
                  groups wired together by build_model.
"""

from __future__ import annotations

from typing import Optional

from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import Dataset

from configs.schemas import (
    ClassifierLossConfig,
    EvalConfig,
    NoisySwapConfig,
    PainterStageConfig,
    ThinkerStageConfig,
    TrainConfig,
    TwoStageConfig,
)
from models.base import BaseModel
from models.painter_thinkers import ThinkerFrozenPainterBase


# ── TrainConfig / EvalConfig builders ─────────────────────────────────────────


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
        noisy_dropout_p_max=float(tr.get("noisy_dropout_p_max", 0.0)),
        minsnr_gamma=float(tr["minsnr_gamma"]) if tr.get("minsnr_gamma") is not None else None,
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


# ── Public builders ────────────────────────────────────────────────────────────


def build_datasets(cfg: DictConfig) -> tuple[Dataset, Dataset]:
    return instantiate(cfg.data.train_dataset), instantiate(cfg.data.val_dataset)


def build_model(cfg: DictConfig, scheduler) -> BaseModel:
    mode = str(cfg.mode)
    train_cfg = _train_cfg(cfg)
    eval_cfg = _eval_cfg(cfg)

    sampling = cfg.get("sampling")

    if mode == "painter_base":
        # Full model is declared under cfg.painter (or cfg.model) with _target_.
        # scheduler / train_cfg / eval_cfg / sampling_pipeline are injected here.
        model_cfg = cfg.get("model") or cfg.get("painter")
        return instantiate(
            model_cfg,
            scheduler=scheduler,
            train_cfg=train_cfg,
            eval_cfg=eval_cfg,
            sampling_pipeline=sampling,
        )

    if mode == "thinker_base":
        # ThinkerFrozenPainterBase is assembled from separate config groups.
        # cfg.thinker / cfg.painter / cfg.condition_encoder / cfg.translator /
        # cfg.loss / cfg.eval_callbacks are each their own Hydra config group.
        # Both thinker and painter are instantiated inside ThinkerFrozenPainterBase.
        return ThinkerFrozenPainterBase(
            thinker=cfg.thinker,
            painter=cfg.painter,
            train_cfg=train_cfg,
            eval_cfg=eval_cfg,
            condition_encoder=cfg.condition_encoder,
            loss=cfg.loss,
            thinker_painter_translator=cfg.translator,
            eval_callbacks=list(cfg.eval_callbacks) if cfg.get("eval_callbacks") else None,
            scheduler=scheduler,
            sampling_pipeline=sampling,
        )

    raise ValueError(
        f"Unknown mode: {mode!r}. "
        "Set cfg.mode to 'painter_base' or 'thinker_base' in your experiment config."
    )

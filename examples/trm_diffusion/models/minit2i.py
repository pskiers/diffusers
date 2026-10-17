"""
models/minit2i.py — frozen MiniT2I-B/16 painter (Wang et al. 2026) for thinker
steering on text-to-image: the released checkpoint as an MMDiTPainter
(models/mmdit_painter.py) with its frozen FLAN-T5-Large text encoder.

MiniT2I is a pixel-space MM-DiT (third_party/minit2i/mmdit.py, loaded verbatim
with the released weights): FLAN-T5-Large text tokens and 16x16 image patches
run through 17 joint-attention blocks, and the model predicts the clean image.
It has no timestep or AdaLN pathway (its blocks ignore `vec`), so steering goes
into the token stream instead.

Steering ("read early, steer late"): the first `steer_from_block` blocks run
unsteered; their output (DataSample.painter_hidden) is what the thinker reads,
via PainterHiddenEncoder; PainterResidualTranslator turns the thinker's output
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

import dataclasses
import json
from typing import Optional

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from models.mmdit_painter import FlowMatchingScheduler, MMDiTPainter, PainterHiddenEncoder, T5TextTokenizer, fuse_attention
from third_party.minit2i import mmdit

MINIT2I_REVISION = "5b4da2870d1554992670d1005567e9d55b1b37bb"  # see third_party/minit2i/README.md

# Earlier names, still referenced by saved run configs.
MiniT2IFlowScheduler = FlowMatchingScheduler
MiniT2IHiddenEncoder = PainterHiddenEncoder


def load_minit2i(repo: str = "MiniT2I/MiniT2I", subfolder: str = "minit2i-b-16", revision: str = MINIT2I_REVISION):
    """The released MiniT2I backbone as a vendored mmdit.MMJiT, frozen, with
    fused attention."""

    config = json.loads(open(hf_hub_download(repo, f"{subfolder}/transformer/config.json", revision=revision)).read())
    names = {f.name for f in dataclasses.fields(mmdit.MMJiTConfig)}
    cfg = mmdit.MMJiTConfig(**{k: (tuple(v) if k == "cfg_interval" else v) for k, v in config.items() if k in names})
    net = mmdit.MMJiT(cfg)
    weights = load_file(hf_hub_download(repo, f"{subfolder}/transformer/diffusion_pytorch_model.safetensors", revision=revision))
    net.load_state_dict({k.removeprefix("model.net."): v for k, v in weights.items()}, strict=True)
    return fuse_attention(net).eval().requires_grad_(False)


class MiniT2IPainter(MMDiTPainter):
    """Frozen MiniT2I with its frozen FLAN-T5 text encoder, steerable after
    block `steer_from_block` (see module docstring).

    Reads DataSample.token_conditions / token_mask (T5 token ids and attention
    mask, e.g. from Std120KDataset); images and x_t are in [-1, 1]. Predicts the
    velocity v_hat (pred_type "v_prediction") consumed by FlowMatchingScheduler
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
        super().__init__(
            scheduler=scheduler,
            train_cfg=train_cfg,
            eval_cfg=eval_cfg,
            optim_cfg=optim_cfg,
            cond_tokenizer=T5TextTokenizer(text_encoder),
            net=load_minit2i(repo, subfolder, revision),
            steer_from_block=steer_from_block,
            painter_dtype=painter_dtype,
            pixel_range="[-1,1]",
            noise_scale=noise_scale,
            velocity_clamp=velocity_clamp,
            t_logit_mean=t_logit_mean,
            t_logit_std=t_logit_std,
            label_drop_rate=label_drop_rate,
            freeze=True,
            eval_callbacks=eval_callbacks,
            sampling_pipeline=sampling_pipeline,
        )

    @property
    def text_encoder(self):
        return self.cond_tokenizer.encoder

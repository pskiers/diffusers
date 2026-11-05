# ABOUTME: GRAM-style stochastic thinker (Generative Recursive Reasoning, arXiv 2605.19376): z_H = u + eps after every
# ABOUTME: H-cycle, eps from a posterior that sees the solved image in training and from a learned prior otherwise.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets.data_sample import DataSample
from hydra.utils import instantiate

from models.painter_thinkers import ThinkerFrozenPainterACT
from models.trm.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1InnerCarry
from models.trm_wrappers import SpatialTRM, SpatialTRMInner


def gauss_kl(mu_q: torch.Tensor, sd_q: torch.Tensor, mu_p: torch.Tensor, sd_p: torch.Tensor) -> torch.Tensor:
    """Elementwise KL(N(mu_q, sd_q^2) || N(mu_p, sd_p^2)) in nats; sd tensors broadcast against mu."""
    return torch.log(sd_p / sd_q) + (sd_q.pow(2) + (mu_q - mu_p).pow(2)) / (2 * sd_p.pow(2)) - 0.5


class GaussHead(nn.Module):
    """Token-wise SwiGLU MLP: (..., in_dim) -> mean (..., out_dim) and std (..., 1 or out_dim), in fp32.

    The output layer starts at zero weight, so at init the mean is 0 and the std is init_std everywhere.
    """

    def __init__(
        self, in_dim: int, out_dim: int, per_channel_std: bool, expansion: float, init_std: float, min_std: float
    ):
        super().__init__()
        inter = int(expansion * out_dim)
        self.out_dim = out_dim
        self.min_std = min_std
        self.gate_up = nn.Linear(in_dim, 2 * inter)
        self.down = nn.Linear(inter, out_dim + (out_dim if per_channel_std else 1))
        nn.init.zeros_(self.down.weight)
        with torch.no_grad():
            self.down.bias.zero_()
            self.down.bias[out_dim:] = math.log(math.expm1(init_std - min_std))  # softplus^-1

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate, up = self.gate_up(x.float()).chunk(2, dim=-1)
        out = self.down(F.silu(gate) * up)
        return out[..., : self.out_dim], F.softplus(out[..., self.out_dim :]) + self.min_std


class GramSpatialTRMInner(SpatialTRMInner):
    """SpatialTRMInner whose every z_H update u is followed by z_H = u + eps (self._h_transition, set per call by
    GramSpatialTRM.reasoning_step; identity when unset). Otherwise the vendored forward, line for line."""

    def forward(self, carry, batch):
        seq_info = {"cos_sin": self.rotary_emb() if hasattr(self, "rotary_emb") else None}
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
        transition = getattr(self, "_h_transition", None) or (lambda u, last: u)
        z_H, z_L = carry.z_H, carry.z_L
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles - 1):
                for _L_step in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                z_H = transition(self.L_level(z_H, z_L, **seq_info), False)
        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = transition(self.L_level(z_H, z_L, **seq_info), True)
        if getattr(self, "_keep_carry_grad", False):
            new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H, z_L=z_L)
        else:
            new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        output = self.lm_head(z_H)[:, self.puzzle_emb_len :]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class GramSpatialTRM(SpatialTRM):
    """SpatialTRM with GRAM's stochastic high-level transition: after every H-cycle's z_H update u,
    z_H = u + eps, eps ~ N(mu, sd^2) from a prior head on u, or in training from a posterior head on
    [u ; posterior_emb] (an encoding of the solved image, set as self._posterior_emb by the owning model).
    The prior/posterior Gaussians of the last (gradient-carrying) transition are kept in self._kl_terms for
    the KL loss; the earlier, no-grad transitions are sampled the same way, as in GRAM's Algorithm 1.

    gram_sigma: "token" (one std per token, as GRAM) or "channel" (one per token and channel).
    prior_mode: "sample" or "mean" (eps = the prior mean), for inference.
    """

    def __init__(
        self,
        *args,
        gram_sigma: str = "token",
        gram_init_std: float = 0.1,
        gram_min_std: float = 1e-3,
        gram_head_expansion: float = 2.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if gram_sigma not in ("token", "channel"):
            raise ValueError(f"gram_sigma must be 'token' or 'channel', got {gram_sigma!r}")
        self.inner = GramSpatialTRMInner(self.inner.config)
        h = self.inner.config.hidden_size
        per_channel = gram_sigma == "channel"
        self.prior_head = GaussHead(h, h, per_channel, gram_head_expansion, gram_init_std, gram_min_std)
        self.posterior_head = GaussHead(2 * h, h, per_channel, gram_head_expansion, gram_init_std, gram_min_std)
        self.prior_mode = "sample"
        self._posterior_emb = None
        self._kl_terms = None

    def _transition(self, u: torch.Tensor, last: bool) -> torch.Tensor:
        uf = u.float()
        post = self._posterior_emb
        if post is not None:
            assert post.shape == uf.shape, (post.shape, uf.shape)
            mu_q, sd_q = self.posterior_head(torch.cat([uf, post.float()], dim=-1))
            eps = mu_q + sd_q * torch.randn_like(mu_q)
            if last:
                mu_p, sd_p = self.prior_head(uf)
                self._kl_terms = (mu_q, sd_q, mu_p, sd_p)
        else:
            mu_p, sd_p = self.prior_head(uf)
            eps = mu_p if self.prior_mode == "mean" else mu_p + sd_p * torch.randn_like(mu_p)
        return u + eps.to(u.dtype)

    def reasoning_step(self, *args, **kwargs):
        self.inner._h_transition = self._transition
        try:
            return super().reasoning_step(*args, **kwargs)
        finally:
            self.inner._h_transition = None

    def kl_loss(self, balance: float, free_bits: float) -> tuple[torch.Tensor, dict]:
        """(B,) per-sample KL loss of the last transition, consuming self._kl_terms: KL balancing (balance on the
        prior's side, 1 - balance on the posterior's), free bits per token on each side (no gradient below), the
        per-token KL summed over channels, then averaged over tokens and divided by hidden_size (per channel)."""
        mu_q, sd_q, mu_p, sd_p = self._kl_terms
        self._kl_terms = None
        kl_prior = gauss_kl(mu_q.detach(), sd_q.detach(), mu_p, sd_p).expand_as(mu_q).sum(-1)  # (B, N), trains p
        kl_post = gauss_kl(mu_q, sd_q, mu_p.detach(), sd_p.detach()).expand_as(mu_q).sum(-1)  # (B, N), trains q
        tok = balance * kl_prior.clamp(min=free_bits) + (1 - balance) * kl_post.clamp(min=free_bits)
        per_sample = tok.mean(1) / mu_q.shape[-1]
        stats = {
            "gram_kl_nats_per_token": kl_prior.detach().mean().item(),
            "gram_sd_q": sd_q.detach().mean().item(),
            "gram_sd_p": sd_p.detach().mean().item(),
            "gram_mu_gap": (mu_q - mu_p).detach().pow(2).mean().sqrt().item(),
        }
        return per_sample, stats


class _LossPlusKL(nn.Module):
    """The model's loss plus beta * the thinker's KL loss whenever the last reasoning step went through the
    posterior (training); unchanged otherwise (eval, prior passes)."""

    def __init__(self, inner: nn.Module, beta: float, balance: float, free_bits: float):
        super().__init__()
        self.inner = inner
        self.beta, self.balance, self.free_bits = beta, balance, free_bits
        self.thinker = None  # set by the owning model; not a submodule

    def forward(self, noise_pred, logits, sample):
        loss, loss_dict = self.inner(noise_pred, logits, sample)
        if self.thinker._kl_terms is not None:
            kl, stats = self.thinker.kl_loss(self.balance, self.free_bits)
            loss = loss + self.beta * kl.mean()
            loss_dict = {**loss_dict, "gram_kl_loss": kl.mean().item(), **stats}
        return loss, loss_dict

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("inner"), name)


class GramThinkerFrozenPainterACT(ThinkerFrozenPainterACT):
    """ACT thinker with a GramSpatialTRM. In training the posterior sees an encoding of the solved image (a fresh
    encoder of the condition encoder's architecture on (solved image in [0, 1], x_t), used only in training) and
    the loss is the painter's loss + beta * KL(q || p) at each supervision step's last transition (KL balancing,
    free bits). Inference (forward, sampling) uses the prior. With halt_label_clean, the halt head is labeled from
    the same step recomputed with eps = the prior mean (no sampling noise in the labels). Eval logs the callbacks
    twice: prior samples (the usual keys) and prior mean (keys with suffix _pmean)."""

    def __init__(self, *args, gram_beta: float, gram_kl_balance: float = 0.8, gram_free_bits: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.thinker, GramSpatialTRM):
            raise ValueError("mode thinker_act_gram needs thinker._target_=models.gram_thinkers.GramSpatialTRM")
        self.posterior_encoder = instantiate(kwargs["condition_encoder"])
        self.loss_fn = _LossPlusKL(self.loss_fn, gram_beta, gram_kl_balance, gram_free_bits)
        object.__setattr__(self.loss_fn, "thinker", self.thinker)
        self._pre_step = None

    def posterior_emb(self, sample: DataSample) -> torch.Tensor:
        """(B, N, hidden) encoding of the solved image (in [0, 1]) and x_t."""
        solved = self.painter.decode_for_eval(sample.images)
        return self.posterior_encoder(solved, x_noisy=sample.x_noisy, timesteps=sample.timesteps).enc_emb

    def reasoning_step(self, sample: DataSample, z_H: torch.Tensor, z_L: torch.Tensor):
        if not self.training:
            return super().reasoning_step(sample, z_H, z_L)
        self._pre_step = (sample, z_H, z_L)
        self.thinker._posterior_emb = self.posterior_emb(sample)
        try:
            return super().reasoning_step(sample, z_H, z_L)
        finally:
            self.thinker._posterior_emb = None

    @torch.no_grad()
    def _clean_step_loss(self, sample: DataSample, logits: torch.Tensor) -> torch.Tensor:
        """The step just taken, recomputed from the same carry with eps = the prior mean, then the parent's
        regularizer-free painter loss of those logits."""
        pre_sample, z_H, z_L = self._pre_step
        assert pre_sample is sample
        mode = self.thinker.prior_mode
        self.thinker.prior_mode = "mean"
        try:
            clean_logits, _, _ = self.thinker.reasoning_step(
                self._encode_condition(sample), z_H, z_L, sample.puzzle_id, timesteps=sample.timesteps
            )
        finally:
            self.thinker.prior_mode = mode
        return super()._clean_step_loss(sample, clean_logits)

    def eval_step(self, dataloader, accelerator, **kwargs) -> dict:
        result = super().eval_step(dataloader, accelerator, **kwargs)
        self.eval()
        self.thinker.prior_mode = "mean"
        try:
            for cb in self.eval_callbacks:
                result.update({f"{k}_pmean": v for k, v in cb(self, dataloader, accelerator, **kwargs).items()})
        finally:
            self.thinker.prior_mode = "sample"
        self.train()
        return result

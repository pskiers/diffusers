"""
datasets/data_sample.py — Standardised per-sample schema for TRM-diffusion.

Every dataset returns a DataSample (or a dict that can be losslessly
converted to one).  All fields are Optional; a dataset only fills the
fields it actually provides.  The schema is the single authoritative
record of what key names mean — add new fields here with a docstring,
never invent ad-hoc keys elsewhere.

Collation
---------
Use ``collate_data_samples`` as the DataLoader ``collate_fn``.  It stacks
tensors field-by-field and propagates None for any field that is absent in
every sample of the batch.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

import torch
from torch import Tensor


@dataclass
class DataSample:
    # ── Pixel observations ──────────────────────────────────────────────────
    images: Optional[Tensor] = None
    """Target image the diffusion model is trained to reconstruct.
    Shape: (C, H, W), float32 in [0, 1] (or latent when using a VAE encoder)."""

    spatial_conditions: Optional[Tensor] = None
    """Spatial (3-D) conditioning signal provided directly to the painter,
    e.g. a layout mask, depth map, or sketch.
    Shape: (C, H, W) — same spatial resolution as ``images``."""

    embedding_conditions: Optional[Tensor] = None
    """Sequence (2-D) conditioning signal for cross-attention style painters,
    e.g. object-feature embeddings from a CLEVR scene graph.
    Shape: (N, D) where N is the number of tokens and D the feature dim."""

    # ── Discrete structure / labels ─────────────────────────────────────────
    solution: Optional[Tensor] = None
    """Ground-truth discrete label sequence.  Shape: (N,), dtype long.
    For MNIST-Sudoku: 81 cell values in [0, 8] (0 = blank).
    Used by the CE loss and 'teacher condition' evaluation."""

    solution_mask: Optional[Tensor] = None
    """Boolean mask marking which positions in ``solution`` are *given* (i.e.
    should not be predicted / are ignored in the CE loss).
    Shape: (N,), dtype bool."""

    token_conditions: Optional[Tensor] = None
    """Discrete token sequence used as a conditioning input rather than a
    generation target — e.g. the painter-vocabulary-encoded puzzle tokens for
    an unconditional painter that receives the board state.
    Shape: (N,), dtype long."""

    # ── Diffusion runtime fields (populated by the training loop / model) ───
    x_noisy: Optional[Tensor] = None
    """Noisy version of ``images`` at a sampled timestep t, i.e. x_t.
    Shape matches ``images``: (C, H, W).
    Not stored in the raw dataset — computed during training by the model
    and injected into the sample before condition encoding.  This is always
    the painter's denoising input."""

    enc_x_noisy: Optional[Tensor] = None
    """Override for the noisy image passed to the condition encoder.
    When None (default) the encoder uses ``x_noisy``.
    Set by NoisyGuidancePredictor during the V0-equivalent pass (zeros) so
    the encoder sees no noisy information while the painter still denoises
    the real ``x_noisy``."""

    timesteps: Optional[Tensor] = None
    """Diffusion timestep t sampled for this example.  Shape: (), dtype long.
    Like ``x_noisy``, populated at training time rather than by the dataset."""

    # ── Identity / metadata ─────────────────────────────────────────────────
    puzzle_id: Optional[Tensor] = None
    """Integer identifier of the puzzle instance used by the thinker's
    puzzle-embedding layer.  Shape: (), dtype long."""

    # ── Attention masks ──────────────────────────────────────────────────────
    embedding_mask: Optional[Tensor] = None
    """Boolean (or float 0/1) mask for ``embedding_conditions`` tokens.
    True / 1.0 = real object slot, False / 0.0 = padding.
    Shape: (N,) matching the token axis of ``embedding_conditions``."""

    # ── Diffusion target (populated by the training loop / model) ────────────
    target: Optional[Tensor] = None
    """Diffusion target for the MSE loss — noise (epsilon) or x0, in
    latent space for VAE models.  Populated by _prepare_training_sample;
    never stored in the raw dataset."""

    # ── Dict-style access (backwards-compat with code that uses mb["key"]) ──

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key) and getattr(self, key) is not None

    def get(self, key: str, default=None):
        val = getattr(self, key, None)
        return val if val is not None else default

    def to(self, device) -> DataSample:
        """Return a new DataSample with all tensors moved to device."""
        kwargs = {}
        for f in fields(DataSample):
            val = getattr(self, f.name)
            kwargs[f.name] = val.to(device) if isinstance(val, Tensor) else val
        return DataSample(**kwargs)

    def slice(self, start: int, end: int) -> DataSample:
        """Return a new DataSample with every tensor field's batch dim sliced
        to [start:end] — e.g. to sub-chunk a dataloader batch before an
        expensive per-instance operation (best-of-N sampling) so the actual
        model batch size stays bounded regardless of the dataloader's."""
        kwargs = {}
        for f in fields(DataSample):
            val = getattr(self, f.name)
            kwargs[f.name] = val[start:end] if isinstance(val, Tensor) else val
        return DataSample(**kwargs)


# ── Collation ────────────────────────────────────────────────────────────────


def collate_data_samples(samples: list[DataSample]) -> DataSample:
    """Stack a list of DataSamples into a single batched DataSample.

    Fields that are None in every sample remain None in the batch.
    Fields present in only some samples raise ValueError — partial presence
    is a dataset bug, not a collation concern.
    """
    kwargs: dict = {}
    for f in fields(DataSample):
        values = [getattr(s, f.name) for s in samples]
        n_none = sum(v is None for v in values)
        if n_none == len(values):
            kwargs[f.name] = None
        elif n_none > 0:
            present = [i for i, v in enumerate(values) if v is not None]
            absent = [i for i, v in enumerate(values) if v is None]
            raise ValueError(
                f"DataSample field '{f.name}' is present in samples "
                f"{present} but absent in {absent}. "
                "All samples in a batch must agree on which fields are populated."
            )
        else:
            first = values[0]
            if isinstance(first, Tensor):
                kwargs[f.name] = torch.stack(values)
            else:
                kwargs[f.name] = torch.tensor(values)
    return DataSample(**kwargs)

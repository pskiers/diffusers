"""
Dataset adapter for discrete (token-level) Sokoban diffusion.

Wraps the existing SokobanDataset to emit flat token sequences [144] ∈ {1,...,7}
instead of bit-encoded image tensors. The token vocabulary matches the original
TRM convention: +1 offset so WALL=1, FLOOR=2, ..., PLAYER_ON_TARGET=7.
Token 0 is reserved for [MASK] in the diffusion process.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from sokoban.diffusion.sokoban_dataset_diffusion import SokobanDataset


class SokobanTokenDataset(Dataset):
    """Wraps SokobanDataset to output flat token sequences for discrete diffusion.

    Each item is a dict with:
        "tokens": [144] long tensor (board tokens 1-7, no MASK at train time)
    """

    def __init__(self, sokoban_dataset: SokobanDataset):
        self.dataset = sokoban_dataset

    def __len__(self) -> int:
        return len(self.dataset)

    @property
    def group_boundaries(self):
        return self.dataset.group_boundaries

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]
        board = item["target"]  # [12, 12] uint8
        tokens = torch.from_numpy(board.flatten()).long() + 1  # [144] ∈ {1,...,7}
        return {"tokens": tokens}

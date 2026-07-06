"""
MNISTSudokuSRMDataset — MNIST Sudoku dataset that exactly replicates the setup
from the SRM paper (Chrixtar et al., 2024).

Key design decisions matching the paper:
  * Puzzles from sudokus.npy  — 1 000 000 × 9 × 9 grids, values 1-9.
  * MNIST digit pool selected by top_5000_values.csv:
      for each digit class 0-9, pick the top `top_n` images sorted by
      classifier confidence (descending).  Only classes 1-9 ever appear in
      sudoku cells.
  * Train/test split: last `test_samples_num` (default 10 000) puzzles are test.
  * Cell size: 28×28 → grid size 252×252 (matches SRM UNet-256 input).
  * Given cells: sampled uniformly in `given_cells_range` for each puzzle.
    Training default (0, 80) mirrors the paper config; evaluation clips to ≥17.
  * Test split is deterministic (seeded by puzzle index).
  * Auto-download: if sudokus.npy / top_5000_values.csv are missing, the
    dataset downloads them from the SRM GitHub v1.0.0 release.

Return format (identical to MNISTSudokuDataset):
  "images"        — (1, 252, 252) float32 [0, 1]   full solved grid
  "conditions"    — (1, 252, 252) float32           given cells only; blanks = 0
  "solution"      — (81,) int64, values 0-8         digit - 1
  "puzzle_id"     — scalar int64
  "puzzle_tokens" — (81,) long, 1=blank, 2-10=digit 1-9
  "given_mask"    — (81,) bool, True where cell is given
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

BLANK_TOKEN  = 1
DIGIT_OFFSET = 2  # token 2 → digit 1 → class 0


def _download_srm_datasets(root_dir: str) -> None:
    """Download sudokus.npy and top_5000_values.csv from SRM GitHub release."""
    import urllib.request

    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)

    url = "https://github.com/Chrixtar/SRM/releases/download/v1.0.0/datasets.zip"
    zip_path = root / "datasets.zip"

    print(f"[MNISTSudokuSRMDataset] Downloading SRM datasets from {url} …")
    try:
        urllib.request.urlretrieve(url, zip_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download SRM datasets from {url}.\n"
            f"Download manually and place sudokus.npy and top_5000_values.csv in {root_dir}.\n"
            f"Error: {e}"
        ) from e

    print(f"[MNISTSudokuSRMDataset] Extracting {zip_path} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        # The zip may have a subdirectory; extract everything and find the files.
        zf.extractall(root)
    zip_path.unlink(missing_ok=True)

    # Some releases nest files inside a subfolder — flatten if needed.
    for fname in ("sudokus.npy", "top_5000_values.csv"):
        target = root / fname
        if not target.exists():
            candidates = list(root.rglob(fname))
            if candidates:
                candidates[0].rename(target)

    if not (root / "sudokus.npy").exists() or not (root / "top_5000_values.csv").exists():
        raise RuntimeError(
            "Download succeeded but could not locate sudokus.npy and/or "
            f"top_5000_values.csv under {root_dir}."
        )
    print("[MNISTSudokuSRMDataset] Download complete.")


class MNISTSudokuSRMDataset(Dataset):
    """MNIST Sudoku dataset matching the SRM paper setup.

    Args:
        root_dir:          Directory containing sudokus.npy, top_5000_values.csv,
                           and the MNIST raw files (or wherever torchvision will
                           download MNIST to).
        split:             "train" or "test".
        top_n:             Number of MNIST images per digit class to draw from
                           (sorted by classifier confidence). Default 1000.
        test_samples_num:  Last N sudokus reserved for test. Default 10 000.
        given_cells_range: (lo, hi) inclusive range for the number of given cells
                           per puzzle. A random value in [lo, hi] is picked for
                           each puzzle.  Default (0, 80) as in the paper.
        seed:              Base RNG seed.
        download:          If True (default), download missing files automatically.
    """

    CELL_SIZE        = 28
    GRID_CELLS       = 81
    DIGIT_OFFSET     = DIGIT_OFFSET
    BLANK_TOKEN      = BLANK_TOKEN

    def __init__(
        self,
        root_dir: str = "data/mnist_sudoku_srm",
        split: str = "train",
        top_n: int = 1000,
        test_samples_num: int = 10_000,
        given_cells_range: Sequence[int] = (0, 80),
        seed: int = 0,
        download: bool = True,
    ) -> None:
        super().__init__()
        assert split in ("train", "test"), f"split must be 'train' or 'test', got {split!r}"
        self.root_dir          = Path(root_dir)
        self.split             = split
        self.top_n             = top_n
        self.test_samples_num  = test_samples_num
        self.given_lo, self.given_hi = int(given_cells_range[0]), int(given_cells_range[1])
        self.seed              = seed
        self.is_train          = split == "train"

        # ── Check / download data ─────────────────────────────────────────────
        sudokus_path = self.root_dir / "sudokus.npy"
        csv_path     = self.root_dir / "top_5000_values.csv"
        if not sudokus_path.exists() or not csv_path.exists():
            if download:
                _download_srm_datasets(str(self.root_dir))
            else:
                raise FileNotFoundError(
                    f"sudokus.npy or top_5000_values.csv not found in {root_dir}. "
                    "Pass download=True to download automatically."
                )

        # ── Load sudoku grids ─────────────────────────────────────────────────
        all_grids = np.load(sudokus_path).astype(np.int32)  # (N, 9, 9), values 1-9
        N = len(all_grids)
        if split == "train":
            self._grids = all_grids[: N - test_samples_num]
        else:
            self._grids = all_grids[N - test_samples_num:]

        # ── Load MNIST images filtered by top_5000_values.csv ────────────────
        self._digit_arrays = self._load_digit_arrays(csv_path)

        self._rng = np.random.default_rng(seed)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_digit_arrays(self, csv_path: Path) -> dict[int, np.ndarray]:
        """Return {digit_class: (top_n, 28, 28) float32 in [0,1]} for classes 0-9."""
        import torchvision

        mnist_ds = torchvision.datasets.MNIST(
            root=str(self.root_dir),
            train=True,
            download=True,
        )

        top_df = pd.read_csv(csv_path)
        digit_arrays: dict[int, np.ndarray] = {}

        for label in range(10):
            label_df = top_df[top_df["label"] == label].copy()
            label_df.sort_values("confidence", ascending=False, inplace=True)
            label_df = label_df.iloc[: self.top_n]
            indices = label_df["image_index"].values

            all_class_imgs = mnist_ds.data[mnist_ds.targets == label]  # (M, 28, 28) uint8
            selected = all_class_imgs[indices].numpy().astype(np.float32) / 255.0
            assert len(selected) == self.top_n, (
                f"Expected {self.top_n} images for digit {label}, got {len(selected)}"
            )
            digit_arrays[label] = selected

        return digit_arrays

    def _render_grid(
        self,
        solution_grid: np.ndarray,  # (9, 9) int, values 1-9
        given_mask_2d: np.ndarray,  # (9, 9) bool, True = given (shown)
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Render full-solution and condition images.

        Returns:
            full_img  — (252, 252) float32 [0,1], every cell filled
            cond_img  — (252, 252) float32 [0,1], blank cells are black
        """
        cs = self.CELL_SIZE
        full_img = np.empty((9 * cs, 9 * cs), dtype=np.float32)
        cond_img = np.zeros((9 * cs, 9 * cs), dtype=np.float32)

        for r in range(9):
            for c in range(9):
                digit = int(solution_grid[r, c])  # 1-9
                candidates = self._digit_arrays[digit]
                idx = rng.integers(0, len(candidates))
                tile = candidates[idx]  # (28, 28) float32

                r0, c0 = r * cs, c * cs
                full_img[r0:r0+cs, c0:c0+cs] = tile
                if given_mask_2d[r, c]:
                    cond_img[r0:r0+cs, c0:c0+cs] = tile

        return full_img, cond_img

    def _sample_given_mask(self, solution_flat: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Randomly select given cells; return (81,) bool mask."""
        n_given = rng.integers(self.given_lo, self.given_hi + 1)
        given = np.zeros(81, dtype=bool)
        indices = rng.choice(81, size=n_given, replace=False)
        given[indices] = True
        return given

    # ── Dataset API ──────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._grids)

    def __getitem__(self, idx: int) -> dict:
        grid_2d = self._grids[idx]  # (9, 9), values 1-9

        # Deterministic for test; random for train.
        if not self.is_train:
            rng = np.random.default_rng(self.seed * 104729 + idx)
        else:
            rng = self._rng

        solution_flat = grid_2d.reshape(81)  # values 1-9

        # Given cell selection
        given_mask = self._sample_given_mask(solution_flat, rng)  # (81,) bool

        # Render images
        full_img, cond_img = self._render_grid(grid_2d, given_mask.reshape(9, 9), rng)

        # Tokens: 1=blank, 2-10 = digit 1-9
        puzzle_tokens = np.where(given_mask, solution_flat + 1, BLANK_TOKEN).astype(np.int64)

        # Solution class indices: digit 1-9 → class 0-8
        solution = (solution_flat - 1).astype(np.int64)  # (81,) in [0, 8]

        return {
            "images":        torch.from_numpy(full_img).unsqueeze(0),    # (1, 252, 252)
            "conditions":    torch.from_numpy(cond_img).unsqueeze(0),    # (1, 252, 252)
            "solution":      torch.from_numpy(solution),                  # (81,) int64
            "puzzle_id":     torch.tensor(idx, dtype=torch.int64),
            "puzzle_tokens": torch.from_numpy(puzzle_tokens),             # (81,) long
            "given_mask":    torch.from_numpy(given_mask),                # (81,) bool
        }

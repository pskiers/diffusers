"""
MNISTSudokuDataset – renders Sudoku puzzles as grids of MNIST digit images.

Each sample contains:
  images             – (1, 9*cell_size, 9*cell_size) float32 [0,1]
                         Complete solved sudoku rendered as MNIST digits.
  spatial_conditions – same shape; blank cells are filled with zeros (black image).
  token_conditions   – (81,) int64; puzzle token sequence (1=blank, 2-10=digit 1-9).
  solution           – (81,) int64; digit class indices 0-8 (digit 1→0, …, digit 9→8).
  solution_mask      – (81,) bool; True = given cell (visible in puzzle input).
  puzzle_id          – () int64; puzzle identifier.

Token convention (from sudoku_dataset.py):
  0   – PAD  (never appears in real puzzles)
  1   – blank cell
  2-10 – given/solution digit (2→digit 1, …, 10→digit 9)

The MNIST images are normalised to [0, 1] (original MNIST is already byte [0,255]).
If `downscale=True` (default False), MNIST images are downsampled from 28×28 to
14×14 before tiling, giving a 126×126 grid instead of 252×252.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from datasets.sudoku_dataset import SudokuDataset, PAD_ID
from datasets.data_sample import DataSample, collate_data_samples


class MNISTSudokuDataset(Dataset):
    """
    Args:
        sudoku_dir:  Directory produced by sudoku_dataset.convert_subset
                     (contains inputs.npy and labels.npy).
        mnist_root:  Root directory for the MNIST dataset (downloaded if absent).
        cell_size:   Side length in pixels of each digit cell.
                     Default 32 – gives a 288×288 grid (divisible by 16 for
                     4-stage UNet downsampling).
        mnist_split: "train" or "test" – which MNIST split to draw digits from.
        mask_given:  Whether to zero out given cells in the condition image.
        seed:        RNG seed for deterministic digit assignment.
        same_digit_images: if True, a given cell's condition render and its
                     solution render use the literal same MNIST bitmap
                     (same drawn instance), not just the same digit class.
    """

    # Digit 1..9 → class index 0..8
    DIGIT_OFFSET = 2   # token offset: token 2 → digit 1 → class 0

    def __init__(
        self,
        sudoku_dir: str,
        mnist_root: str = "data/mnist",
        cell_size: int = 32,
        mnist_split: str = "train",
        mask_given: bool = True,
        seed: int = 0,
        num_givens: Optional[int] = None,
        same_digit_images: bool = False,
    ):
        super().__init__()
        self.cell_size  = cell_size
        self.mask_given = mask_given
        self.num_givens = num_givens
        self.same_digit_images = same_digit_images
        self._given_seed = seed  # separate seed for given-cell selection

        # ── Load Sudoku data ──────────────────────────────────────────────────
        self.sudoku = SudokuDataset(sudoku_dir, mask_given=False)

        # ── Load MNIST ────────────────────────────────────────────────────────
        # Check for a pre-built numpy cache first (allows tests to inject fake
        # data without triggering a torchvision download).
        # Cache format: npz with keys "digit_1" … "digit_9", each (N, 28, 28) float32.
        cache_path = os.path.join(mnist_root, f"mnist_{mnist_split}_cache.npz")
        self._digit_imgs: dict[int, list[np.ndarray]] = {d: [] for d in range(1, 10)}
        if os.path.exists(cache_path):
            data = np.load(cache_path)
            for d in range(1, 10):
                key = f"digit_{d}"
                if key in data:
                    self._digit_imgs[d] = list(data[key].astype(np.float32))
        else:
            mnist_ds = datasets.MNIST(
                root=mnist_root,
                train=(mnist_split == "train"),
                download=True,
                transform=None,
            )
            for img, label in mnist_ds:
                if label == 0:
                    continue
                arr = np.array(img, dtype=np.float32) / 255.0  # [0,1], (28,28)
                self._digit_imgs[label].append(arr)

        # ── Resize MNIST tiles if cell_size != 28 ────────────────────────────
        if cell_size != 28:
            resized: dict[int, list[np.ndarray]] = {d: [] for d in range(1, 10)}
            for d, imgs in self._digit_imgs.items():
                t = torch.from_numpy(np.stack(imgs)).unsqueeze(1)   # (N,1,28,28)
                t = F.interpolate(t, size=(cell_size, cell_size), mode="bilinear", align_corners=False)
                resized[d] = [t[i, 0].numpy() for i in range(len(imgs))]
            self._digit_imgs = resized

        # Pre-build numpy arrays per digit for fast indexing
        self._digit_arrays = {
            d: np.stack(imgs) for d, imgs in self._digit_imgs.items()
        }
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.sudoku)

    def _render_grid(
        self,
        tokens: np.ndarray,
        given_mask: np.ndarray | None,
        chosen_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        tokens: (81,) int, values in {1,2,...,10}
                1 = blank, 2-10 = digit 1-9
        given_mask: if not None, cells where given_mask==True are blanked (zeros).
        chosen_indices: (81,) int, optional. If given, use chosen_indices[idx] as
            the MNIST instance index for cell idx instead of drawing a fresh
            random one — lets two _render_grid calls (e.g. condition and
            solution) render the SAME physical MNIST bitmap for a shared cell.
            Only meaningful at indices where tokens[idx] > 1.

        Returns (cell_size*9, cell_size*9) float32 numpy array.
        """
        cs = self.cell_size
        grid = np.zeros((9 * cs, 9 * cs), dtype=np.float32)
        for idx in range(81):
            tok = int(tokens[idx])
            row, col = divmod(idx, 9)
            r0, c0 = row * cs, col * cs

            if tok <= 1:
                # PAD or blank → leave black
                continue
            if given_mask is not None and given_mask[idx]:
                # Given cell in condition: black
                continue

            digit = tok - 1  # token 2 → digit 1, … token 10 → digit 9
            arr   = self._digit_arrays[digit]
            chosen_idx = int(chosen_indices[idx]) if chosen_indices is not None else self._rng.integers(len(arr))
            grid[r0:r0+cs, c0:c0+cs] = arr[chosen_idx]

        return grid

    def _sample_chosen_indices(self, tokens: np.ndarray) -> np.ndarray:
        """Pre-draw one MNIST instance index per cell (only meaningful where
        tokens[idx] > 1) — used by same_digit_images so the condition and
        solution renders agree on which physical bitmap to use per cell."""
        chosen = np.zeros(81, dtype=np.int64)
        for idx in range(81):
            tok = int(tokens[idx])
            if tok > 1:
                chosen[idx] = self._rng.integers(len(self._digit_arrays[tok - 1]))
        return chosen

    def __getitem__(self, idx: int) -> DataSample:
        sudoku_item = self.sudoku[idx]
        inputs_tok  = sudoku_item["inputs"].numpy()   # (81,) tokens
        labels_tok  = sudoku_item["labels"].numpy()   # (81,) tokens (solution, full)

        # The "labels" from SudokuDataset with mask_given=False are the full solutions.
        # Blank cells in inputs are token 1; given cells have real token 2-10.

        # ── Easy-mode override: randomly reveal num_givens cells ──────────────
        if self.num_givens is not None:
            # Use a per-item deterministic RNG so access order doesn't matter.
            item_rng = np.random.default_rng(self._given_seed * 104729 + idx)
            valid_cells = np.where((labels_tok >= 2) & (labels_tok <= 10))[0]
            n = min(self.num_givens, len(valid_cells))
            chosen = item_rng.choice(valid_cells, size=n, replace=False)
            inputs_tok = np.full(81, SudokuDataset.BLANK_TOKEN, dtype=labels_tok.dtype)
            inputs_tok[chosen] = labels_tok[chosen]

        # ── Full solved image ─────────────────────────────────────────────────
        # Use solution for every cell (so blank cells get their answer digit).
        full_tokens = labels_tok.copy()
        # same_digit_images: pre-draw per-cell MNIST instance indices from the
        # full solution and reuse them for the condition render too, so a
        # given cell shows the literal same bitmap in both — removes any
        # incidental visual mismatch between condition and target digits as
        # a possible confound, independent of solving/placement correctness.
        chosen_indices = self._sample_chosen_indices(full_tokens) if self.same_digit_images else None
        image_grid  = self._render_grid(full_tokens, given_mask=None, chosen_indices=chosen_indices)

        # ── Condition image ───────────────────────────────────────────────────
        # Given cells: use given token (from inputs). Blank cells: black.
        if self.mask_given:
            blank = (inputs_tok == PAD_ID) | (inputs_tok == SudokuDataset.BLANK_TOKEN)
            cond_grid = self._render_grid(inputs_tok, given_mask=blank, chosen_indices=chosen_indices)
        else:
            cond_grid = self._render_grid(inputs_tok, given_mask=None, chosen_indices=chosen_indices)

        # ── Solution class labels (0-based) ──────────────────────────────────
        # Tokens 2-10 → class 0-8; blank/pad cells → -100 (ignored in loss).
        # Given cells are NOT masked — CE loss trains on all 81 cells so the
        # thinker learns to carry correct information for given cells (needed
        # for the painter to reconstruct them).
        solution = np.full(81, fill_value=-100, dtype=np.int64)
        valid = (labels_tok >= 2) & (labels_tok <= 10)
        solution[valid] = labels_tok[valid] - self.DIGIT_OFFSET  # 0..8

        # given_mask: True where a cell is given (visible in puzzle input).
        # Used by accuracy metrics: cell acc evaluates blank cells only;
        # puzzle acc evaluates all cells.
        given_mask = (inputs_tok >= 2) & (inputs_tok <= 10)  # (81,) bool

        return DataSample(
            images=torch.from_numpy(image_grid).unsqueeze(0),          # (1,H,W)
            spatial_conditions=torch.from_numpy(cond_grid).unsqueeze(0),# (1,H,W)
            solution=torch.from_numpy(solution),                        # (81,)
            puzzle_id=sudoku_item["puzzle_id"],                         # scalar
            token_conditions=torch.from_numpy(inputs_tok.copy()),       # (81,) long
            solution_mask=torch.from_numpy(given_mask),                 # (81,) bool
        )

    collate_fn = staticmethod(collate_data_samples)


class MNISTSudokuScaledDataset(MNISTSudokuDataset):
    """
    Same puzzle/conditioning as MNISTSudokuDataset, but the diffusion target
    (``images``) is the solved grid randomly scaled down and pasted onto a
    black canvas at a random offset. ``spatial_conditions`` and every other
    field are unchanged, so the model is given the same puzzle hints but must
    paint the solution at an unknown scale and position.

    Args:
        scale_min, scale_max: the solved grid is resized to a random fraction
            of the full resolution drawn uniformly from [scale_min, scale_max]
            before being pasted at a random offset.
        center_condition: if True, also scale the puzzle condition image by
            the SAME random factor drawn for the target this sample, but
            paste it centered on the canvas — rather than left at full
            resolution (the default) or moved to the target's actual
            (matching) offset like MNISTSudokuAlignedScaledDataset. Isolates
            whether a scale mismatch between condition and target, on its
            own, is part of what breaks the pipeline — independent of the
            position mismatch mnist_sudoku_scaled already has, and without
            removing that position mismatch the way the aligned dataset does.
    """

    def __init__(
        self,
        *args,
        scale_min: float = 0.8,
        scale_max: float = 0.9,
        center_condition: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.center_condition = center_condition

    def __getitem__(self, idx: int) -> DataSample:
        sample = super().__getitem__(idx)
        images = sample.images  # (1, H, W)
        full_size = images.shape[-1]

        scale = self._rng.uniform(self.scale_min, self.scale_max)
        new_size = max(1, round(full_size * scale))
        resized = F.interpolate(
            images.unsqueeze(0), size=(new_size, new_size), mode="bilinear", align_corners=False
        ).squeeze(0)

        canvas = torch.zeros_like(images)
        max_off = full_size - new_size
        off_y = int(self._rng.integers(0, max_off + 1))
        off_x = int(self._rng.integers(0, max_off + 1))
        canvas[:, off_y : off_y + new_size, off_x : off_x + new_size] = resized

        updates = {"images": canvas}
        if self.center_condition:
            cond = sample.spatial_conditions
            cond_resized = F.interpolate(
                cond.unsqueeze(0), size=(new_size, new_size), mode="bilinear", align_corners=False
            ).squeeze(0)
            cond_canvas = torch.zeros_like(cond)
            center_off = max_off // 2
            cond_canvas[:, center_off : center_off + new_size, center_off : center_off + new_size] = cond_resized
            updates["spatial_conditions"] = cond_canvas

        return dataclasses.replace(sample, **updates)


class MNISTSudokuAlignedScaledDataset(MNISTSudokuDataset):
    """
    Diagnostic variant of MNISTSudokuScaledDataset: applies the SAME random
    scale+offset transform to BOTH the diffusion target (``images``) and the
    puzzle condition (``spatial_conditions``), instead of only the target.

    Isolates one variable from mnist_sudoku_scaled: whether random scale/
    offset variety by itself breaks the pipeline, independent of the puzzle-
    vs-target coordinate mismatch (mnist_sudoku_scaled keeps the condition at
    a fixed canonical layout while only the target moves). Here both move
    together, so the task reduces to "solve sudoku, then paint the whole
    thing at some scale/offset" — structurally close to plain mnist_sudoku,
    just rigidly transformed as a whole, with no puzzle/target mismatch to
    resolve.

    Args:
        scale_min, scale_max: the solved grid *and* the puzzle condition are
            both resized to the same random fraction, drawn uniformly from
            [scale_min, scale_max], and pasted at the same random offset.
    """

    def __init__(self, *args, scale_min: float = 0.8, scale_max: float = 0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale_min = scale_min
        self.scale_max = scale_max

    @staticmethod
    def _scale_and_offset(x: torch.Tensor, new_size: int, off_y: int, off_x: int) -> torch.Tensor:
        resized = F.interpolate(
            x.unsqueeze(0), size=(new_size, new_size), mode="bilinear", align_corners=False
        ).squeeze(0)
        canvas = torch.zeros_like(x)
        canvas[:, off_y : off_y + new_size, off_x : off_x + new_size] = resized
        return canvas

    def __getitem__(self, idx: int) -> DataSample:
        sample = super().__getitem__(idx)
        full_size = sample.images.shape[-1]

        scale = self._rng.uniform(self.scale_min, self.scale_max)
        new_size = max(1, round(full_size * scale))
        max_off = full_size - new_size
        off_y = int(self._rng.integers(0, max_off + 1))
        off_x = int(self._rng.integers(0, max_off + 1))

        images = self._scale_and_offset(sample.images, new_size, off_y, off_x)
        spatial_conditions = self._scale_and_offset(sample.spatial_conditions, new_size, off_y, off_x)
        return dataclasses.replace(sample, images=images, spatial_conditions=spatial_conditions)


def get_solution_tokens(solution: torch.Tensor) -> torch.Tensor:
    """Raw solution (0-8) → full token grid (2-10, no blanks) for painter stage."""
    return solution.clamp(min=0) + 2

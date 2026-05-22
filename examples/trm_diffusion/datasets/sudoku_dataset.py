from typing import List, Optional

import os
import csv
import json
import numpy as np
import torch
from torch.utils.data import Dataset

import pydantic
import numpy as np
from pydantic import BaseModel
from tqdm import tqdm
from huggingface_hub import hf_hub_download

try:
    from argdantic import ArgParser
except ImportError:
    ArgParser = None
    print("WARNING: argdantic not found, CLI will be unavailable. Install with `pip install argdantic`.")


# Global list mapping each dihedral transform id to its inverse.
# Index corresponds to the original tid, and the value is its inverse.
DIHEDRAL_INVERSE = [0, 3, 2, 1, 4, 5, 6, 7]


class PuzzleDatasetMetadata(pydantic.BaseModel):
    pad_id: int
    ignore_label_id: Optional[int]
    blank_identifier_id: int
    vocab_size: int
    seq_len: int
    num_puzzle_identifiers: int
    total_groups: int
    mean_puzzle_examples: float
    total_puzzles: int
    sets: List[str]


def dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    """8 dihedral symmetries by rotate, flip and mirror"""

    if tid == 0:
        return arr  # identity
    elif tid == 1:
        return np.rot90(arr, k=1)
    elif tid == 2:
        return np.rot90(arr, k=2)
    elif tid == 3:
        return np.rot90(arr, k=3)
    elif tid == 4:
        return np.fliplr(arr)       # horizontal flip
    elif tid == 5:
        return np.flipud(arr)       # vertical flip
    elif tid == 6:
        return arr.T                # transpose (reflection along main diagonal)
    elif tid == 7:
        return np.fliplr(np.rot90(arr, k=1))  # anti-diagonal reflection
    else:
        return arr


def inverse_dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    return dihedral_transform(arr, DIHEDRAL_INVERSE[tid])




class DataProcessConfig(BaseModel):
    source_repo: str = "sapientinc/sudoku-extreme"
    output_dir: str = "data/sudoku-extreme-full"

    subsample_size: Optional[int] = None
    min_difficulty: Optional[int] = None
    num_aug: int = 0


def shuffle_sudoku(board: np.ndarray, solution: np.ndarray):
    # Create a random digit mapping: a permutation of 1..9, with zero (blank) unchanged
    digit_map = np.pad(np.random.permutation(np.arange(1, 10)), (1, 0))

    # Randomly decide whether to transpose.
    transpose_flag = np.random.rand() < 0.5

    # Generate a valid row permutation:
    # - Shuffle the 3 bands (each band = 3 rows) and for each band, shuffle its 3 rows.
    bands = np.random.permutation(3)
    row_perm = np.concatenate([b * 3 + np.random.permutation(3) for b in bands])

    # Similarly for columns (stacks).
    stacks = np.random.permutation(3)
    col_perm = np.concatenate([s * 3 + np.random.permutation(3) for s in stacks])

    # Build an 81->81 mapping. For each new cell at (i, j)
    # (row index = i // 9, col index = i % 9),
    # its value comes from old row = row_perm[i//9] and old col = col_perm[i%9].
    mapping = np.array([row_perm[i // 9] * 9 + col_perm[i % 9] for i in range(81)])

    def apply_transformation(x: np.ndarray) -> np.ndarray:
        # Apply transpose flag
        if transpose_flag:
            x = x.T
        # Apply the position mapping.
        new_board = x.flatten()[mapping].reshape(9, 9).copy()
        # Apply digit mapping
        return digit_map[new_board]

    return apply_transformation(board), apply_transformation(solution)


def convert_subset(set_name: str, config: DataProcessConfig):
    # Read CSV
    inputs = []
    labels = []

    with open(hf_hub_download(config.source_repo, f"{set_name}.csv", repo_type="dataset"), newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for source, q, a, rating in reader:
            if (config.min_difficulty is None) or (int(rating) >= config.min_difficulty):
                assert len(q) == 81 and len(a) == 81

                inputs.append(np.frombuffer(q.replace('.', '0').encode(), dtype=np.uint8).reshape(9, 9) - ord('0'))
                labels.append(np.frombuffer(a.encode(), dtype=np.uint8).reshape(9, 9) - ord('0'))

    # If subsample_size is specified for the training set,
    # randomly sample the desired number of examples.
    if set_name == "train" and config.subsample_size is not None:
        total_samples = len(inputs)
        if config.subsample_size < total_samples:
            indices = np.random.choice(total_samples, size=config.subsample_size, replace=False)
            inputs = [inputs[i] for i in indices]
            labels = [labels[i] for i in indices]

    # Generate dataset
    num_augments = config.num_aug if set_name == "train" else 0

    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    puzzle_id = 0
    example_id = 0

    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)

    for orig_inp, orig_out in zip(tqdm(inputs), labels):
        for aug_idx in range(1 + num_augments):
            # First index is not augmented
            if aug_idx == 0:
                inp, out = orig_inp, orig_out
            else:
                inp, out = shuffle_sudoku(orig_inp, orig_out)

            # Push puzzle (only single example)
            results["inputs"].append(inp)
            results["labels"].append(out)
            example_id += 1
            puzzle_id += 1

            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(0)

        # Push group
        results["group_indices"].append(puzzle_id)

    # To Numpy
    def _seq_to_numpy(seq):
        arr = np.concatenate(seq).reshape(len(seq), -1)

        assert np.all((arr >= 0) & (arr <= 9))
        return arr + 1

    results = {
        "inputs": _seq_to_numpy(results["inputs"]),
        "labels": _seq_to_numpy(results["labels"]),

        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    # Metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=81,
        vocab_size=10 + 1,  # PAD + "0" ... "9"
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1,
        total_puzzles=len(results["group_indices"]) - 1,
        sets=["all"]
    )

    # Save metadata as JSON.
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)

    # Save data
    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)

    # Save IDs mapping (for visualization only)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)


# ---------------------------------------------------------------------------
# PAD / IGNORE label constants (matches original repo's conventions)
# ---------------------------------------------------------------------------
PAD_ID           = 0
IGNORE_LABEL_ID  = -100  # standard PyTorch cross-entropy ignore index


class SudokuDataset(Dataset):
    """
    PyTorch Dataset that loads pre-processed Sudoku data saved by convert_subset.

    Directory layout expected::

        data_dir/
            inputs.npy          # int32 (N, 81)  – puzzle tokens  (0=pad, 1=blank, 2-10=digits 1-9)
            labels.npy          # int32 (N, 81)  – solution tokens
            [identifiers.npy]   # int32 (N,)     – puzzle IDs (optional, zeros if absent)

    Token convention (same as convert_subset output, shifted by +1 so that
    digit '0'/blank becomes 1 and digits 1-9 become 2-10):
        0   – PAD (used for out-of-bounds padding)
        1   – blank cell in the input puzzle
        2–10 – given / solution digits 1–9

    Returns dicts with keys:
        "inputs"  – (81,) long tensor
        "labels"  – (81,) long tensor; given cells are set to IGNORE_LABEL_ID
                    so the loss only penalises blank-cell predictions
    """

    def __init__(self, data_dir: str, mask_given: bool = True):
        """
        Args:
            data_dir:    Path to the directory produced by convert_subset.
            mask_given:  If True (default), replace label entries for cells
                         that were already given in the input with
                         IGNORE_LABEL_ID so the loss ignores them.
        """
        self.mask_given = mask_given

        inputs_path = os.path.join(data_dir, "inputs.npy")
        labels_path = os.path.join(data_dir, "labels.npy")

        if not os.path.exists(inputs_path):
            # Try the all__* naming used by older convert_subset versions
            inputs_path = os.path.join(data_dir, "all__inputs.npy")
            labels_path = os.path.join(data_dir, "all__labels.npy")

        self.inputs  = np.load(inputs_path).astype(np.int64)   # (N, 81)
        self.labels  = np.load(labels_path).astype(np.int64)   # (N, 81)

        id_path = os.path.join(data_dir, "all__puzzle_identifiers.npy")
        if os.path.exists(id_path):
            self.identifiers = np.load(id_path).astype(np.int64)
        else:
            self.identifiers = np.zeros(len(self.inputs), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        inputs  = torch.from_numpy(self.inputs[idx].copy())   # (81,)
        labels  = torch.from_numpy(self.labels[idx].copy())   # (81,)

        if self.mask_given:
            # Cells that are already provided in the puzzle (not blank, not pad)
            # should not contribute to the loss.
            given = (inputs != PAD_ID) & (inputs != SudokuDataset.BLANK_TOKEN)
            labels = labels.clone()
            labels[given] = IGNORE_LABEL_ID

        puzzle_id = torch.tensor(self.identifiers[idx], dtype=torch.int64)
        return {"inputs": inputs, "labels": labels, "puzzle_id": puzzle_id}

    # Token ID for blank cells (class constant for convenience)
    BLANK_TOKEN: int = 1


if ArgParser is not None:
    cli = ArgParser()

    @cli.command(singleton=True)
    def preprocess_data(config: DataProcessConfig):
        convert_subset("train", config)
        convert_subset("test", config)

    if __name__ == "__main__":
        cli()
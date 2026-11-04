# ABOUTME: Tests MNISTSudokuDataset's condition_is_solution switch on the real EXTREME test split.
# ABOUTME: With it the condition image is the solved image; every other field matches the default dataset.
import os
import sys

import pytest
import torch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from datasets.mnist_sudoku_dataset import MNISTSudokuDataset  # noqa: E402


SUDOKU_DIR = "data/sudoku-extreme-1k-aug-1000/test"

pytestmark = pytest.mark.skipif(not os.path.isdir(SUDOKU_DIR), reason="needs the Sudoku EXTREME data")


def _ds(**kw):
    return MNISTSudokuDataset(SUDOKU_DIR, mnist_root="data/mnist", cell_size=16, mnist_split="test", seed=0, **kw)


def test_condition_is_solution():
    default, seex0 = _ds(), _ds(condition_is_solution=True)
    for idx in range(4):
        a, b = default[idx], seex0[idx]
        assert b.spatial_conditions.shape == (1, 144, 144)
        assert torch.equal(b.spatial_conditions, b.images)
        for field in ("images", "solution", "solution_mask", "token_conditions"):
            assert torch.equal(getattr(a, field), getattr(b, field)), field
        # the default condition hides the blanks, so it differs from the solved image
        assert not torch.equal(a.spatial_conditions, a.images)
        assert 0 < int(a.solution_mask.sum()) < 81

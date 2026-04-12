"""
Tests for MNIST Sudoku dataset and Ratatouille model variants.

Run:
    pytest test_mnist_sudoku.py -v
    pytest test_mnist_sudoku.py -v -k dataset   # dataset tests only
    pytest test_mnist_sudoku.py -v -k models    # model shape tests only
"""

import os
import shutil
import tempfile

import numpy as np
import pytest
import torch

from sudoku_dataset import PAD_ID, SudokuDataset

# ── Helpers ─────────────────────────────────────────────────────────────────────

CELL_SIZE    = 8   # tiny cell for fast tests
PAINTER_SIZE = 9 * CELL_SIZE  # 72


def make_sudoku_files(n: int, data_dir: str):
    """Write minimal inputs.npy / labels.npy compatible with SudokuDataset."""
    rng = np.random.default_rng(42)
    labels = rng.integers(2, 11, size=(n, 81)).astype(np.int32)
    inputs = labels.copy()
    blank  = rng.random((n, 81)) < 0.5
    inputs[blank] = 1
    np.save(os.path.join(data_dir, "inputs.npy"), inputs)
    np.save(os.path.join(data_dir, "labels.npy"), labels)


class FakeMNIST:
    """Stand-in for torchvision MNIST: returns random (28,28) PIL-like arrays."""

    def __init__(self, n=200):
        rng = np.random.default_rng(7)
        self._data  = [(rng.integers(0, 256, (28, 28), dtype=np.uint8),
                        int(rng.integers(1, 10))) for _ in range(n)]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)


# ── Dataset tests ────────────────────────────────────────────────────────────────

class TestMNISTSudokuDataset:

    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        make_sudoku_files(10, self.tmp)

        # Patch torchvision.datasets.MNIST so we don't download anything
        import unittest.mock as mock
        import PIL.Image
        from mnist_sudoku_dataset import MNISTSudokuDataset

        fake_mnist = FakeMNIST(200)
        pil_items  = [(PIL.Image.fromarray(img), lbl) for img, lbl in fake_mnist]

        with mock.patch("torchvision.datasets.MNIST", return_value=pil_items):
            self.ds = MNISTSudokuDataset(
                sudoku_dir=self.tmp,
                mnist_root=self.tmp,
                cell_size=CELL_SIZE,
                mnist_split="train",
                mask_given=True,
            )

    def teardown_method(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_length(self):
        assert len(self.ds) == 10

    def test_keys(self):
        item = self.ds[0]
        assert {"images", "conditions", "solution", "puzzle_id"} == set(item.keys())

    def test_image_shape(self):
        item = self.ds[0]
        assert item["images"].shape    == (1, PAINTER_SIZE, PAINTER_SIZE)
        assert item["conditions"].shape == (1, PAINTER_SIZE, PAINTER_SIZE)

    def test_solution_shape(self):
        assert self.ds[0]["solution"].shape == (81,)

    def test_image_range(self):
        item = self.ds[0]
        assert item["images"].min()    >= 0.0
        assert item["images"].max()    <= 1.0
        assert item["conditions"].min() >= 0.0
        assert item["conditions"].max() <= 1.0

    def test_solution_values(self):
        # Values should be in 0..8 or -100 (ignore)
        sol = self.ds[0]["solution"]
        valid = (sol == -100) | ((sol >= 0) & (sol <= 8))
        assert valid.all(), f"Unexpected solution values: {sol[~valid]}"

    def test_condition_has_blank_cells(self):
        # At least some cells should be black (blank) in the condition
        item = self.ds[0]
        assert (item["conditions"] == 0).any()

    def test_dtypes(self):
        item = self.ds[0]
        assert item["images"].dtype    == torch.float32
        assert item["conditions"].dtype == torch.float32
        assert item["solution"].dtype  == torch.int64

    def test_puzzle_id_dtype(self):
        assert self.ds[0]["puzzle_id"].dtype == torch.int64


# ── Model shape tests ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model_inputs():
    B = 2
    return {
        "noisy":  torch.randn(B, 1, PAINTER_SIZE, PAINTER_SIZE),
        "t":      torch.randint(0, 1000, (B,)),
        "cond":   torch.randn(B, 1, PAINTER_SIZE, PAINTER_SIZE),
        "B":      B,
    }


def _check_forward(model, inp, expect_logits: bool, num_classes: int = 9):
    model.eval()
    B = inp["B"]
    with torch.no_grad():
        noise_pred, logits = model(inp["noisy"], inp["t"], inp["cond"])
    assert noise_pred.shape == (B, 1, PAINTER_SIZE, PAINTER_SIZE), \
        f"Bad noise_pred shape: {noise_pred.shape}"
    if expect_logits:
        assert logits is not None, "Expected sudoku logits but got None"
        assert logits.shape == (B, 81, num_classes), f"Bad logits shape: {logits.shape}"
    else:
        assert logits is None, f"Expected no logits but got shape {logits.shape if logits is not None else None}"


class TestMNISTRatatouilleShapes:

    @pytest.fixture(autouse=True)
    def _inp(self, model_inputs):
        self.inp = model_inputs

    def _make_v0(self):
        from mnist_sudoku_models import MNISTRatatouilleV0
        return MNISTRatatouilleV0(
            painter_size=PAINTER_SIZE,
            cell_size=CELL_SIZE,
            thinker_hidden=16,
            thinker_layers=2,
            bridge_channels=4,
            painter_channels=(16, 32),
        )

    def _make_v1(self):
        from mnist_sudoku_models import MNISTRatatouilleV1
        # cell_size=8 → n_halvings=3 → 72/8=9 ✓
        return MNISTRatatouilleV1(
            painter_size=PAINTER_SIZE,
            cell_size=CELL_SIZE,
            enc_out_channels=8,
            bridge_channels=4,
            thinker_hidden=16,
            thinker_layers=2,
            painter_channels=(16, 32),
        )

    def _make_v2(self):
        from mnist_sudoku_models import MNISTRatatouilleV2
        return MNISTRatatouilleV2(
            painter_size=PAINTER_SIZE,
            cell_size=CELL_SIZE,
            enc_out_channels=8,
            bridge_channels=4,
            thinker_hidden=16,
            thinker_layers=2,
            painter_channels=(16, 32),
        )

    def _make_v3(self):
        from mnist_sudoku_models import MNISTRatatouilleV3
        return MNISTRatatouilleV3(
            painter_size=PAINTER_SIZE,
            cell_size=CELL_SIZE,
            enc_out_channels=8,
            thinker_out_channels=32,
            bridge_channels=4,
            thinker_hidden=16,
            thinker_layers=2,
            painter_channels=(16, 32),
        )

    def _make_v4(self):
        from mnist_sudoku_models import MNISTRatatouilleV4
        # compression_factor=4 → 72/4=18×18 thinker grid (PAINTER_SIZE=72)
        return MNISTRatatouilleV4(
            painter_size=PAINTER_SIZE,
            compression_factor=4,
            enc_out_channels=8,
            thinker_out_channels=32,
            bridge_channels=4,
            thinker_hidden=16,
            thinker_layers=2,
            painter_channels=(16, 32),
        )

    def test_v0_shapes(self):
        _check_forward(self._make_v0(), self.inp, expect_logits=True)

    def test_v1_shapes(self):
        _check_forward(self._make_v1(), self.inp, expect_logits=True)

    def test_v2_shapes(self):
        _check_forward(self._make_v2(), self.inp, expect_logits=False)

    def test_v3_shapes(self):
        _check_forward(self._make_v3(), self.inp, expect_logits=False)

    def test_v4_shapes(self):
        _check_forward(self._make_v4(), self.inp, expect_logits=False)

    def test_v0_param_count_positive(self):
        model = self._make_v0()
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) > 0

    def test_v1_gradients_flow(self):
        from mnist_sudoku_models import MNISTRatatouilleV1
        model = self._make_v1()
        model.train()
        inp = self.inp
        noise_pred, logits = model(inp["noisy"], inp["t"], inp["cond"])
        loss = noise_pred.mean()
        if logits is not None:
            loss = loss + logits.mean()
        loss.backward()
        no_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        assert not no_grad, f"No grad for: {no_grad}"

    def test_bridge_upsamples_to_painter_size(self):
        """Bridge output must match painter resolution."""
        from mnist_sudoku_models import MNISTRatatouilleV3
        model = self._make_v3()
        model.eval()
        with torch.no_grad():
            bridge_out, _ = model.thinker_forward(self.inp["cond"])
        assert bridge_out.shape[-1] == PAINTER_SIZE
        assert bridge_out.shape[-2] == PAINTER_SIZE

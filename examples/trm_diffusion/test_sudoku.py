"""
Tests for SudokuTRM model and training pipeline.

Run:
    pytest test_sudoku.py -v
    pytest test_sudoku.py -v -k training   # training only
    pytest test_sudoku.py -v -k inference  # inference only
"""

import os
import shutil
import tempfile

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from sudoku_dataset import SudokuDataset, IGNORE_LABEL_ID, PAD_ID
from sudoku_models import SudokuTRM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_dummy_data(n: int, data_dir: str):
    """Write minimal inputs/labels .npy files that SudokuDataset can load."""
    rng = np.random.default_rng(0)
    # Labels: all cells filled with digits 2-10
    labels  = rng.integers(2, 11, size=(n, 81)).astype(np.int64)
    # Inputs: same as labels but ~60% of cells blanked (token 1)
    inputs  = labels.copy()
    blank   = rng.random((n, 81)) < 0.6
    inputs[blank] = 1
    np.save(os.path.join(data_dir, "inputs.npy"), inputs.astype(np.int32))
    np.save(os.path.join(data_dir, "labels.npy"), labels.astype(np.int32))


def small_model(**kwargs) -> SudokuTRM:
    defaults = dict(
        vocab_size=11,
        seq_len=81,
        d_model=32,
        n_heads=2,
        n_layers=1,
        L_cycles=2,
        H_cycles=2,
        n_sup=2,
        dropout=0.0,
    )
    defaults.update(kwargs)
    return SudokuTRM(**defaults)


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

class TestSudokuDataset:

    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        make_dummy_data(20, self.tmp)
        self.ds = SudokuDataset(self.tmp, mask_given=True)

    def teardown_method(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_length(self):
        assert len(self.ds) == 20

    def test_keys(self):
        item = self.ds[0]
        assert set(item.keys()) == {"inputs", "labels"}

    def test_shapes(self):
        item = self.ds[0]
        assert item["inputs"].shape  == (81,)
        assert item["labels"].shape  == (81,)

    def test_dtypes(self):
        item = self.ds[0]
        assert item["inputs"].dtype  == torch.int64
        assert item["labels"].dtype  == torch.int64

    def test_inputs_valid_tokens(self):
        # inputs should only contain: PAD(0), blank(1), digits(2-10)
        for i in range(min(len(self.ds), 5)):
            inp = self.ds[i]["inputs"]
            assert inp.min() >= 0 and inp.max() <= 10, \
                f"Sample {i}: unexpected token value in inputs"

    def test_labels_valid_tokens(self):
        # labels contain digits 2-10 OR IGNORE_LABEL_ID (-100)
        for i in range(min(len(self.ds), 5)):
            lbl = self.ds[i]["labels"]
            valid = (lbl == IGNORE_LABEL_ID) | ((lbl >= 2) & (lbl <= 10))
            assert valid.all(), f"Sample {i}: unexpected value in labels"

    def test_given_cells_masked(self):
        # Every cell that is a given digit in the input must be IGNORE_LABEL_ID in labels
        for i in range(min(len(self.ds), 5)):
            item  = self.ds[i]
            given = (item["inputs"] != PAD_ID) & (item["inputs"] != SudokuDataset.BLANK_TOKEN)
            assert (item["labels"][given] == IGNORE_LABEL_ID).all(), \
                f"Sample {i}: given cell not masked in labels"

    def test_blank_cells_have_labels(self):
        # At least some blank cells should have a real label
        for i in range(min(len(self.ds), 5)):
            item  = self.ds[i]
            blank = item["inputs"] == SudokuDataset.BLANK_TOKEN
            assert (item["labels"][blank] != IGNORE_LABEL_ID).any(), \
                f"Sample {i}: no blank cell has a real label"

    def test_no_mask_mode(self):
        ds_unmasked = SudokuDataset(self.tmp, mask_given=False)
        item = ds_unmasked[0]
        # Without masking, labels should NOT contain IGNORE_LABEL_ID
        assert (item["labels"] != IGNORE_LABEL_ID).all()


# ---------------------------------------------------------------------------
# Model shape / forward tests
# ---------------------------------------------------------------------------

class TestSudokuTRMShapes:

    def setup_method(self):
        self.model = small_model()

    def test_parameter_count_positive(self):
        assert self.model.count_parameters() > 0

    def test_embed_shape(self):
        x = torch.randint(1, 11, (4, 81))
        emb = self.model.embed(x)
        assert emb.shape == (4, 81, 32)

    def test_initial_states_shape(self):
        z_H, z_L = self.model.get_initial_states(4)
        assert z_H.shape == (4, 81, 32)
        assert z_L.shape == (4, 81, 32)

    def test_initial_states_zero(self):
        z_H, z_L = self.model.get_initial_states(3)
        assert z_H.abs().max() == 0.0
        assert z_L.abs().max() == 0.0

    def test_reasoning_step_shapes(self):
        B = 3
        inputs   = torch.randint(1, 11, (B, 81))
        emb      = self.model.embed(inputs)
        z_H, z_L = self.model.get_initial_states(B)

        logits, z_H_new, z_L_new = self.model.reasoning_step(emb, z_H, z_L)

        assert logits.shape   == (B, 81, 11), f"logits: {logits.shape}"
        assert z_H_new.shape  == (B, 81, 32)
        assert z_L_new.shape  == (B, 81, 32)

    def test_reasoning_step_states_are_detached(self):
        B = 2
        inputs   = torch.randint(1, 11, (B, 81))
        emb      = self.model.embed(inputs)
        z_H, z_L = self.model.get_initial_states(B)
        _, z_H_new, z_L_new = self.model.reasoning_step(emb, z_H, z_L)
        assert not z_H_new.requires_grad
        assert not z_L_new.requires_grad

    def test_predict_shape(self):
        inputs = torch.randint(1, 11, (5, 81))
        logits = self.model.predict(inputs)
        assert logits.shape == (5, 81, 11)

    def test_predict_no_grad_context(self):
        # predict() is wrapped in @torch.no_grad, so output should not have grad
        inputs = torch.randint(1, 11, (2, 81))
        logits = self.model.predict(inputs)
        assert not logits.requires_grad

    def test_n_sup_override_in_predict(self):
        inputs = torch.randint(1, 11, (2, 81))
        # Should not raise; different n_sup values produce same shape
        for n in [1, 2, 4]:
            out = self.model.predict(inputs, n_sup=n)
            assert out.shape == (2, 81, 11)


# ---------------------------------------------------------------------------
# Training smoke tests
# ---------------------------------------------------------------------------

class TestSudokuTRMTraining:

    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        make_dummy_data(16, self.tmp)
        self.ds = SudokuDataset(self.tmp, mask_given=True)

    def teardown_method(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _get_batch(self, bsz=4):
        indices = torch.randint(len(self.ds), (bsz,))
        inputs  = torch.stack([self.ds[i]["inputs"] for i in indices])
        labels  = torch.stack([self.ds[i]["labels"] for i in indices])
        return {"inputs": inputs, "labels": labels}

    def test_loss_is_finite(self):
        model = small_model()
        model.train()
        batch  = self._get_batch()
        inputs = batch["inputs"]
        labels = batch["labels"]
        emb    = model.embed(inputs)
        z_H, z_L = model.get_initial_states(inputs.shape[0])
        logits, _, _ = model.reasoning_step(emb, z_H, z_L)
        loss = F.cross_entropy(
            logits.view(-1, model.vocab_size), labels.view(-1),
            ignore_index=IGNORE_LABEL_ID,
        )
        assert torch.isfinite(loss), f"Loss not finite: {loss.item()}"

    def test_loss_decreases_over_steps(self):
        """Verify that the model can actually learn on a tiny batch (overfit check)."""
        model = small_model()
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        batch     = self._get_batch(bsz=8)
        inputs    = batch["inputs"]
        labels    = batch["labels"]

        losses = []
        for _ in range(20):
            emb      = model.embed(inputs)
            z_H, z_L = model.get_initial_states(inputs.shape[0])
            total_loss = None
            for _ in range(model.n_sup):
                logits, z_H, z_L = model.reasoning_step(emb, z_H, z_L)
                step_loss = F.cross_entropy(
                    logits.view(-1, model.vocab_size), labels.view(-1),
                    ignore_index=IGNORE_LABEL_ID,
                )
                total_loss = step_loss if total_loss is None else total_loss + step_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            losses.append(total_loss.item())

        assert losses[-1] < losses[0], \
            f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"

    def test_gradients_flow(self):
        """All trainable parameters should receive gradients after a backward pass."""
        model = small_model()
        model.train()
        batch  = self._get_batch()
        inputs = batch["inputs"]
        labels = batch["labels"]
        emb    = model.embed(inputs)
        z_H, z_L = model.get_initial_states(inputs.shape[0])
        logits, _, _ = model.reasoning_step(emb, z_H, z_L)
        loss = F.cross_entropy(
            logits.view(-1, model.vocab_size), labels.view(-1),
            ignore_index=IGNORE_LABEL_ID,
        )
        loss.backward()

        no_grad_params = [
            name for name, p in model.named_parameters()
            if p.requires_grad and p.grad is None
        ]
        assert not no_grad_params, \
            f"Parameters without gradients: {no_grad_params}"

    def test_n_sup_loop_accumulates_loss(self):
        """Sum of per-step losses (n_sup>1) should differ from a single-step loss."""
        model  = small_model(n_sup=3)
        model.train()
        batch  = self._get_batch()
        inputs = batch["inputs"]
        labels = batch["labels"]

        def run(n_sup):
            emb      = model.embed(inputs)
            z_H, z_L = model.get_initial_states(inputs.shape[0])
            total    = None
            for _ in range(n_sup):
                logits, z_H, z_L = model.reasoning_step(emb, z_H, z_L)
                l = F.cross_entropy(
                    logits.view(-1, model.vocab_size), labels.view(-1),
                    ignore_index=IGNORE_LABEL_ID,
                )
                total = l if total is None else total + l
            return total.item()

        loss_1 = run(1)
        loss_3 = run(3)
        # Three supervision steps sum to ~3× the single-step value (approx)
        assert loss_3 > loss_1, \
            f"n_sup=3 loss ({loss_3:.4f}) should exceed n_sup=1 loss ({loss_1:.4f})"

    def test_state_updates_between_n_sup_steps(self):
        """z_H and z_L should change across supervision steps."""
        model  = small_model(n_sup=2)
        model.train()
        batch  = self._get_batch(bsz=2)
        inputs = batch["inputs"]
        emb    = model.embed(inputs)
        z_H, z_L = model.get_initial_states(inputs.shape[0])

        z_H_0 = z_H.clone()
        _, z_H_1, _ = model.reasoning_step(emb, z_H, z_L)

        assert not torch.allclose(z_H_0, z_H_1), \
            "z_H did not change after one supervision step"

    def test_train_script_runs(self):
        """
        Smoke-test train_sudoku.py via subprocess for a tiny number of steps.
        Creates a minimal dataset on disk and runs 5 steps.
        """
        import subprocess, sys

        out_dir  = os.path.join(self.tmp, "run")
        # train_sudoku.py expects data_dir/train/ and data_dir/test/
        train_data = os.path.join(self.tmp, "train")
        test_data  = os.path.join(self.tmp, "test")
        os.makedirs(train_data, exist_ok=True)
        os.makedirs(test_data,  exist_ok=True)
        make_dummy_data(16, train_data)
        make_dummy_data(4,  test_data)
        data_dir = self.tmp

        cmd = [
            sys.executable, "train_sudoku.py",
            f"data_dir={data_dir}",
            f"output_dir={out_dir}",
            "train.num_steps=5",
            "train.eval_every=5",
            "train.save_every=5",
            "train.log_every=1",
            "train.batch_size=4",
            "model.d_model=32",
            "model.n_heads=2",
            "model.n_layers=1",
            "model.L_cycles=2",
            "model.H_cycles=2",
            "model.n_sup=2",
        ]

        result = subprocess.run(
            cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, \
            f"train_sudoku.py failed:\n{result.stdout}\n{result.stderr}"

        # A checkpoint should have been saved
        assert any(
            f.endswith(".pt") for f in os.listdir(out_dir)
        ), "No checkpoint was saved"

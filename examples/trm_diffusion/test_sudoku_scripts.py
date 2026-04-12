"""
Integration tests for train_sudoku.py, train_mnist_sudoku.py, and sample_mnist_sudoku.py.

Each test:
  1. Creates a minimal data fixture in a temp dir.
  2. Runs the script for a handful of steps via subprocess.
  3. Verifies a checkpoint was written.
  4. Re-runs the script from that checkpoint (resume test).
  5. Cleans up.

Run a specific group:
    pytest test_sudoku_scripts.py -v -m sudoku
    pytest test_sudoku_scripts.py -v -m mnist_sudoku
    pytest test_sudoku_scripts.py -v -m mnist_sudoku_sample
    pytest test_sudoku_scripts.py -v -m mnist_sudoku_eval
"""

import glob
import inspect
import os
import shutil
import subprocess

import numpy as np
import pytest
import torch


# ── Data fixture helpers ────────────────────────────────────────────────────────

def create_sudoku_data(data_dir: str, n: int = 20) -> None:
    """Write minimal inputs.npy / labels.npy compatible with SudokuDataset."""
    os.makedirs(data_dir, exist_ok=True)
    rng    = np.random.default_rng(42)
    labels = rng.integers(2, 11, size=(n, 81)).astype(np.int32)
    inputs = labels.copy()
    blank  = rng.random((n, 81)) < 0.5
    inputs[blank] = 1                                   # blank token
    np.save(os.path.join(data_dir, "inputs.npy"), inputs)
    np.save(os.path.join(data_dir, "labels.npy"), labels)


def create_mnist_cache(mnist_root: str, split: str, n_per_digit: int = 10) -> None:
    """
    Write a mnist_{split}_cache.npz so MNISTSudokuDataset skips torchvision.
    Each digit 1-9 gets n_per_digit random (28, 28) float32 images.
    """
    os.makedirs(mnist_root, exist_ok=True)
    rng  = np.random.default_rng(7)
    data = {f"digit_{d}": rng.random((n_per_digit, 28, 28)).astype(np.float32)
            for d in range(1, 10)}
    np.savez(os.path.join(mnist_root, f"mnist_{split}_cache.npz"), **data)


# ── Script runner helpers ───────────────────────────────────────────────────────

def _run(cmd: list[str], cwd: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["WANDB_MODE"] = "offline"
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, env=env)


# ── Sudoku training ─────────────────────────────────────────────────────────────

def run_sudoku_training(tag: str, extra_args: list[str]) -> None:
    """
    Run train_sudoku.py for 4 steps, verify a checkpoint is written,
    then resume from it and run 4 more steps.
    """
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir    = os.path.join(scripts_dir, f"_test_sudoku_{tag}")
    data_dir    = os.path.join(base_dir, "data")
    out_dir     = os.path.join(base_dir, "output")

    os.makedirs(base_dir, exist_ok=True)
    create_sudoku_data(os.path.join(data_dir, "train"), n=20)

    fast_args = [
        f"data_dir={data_dir}",
        f"output_dir={out_dir}",
        "train.num_steps=4",
        "train.save_every=2",
        "train.eval_every=100",   # avoid eval during fast test
        "train.log_every=1",
        "train.warmup_steps=0",
        "train.batch_size=4",
        "model.d_model=32",
        "model.n_heads=2",
        "model.n_layers=1",
        "model.L_cycles=1",
        "model.H_cycles=1",
        "model.n_sup=1",
        "num_workers=0",
    ] + extra_args

    try:
        # ── Initial run ───────────────────────────────────────────────────────
        cmd_initial = ["python", "train_sudoku.py"] + fast_args
        res = _run(cmd_initial, cwd=scripts_dir)
        assert res.returncode == 0, \
            f"[{tag}] Initial run failed:\n{res.stdout}\n{res.stderr}"

        # ── Find checkpoint ───────────────────────────────────────────────────
        ckpts = glob.glob(os.path.join(out_dir, "checkpoint_step-*.pt"))
        assert ckpts, f"[{tag}] No step checkpoint found in {out_dir}"
        ckpt_path = sorted(ckpts)[-1]

        # ── Resume run ────────────────────────────────────────────────────────
        resume_args = fast_args + [
            f"resume_from_checkpoint={ckpt_path}",
            "train.num_steps=8",   # run 4 more steps after resuming from step 2/4
        ]
        cmd_resume = ["python", "train_sudoku.py"] + resume_args
        res_resume = _run(cmd_resume, cwd=scripts_dir)
        assert res_resume.returncode == 0, \
            f"[{tag}] Resume run failed:\n{res_resume.stdout}\n{res_resume.stderr}"

    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


@pytest.mark.sudoku
def test_sudoku_trm_default():
    run_sudoku_training("default", [])


@pytest.mark.sudoku
def test_sudoku_trm_puzzle_ids():
    run_sudoku_training("puzzle_ids", ["model.num_puzzle_ids=10"])


# ── MNIST Sudoku training ───────────────────────────────────────────────────────

def run_mnist_sudoku_training(tag: str, experiment: str, extra_args: list[str]) -> None:
    """
    Run train_mnist_sudoku.py for 4 steps, verify checkpoint, then resume.
    """
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir    = os.path.join(scripts_dir, f"_test_mnist_{tag}")
    data_dir    = os.path.join(base_dir, "data")
    mnist_dir   = os.path.join(base_dir, "mnist")
    out_dir     = os.path.join(base_dir, "output")

    os.makedirs(base_dir, exist_ok=True)
    # Sudoku data
    create_sudoku_data(os.path.join(data_dir, "train"), n=20)
    # MNIST cache (bypasses torchvision)
    create_mnist_cache(mnist_dir, "train", n_per_digit=10)
    create_mnist_cache(mnist_dir, "test",  n_per_digit=10)

    # cell_size=8 → painter_size=72, n_halvings=3
    fast_args = [
        f"experiment={experiment}",
        f"data.sudoku_dir={data_dir}",
        f"data.mnist_root={mnist_dir}",
        f"data.cell_size=8",
        f"output_dir={out_dir}",
        "train.num_steps=4",
        "train.save_every=2",
        "train.eval_every=100",
        "train.log_every=1",
        "train.warmup_steps=0",
        "train.batch_size=2",
        # Tiny model – use ++ so Hydra won't reject keys not already in the struct
        "++model.kwargs.thinker_hidden=16",
        "++model.kwargs.thinker_layers=2",
        "++model.kwargs.bridge_channels=4",
        "++model.kwargs.painter_channels=[16,32]",
        "num_workers=0",
    ] + extra_args

    try:
        # ── Initial run ───────────────────────────────────────────────────────
        cmd_initial = ["python", "train_mnist_sudoku.py"] + fast_args
        res = _run(cmd_initial, cwd=scripts_dir)
        assert res.returncode == 0, \
            f"[{tag}] Initial run failed:\n{res.stdout}\n{res.stderr}"

        # ── Find checkpoint ───────────────────────────────────────────────────
        ckpts = glob.glob(os.path.join(out_dir, "checkpoint_step-*.pt"))
        assert ckpts, f"[{tag}] No step checkpoint found in {out_dir}"
        ckpt_path = sorted(ckpts)[-1]

        # ── Resume run ────────────────────────────────────────────────────────
        resume_args = fast_args + [
            f"resume_from_checkpoint={ckpt_path}",
            "train.num_steps=8",
        ]
        cmd_resume = ["python", "train_mnist_sudoku.py"] + resume_args
        res_resume = _run(cmd_resume, cwd=scripts_dir)
        assert res_resume.returncode == 0, \
            f"[{tag}] Resume run failed:\n{res_resume.stdout}\n{res_resume.stderr}"

    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


# V0 needs explicit encoder-kwargs override (it doesn't have enc_out_channels)
@pytest.mark.mnist_sudoku
def test_mnist_sudoku_v0():
    run_mnist_sudoku_training("v0", "v0", [
        "train.sudoku_loss_weight=0.1",
    ])


@pytest.mark.mnist_sudoku
def test_mnist_sudoku_v1():
    run_mnist_sudoku_training("v1", "v1", [
        "train.sudoku_loss_weight=0.1",
        "++model.kwargs.enc_out_channels=8",
    ])


@pytest.mark.mnist_sudoku
def test_mnist_sudoku_v2():
    run_mnist_sudoku_training("v2", "v2", ["++model.kwargs.enc_out_channels=8"])


@pytest.mark.mnist_sudoku
def test_mnist_sudoku_v3():
    run_mnist_sudoku_training("v3", "v3", [
        "++model.kwargs.enc_out_channels=8",
        "++model.kwargs.thinker_out_channels=16",
    ])


@pytest.mark.mnist_sudoku
def test_mnist_sudoku_v4():
    # compression_factor=4 → 72/4=18×18 thinker grid
    run_mnist_sudoku_training("v4", "v4", [
        "++model.kwargs.enc_out_channels=8",
        "++model.kwargs.thinker_out_channels=16",
        "++model.kwargs.compression_factor=4",
    ])


# ── MNIST Sudoku sampling ───────────────────────────────────────────────────────

def create_mnist_sudoku_checkpoint(ckpt_path: str, experiment: str, extra_model_args: dict) -> None:
    """Instantiate the model and save a minimal checkpoint .pt file."""
    from mnist_sudoku_models import (
        MNISTRatatouilleV0, MNISTRatatouilleV1, MNISTRatatouilleV2,
        MNISTRatatouilleV3, MNISTRatatouilleV4,
    )
    registry = {
        "v0": MNISTRatatouilleV0, "v1": MNISTRatatouilleV1,
        "v2": MNISTRatatouilleV2, "v3": MNISTRatatouilleV3,
        "v4": MNISTRatatouilleV4,
    }
    ModelCls = registry[experiment]
    # Filter to only the kwargs the model's __init__ actually accepts
    valid = set(inspect.signature(ModelCls.__init__).parameters) - {"self"}
    kwargs = {k: v for k, v in extra_model_args.items() if k in valid}
    model = ModelCls(**kwargs)
    torch.save({"step": 0, "model_state": model.state_dict(), "optimizer_state": {}}, ckpt_path)


def run_mnist_sudoku_sample_test(tag: str, experiment: str, extra_model_kwargs: dict,
                                  extra_args: list[str]) -> None:
    """
    Create a dummy checkpoint, run sample_mnist_sudoku.py, verify PNG + JSONL output.
    """
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir    = os.path.join(scripts_dir, f"_test_sample_mnist_{tag}")
    out_dir     = os.path.join(base_dir, "output")
    ckpt_path   = os.path.join(base_dir, "checkpoint_test.pt")

    os.makedirs(base_dir, exist_ok=True)

    # cell_size=8 so painter_size=72
    cell_size    = 8
    painter_size = cell_size * 9   # 72
    model_kwargs = {
        "painter_size":    painter_size,
        "cell_size":       cell_size,
        "bridge_channels": 4,
        "thinker_hidden":  16,
        "thinker_layers":  2,
        "painter_channels": (16, 32),
        **extra_model_kwargs,
    }
    create_mnist_sudoku_checkpoint(ckpt_path, experiment, model_kwargs)

    cmd = [
        "python", "sample_mnist_sudoku.py",
        f"experiment={experiment}",
        f"checkpoint_path={ckpt_path}",
        f"output_dir={out_dir}",
        "num_samples=2",
        "sample_batch_size=2",
        "ddpm_num_inference_steps=2",
        "use_ddim=true",
        f"data.cell_size={cell_size}",
        "++model.kwargs.thinker_hidden=16",
        "++model.kwargs.thinker_layers=2",
        "++model.kwargs.bridge_channels=4",
        "++model.kwargs.painter_channels=[16,32]",
    ] + extra_args

    try:
        res = _run(cmd, cwd=scripts_dir)
        assert res.returncode == 0, \
            f"[{tag}] Sample failed:\n{res.stdout}\n{res.stderr}"

        samples_dir = os.path.join(out_dir, "samples")
        assert os.path.exists(samples_dir), f"[{tag}] Samples dir not found"

        png_files = [f for f in os.listdir(samples_dir) if f.endswith(".png")]
        assert len(png_files) == 2, f"[{tag}] Expected 2 PNGs, got {len(png_files)}"

        jsonl = os.path.join(samples_dir, "metadata_rank0.jsonl")
        assert os.path.exists(jsonl), f"[{tag}] metadata JSONL missing"
        with open(jsonl) as f:
            lines = [l for l in f.read().splitlines() if l.strip()]
        assert len(lines) == 2, f"[{tag}] Expected 2 JSONL lines, got {len(lines)}"

    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


@pytest.mark.mnist_sudoku_sample
def test_sample_mnist_sudoku_v0():
    run_mnist_sudoku_sample_test("v0", "v0", {}, [])


@pytest.mark.mnist_sudoku_sample
def test_sample_mnist_sudoku_v1():
    run_mnist_sudoku_sample_test("v1", "v1", {"enc_out_channels": 8}, [
        "++model.kwargs.enc_out_channels=8",
    ])


@pytest.mark.mnist_sudoku_sample
def test_sample_mnist_sudoku_v2():
    run_mnist_sudoku_sample_test("v2", "v2", {"enc_out_channels": 8}, [
        "++model.kwargs.enc_out_channels=8",
    ])


@pytest.mark.mnist_sudoku_sample
def test_sample_mnist_sudoku_v3():
    run_mnist_sudoku_sample_test("v3", "v3",
        {"enc_out_channels": 8, "thinker_out_channels": 16},
        ["++model.kwargs.enc_out_channels=8", "++model.kwargs.thinker_out_channels=16"])


@pytest.mark.mnist_sudoku_sample
def test_sample_mnist_sudoku_v4():
    # compression_factor=4 → 72/4=18 thinker grid
    run_mnist_sudoku_sample_test("v4", "v4",
        {"enc_out_channels": 8, "thinker_out_channels": 16, "compression_factor": 4},
        ["++model.kwargs.enc_out_channels=8", "++model.kwargs.thinker_out_channels=16",
         "++model.kwargs.compression_factor=4"])


# ── MNIST Sudoku evaluation (classifier + DDIM) ────────────────────────────────

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.mnist_sudoku_eval
def test_mnist_eval_shapes():
    """Unit test: MNISTCellClassifier forward shape and evaluate_grids output keys."""
    import sys
    if SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, SCRIPTS_DIR)
    from mnist_eval import MNISTCellClassifier, evaluate_grids

    cell_size = 8
    clf = MNISTCellClassifier(cell_size=cell_size)
    clf.eval()

    # Classifier output shape
    x = torch.zeros(4, 1, cell_size, cell_size)
    logits = clf(x)
    assert logits.shape == (4, 9), f"Expected (4, 9), got {logits.shape}"

    # evaluate_grids returns cell_acc and puzzle_acc in [0, 1]
    B = 3
    images    = torch.rand(B, 1, cell_size * 9, cell_size * 9)
    solutions = torch.randint(0, 9, (B, 81))
    result = evaluate_grids(images, solutions, clf, cell_size)
    assert set(result.keys()) == {"cell_acc", "puzzle_acc"}
    assert 0.0 <= result["cell_acc"]   <= 1.0
    assert 0.0 <= result["puzzle_acc"] <= 1.0


@pytest.mark.mnist_sudoku_eval
def test_mnist_classifier_train_and_load():
    """Unit test: train classifier from numpy cache, reload it, run evaluate_grids."""
    import sys
    if SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, SCRIPTS_DIR)
    from mnist_eval import evaluate_grids, load_or_train_classifier

    base_dir  = os.path.join(SCRIPTS_DIR, "_test_clf_unit")
    mnist_dir = os.path.join(base_dir, "mnist")
    clf_path  = os.path.join(base_dir, "classifier.pt")
    os.makedirs(base_dir, exist_ok=True)
    create_mnist_cache(mnist_dir, "train", n_per_digit=5)

    try:
        device    = torch.device("cpu")
        cell_size = 8

        # First call: trains and saves
        clf = load_or_train_classifier(clf_path, mnist_dir, cell_size, device)
        assert os.path.exists(clf_path), "Classifier file not created after training"

        # Second call: loads from disk
        clf2 = load_or_train_classifier(clf_path, mnist_dir, cell_size, device)

        # evaluate_grids runs without error and returns sensible values
        images    = torch.rand(2, 1, cell_size * 9, cell_size * 9)
        solutions = torch.randint(0, 9, (2, 81))
        result    = evaluate_grids(images, solutions, clf2, cell_size)
        assert 0.0 <= result["cell_acc"]   <= 1.0
        assert 0.0 <= result["puzzle_acc"] <= 1.0
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


@pytest.mark.mnist_sudoku_eval
def test_mnist_sudoku_eval_integration():
    """End-to-end: training with eval_classifier_path triggers classifier + DDIM eval."""
    scripts_dir = SCRIPTS_DIR
    base_dir    = os.path.join(scripts_dir, "_test_mnist_eval")
    data_dir    = os.path.join(base_dir, "data")
    mnist_dir   = os.path.join(base_dir, "mnist")
    out_dir     = os.path.join(base_dir, "output")
    clf_path    = os.path.join(base_dir, "classifier.pt")

    os.makedirs(base_dir, exist_ok=True)
    create_sudoku_data(os.path.join(data_dir, "train"), n=20)
    create_mnist_cache(mnist_dir, "train", n_per_digit=10)
    create_mnist_cache(mnist_dir, "test",  n_per_digit=10)

    cmd = [
        "python", "train_mnist_sudoku.py",
        "experiment=v2",
        f"data.sudoku_dir={data_dir}",
        f"data.mnist_root={mnist_dir}",
        "data.cell_size=8",
        f"output_dir={out_dir}",
        "train.num_steps=4",
        "train.save_every=100",        # skip step checkpoints
        "train.eval_every=2",          # eval at steps 2 and 4
        "train.log_every=1",
        "train.warmup_steps=0",
        "train.batch_size=2",
        "++model.kwargs.enc_out_channels=8",
        "++model.kwargs.thinker_hidden=16",
        "++model.kwargs.thinker_layers=2",
        "++model.kwargs.bridge_channels=4",
        "++model.kwargs.painter_channels=[16,32]",
        f"eval_classifier_path={clf_path}",
        "eval_num_ddim_steps=2",       # fast DDIM for tests
        "eval_num_samples=2",
        "num_workers=0",
    ]

    try:
        res = _run(cmd, cwd=scripts_dir)
        assert res.returncode == 0, \
            f"Eval integration run failed:\n{res.stdout}\n{res.stderr}"

        # Classifier should have been trained and saved on disk
        assert os.path.exists(clf_path), \
            "Classifier file was not created during training"

        # cell_acc should appear in combined script output
        combined = res.stdout + res.stderr
        assert "cell_acc" in combined, \
            f"Digit eval metrics not logged in output:\n{combined}"
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)

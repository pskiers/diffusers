import os
import shutil
import subprocess
import pytest
import torch
from PIL import Image
from hydra import compose, initialize
from hydra.utils import instantiate
from hydra.core.global_hydra import GlobalHydra


def create_dummy_checkpoint(experiment_name, ckpt_dir):
    """
    Dynamically generates a checkpoint with random weights based on the YAML config
    so we can test the sampling script without waiting for an actual training run.
    """
    # Clear Hydra state to prevent Pytest clashes
    GlobalHydra.instance().clear()

    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name="config", overrides=[f"experiment={experiment_name}"])

    unet = instantiate(cfg.model, _convert_="all")
    unet_dir = os.path.join(ckpt_dir, "unet")
    unet.save_pretrained(unet_dir)

    # Generate the dummy TRM loop anchors if the experiment calls for it
    if cfg.get("use_small_loop", False):
        res = cfg.dataset.resolution
        channels = cfg.dataset.input_channels
        sz = res if cfg.dataset.get("vae_name") is None else res // 8

        y_init = torch.randn(1, channels, sz, sz)
        z_init = torch.randn(1, channels, sz, sz)
        torch.save(y_init, os.path.join(ckpt_dir, "y_init.pt"))
        torch.save(z_init, os.path.join(ckpt_dir, "z_init.pt"))

    return cfg.dataset.resolution


def run_sample_test(experiment_name):
    base_dir = f"test_sample_{experiment_name}"
    ckpt_dir = os.path.join(base_dir, "checkpoint")
    out_dir = os.path.join(base_dir, "output")

    os.makedirs(ckpt_dir, exist_ok=True)

    try:
        # 1. Create the fake checkpoint
        expected_res = create_dummy_checkpoint(experiment_name, ckpt_dir)

        # 2. Run sample.py
        env = os.environ.copy()

        cmd = [
            "accelerate",
            "launch",
            "--num_processes=1",
            "--mixed_precision=fp16",
            "sample.py",
            f"experiment={experiment_name}",
            f"checkpoint_path={ckpt_dir}",
            f"output_dir={out_dir}",
            "num_samples=2",
            "sample_batch_size=2",
            "ddpm_num_inference_steps=2",  # 2 steps makes the test finish in seconds!
            "use_ddim=true",  # DDIM is required to safely do only 2 steps
        ]

        res = subprocess.run(cmd, env=env, capture_output=True, text=True)
        assert res.returncode == 0, f"Sampling failed for {experiment_name}:\n{res.stdout}\n{res.stderr}"

        # 3. Verify output images
        samples_dir = os.path.join(out_dir, "samples")
        assert os.path.exists(samples_dir), f"Samples directory missing for {experiment_name}"

        imgs = os.listdir(samples_dir)
        assert len(imgs) == 2, f"Expected 2 images, got {len(imgs)}"

        for img_name in imgs:
            img_path = os.path.join(samples_dir, img_name)

            with Image.open(img_path) as img:
                img.verify()  # Validates the PNG headers are uncorrupted

            # Reopen to check the physical canvas size
            with Image.open(img_path) as img:
                assert img.size == (
                    expected_res,
                    expected_res,
                ), f"Wrong resolution! Expected {expected_res}, got {img.size}"

    finally:
        # Clean up the gigabytes of dummy models we just made
        shutil.rmtree(base_dir, ignore_errors=True)


# ---------------------------------------------------------
# The Test Suite
# ---------------------------------------------------------


def test_sample_uncond_cifar100_std():
    run_sample_test("uncond_cifar100_std")


def test_sample_uncond_cifar100_trm():
    run_sample_test("uncond_cifar100_trm")


def test_sample_cond_cifar100_std():
    run_sample_test("cond_cifar100_std")


def test_sample_cond_cifar100_trm():
    run_sample_test("cond_cifar100_trm")


def test_sample_cond_imgnet_std():
    run_sample_test("cond_imgnet_std")


def test_sample_cond_imgnet_trm():
    run_sample_test("cond_imgnet_trm")


def test_sample_clevr_relative_std():
    run_sample_test("clevr_relative_std")


def test_sample_clevr_relative_trm():
    run_sample_test("clevr_relative_trm")

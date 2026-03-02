import os
import shutil
import subprocess
import pytest
import torch
from PIL import Image
from hydra import compose, initialize
from hydra.utils import instantiate
from hydra.core.global_hydra import GlobalHydra


def create_dummy_checkpoint(experiment_name, out_dir, ckpt_dir):
    """
    Generates a realistic directory structure:
    out_dir/
      y_init.pt
      z_init.pt
      ckpt_dir/
        unet/
          diffusion_pytorch_model.safetensors
    """
    GlobalHydra.instance().clear()

    with initialize(version_base=None, config_path="configs"):
        # We explicitly override output_dir so the config knows where "root" is
        cfg = compose(config_name="config", overrides=[
            f"experiment={experiment_name}",
            f"output_dir={out_dir}"
        ])

    unet = instantiate(cfg.model, _convert_="all")
    unet_dir = os.path.join(ckpt_dir, "unet")
    unet.save_pretrained(unet_dir)

    # Save the dummy TRM anchors in the ROOT out_dir, not the checkpoint dir!
    if cfg.get("use_small_loop", False):
        res = cfg.dataset.resolution
        channels = cfg.dataset.input_channels
        sz = res if cfg.dataset.get("vae_name") is None else res // 8

        y_init = torch.randn(1, channels, sz, sz)
        z_init = torch.randn(1, channels, sz, sz)
        torch.save(y_init, os.path.join(out_dir, "y_init.pt"))
        torch.save(z_init, os.path.join(out_dir, "z_init.pt"))

    return cfg.dataset.resolution


def run_sample_test(experiment_name):
    base_dir = f"test_sample_{experiment_name}"
    out_dir = base_dir  # This acts as the root args.output_dir
    ckpt_dir = os.path.join(out_dir, "checkpoint-1000")

    os.makedirs(ckpt_dir, exist_ok=True)

    try:
        expected_res = create_dummy_checkpoint(experiment_name, out_dir, ckpt_dir)

        env = os.environ.copy()

        cmd = [
            "accelerate", "launch", "--num_processes=1", "--mixed_precision=fp16",
            "sample.py",
            f"experiment={experiment_name}",
            f"output_dir={out_dir}",
            f"checkpoint_path={ckpt_dir}", # Point explicitly to the dummy step folder
            "num_samples=2",
            "sample_batch_size=2",
            "ddpm_num_inference_steps=2",
            "use_ddim=true"
        ]

        res = subprocess.run(cmd, env=env, capture_output=True, text=True)
        assert res.returncode == 0, f"Sampling failed for {experiment_name}:\n{res.stdout}\n{res.stderr}"

        # Verify JSONL and PNGs
        samples_dir = os.path.join(out_dir, "samples")
        assert os.path.exists(samples_dir), f"Samples directory missing for {experiment_name}"

        png_files = [f for f in os.listdir(samples_dir) if f.endswith(".png")]
        assert len(png_files) == 2, f"Expected 2 images, got {len(png_files)}"

        metadata_file = os.path.join(samples_dir, "metadata_rank0.jsonl")
        assert os.path.exists(metadata_file), f"Metadata file {metadata_file} is missing!"

        with open(metadata_file, "r") as f:
            assert len(f.readlines()) == 2, "Expected exactly 2 lines in metadata JSONL."

    finally:
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

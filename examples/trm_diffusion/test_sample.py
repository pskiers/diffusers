import os
import shutil
import subprocess
import pytest
import torch
from PIL import Image
from hydra import compose, initialize
from hydra.utils import instantiate
from hydra.core.global_hydra import GlobalHydra


def create_dummy_checkpoint(experiment_name, out_dir, ckpt_dir, extra_args=[]):
    """
    Generates a realistic directory structure matching the exact architecture specified.
    out_dir/
      ckpt_dir/
        unet/
          diffusion_pytorch_model.safetensors
        unet_ema/          <- copy so sample.py works with use_ema=true (the global default)
          diffusion_pytorch_model.safetensors
    """
    GlobalHydra.instance().clear()

    overrides = [f"experiment={experiment_name}", f"output_dir={out_dir}"] + extra_args

    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name="config", overrides=overrides)

    unet = instantiate(cfg.model, _convert_="all")
    unet_to_save = unet.core_model if hasattr(unet, "core_model") else unet

    unet_to_save.save_pretrained(os.path.join(ckpt_dir, "unet"))
    # Also save to unet_ema so sample.py works with the global use_ema=true default.
    unet_to_save.save_pretrained(os.path.join(ckpt_dir, "unet_ema"))

    return cfg.dataset.resolution


def run_sample_test(experiment_name, extra_args=None):
    if extra_args is None:
        extra_args = []

    base_dir = f"test_sample_{experiment_name}_{abs(hash(str(extra_args))) % 10000}"
    out_dir = base_dir
    ckpt_dir = os.path.join(out_dir, "checkpoint-1000")

    os.makedirs(ckpt_dir, exist_ok=True)

    try:
        # Pass extra_args so the dummy checkpoint matches the requested architecture
        expected_res = create_dummy_checkpoint(experiment_name, out_dir, ckpt_dir, extra_args)
        env = os.environ.copy()

        cmd = [
            "accelerate",
            "launch",
            "--num_processes=1",
            "--mixed_precision=fp16",
            "sample.py",
            f"experiment={experiment_name}",
            f"output_dir={out_dir}",
            f"checkpoint_path={ckpt_dir}",
            "num_samples=2",
            "sample_batch_size=2",
            "ddpm_num_inference_steps=2",
            "use_ddim=true",
        ] + extra_args

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


# =========================================================================================
# The Test Suite (Matched exactly to test_training_scripts.py)
# =========================================================================================


# -----------------------------------------------------------------------------------------
# Standard UNet diffusion
# -----------------------------------------------------------------------------------------
@pytest.mark.cifar100
@pytest.mark.unet
@pytest.mark.standard
def test_sample_unconditional_standard_cifar():
    run_sample_test("uncond_cifar100_std")


@pytest.mark.cifar100
@pytest.mark.unet
@pytest.mark.standard
def test_sample_conditional_standard_cifar():
    run_sample_test("cond_cifar100_std")


@pytest.mark.imagenet
@pytest.mark.unet
@pytest.mark.standard
def test_sample_conditional_standard_imagenet():
    run_sample_test("cond_imgnet_std")


@pytest.mark.clevr
@pytest.mark.unet
@pytest.mark.standard
def test_sample_clevr_standard():
    run_sample_test("clevr_relative_std")


# -----------------------------------------------------------------------------------------
# TRM Diffusion v1 UNet
# -----------------------------------------------------------------------------------------
@pytest.mark.cifar100
@pytest.mark.unet
@pytest.mark.trm
@pytest.mark.v1
def test_sample_unconditional_trm_v1_cifar():
    run_sample_test("uncond_cifar100_trm")


@pytest.mark.cifar100
@pytest.mark.unet
@pytest.mark.trm
@pytest.mark.v1
def test_sample_conditional_trm_v1_cifar():
    run_sample_test("cond_cifar100_trm")


@pytest.mark.imagenet
@pytest.mark.unet
@pytest.mark.trm
@pytest.mark.v1
def test_sample_conditional_trm_v1_imagenet():
    run_sample_test("cond_imgnet_trm")


@pytest.mark.clevr
@pytest.mark.unet
@pytest.mark.trm
@pytest.mark.v1
def test_sample_clevr_trm_v1():
    run_sample_test("clevr_relative_trm")


# -----------------------------------------------------------------------------------------
# Standard ViT diffusion
# -----------------------------------------------------------------------------------------
@pytest.mark.cifar100
@pytest.mark.vit
@pytest.mark.standard
def test_sample_cond_cifar_vit_std():
    run_sample_test(
        "cond_cifar100_std",
        ["model=cifar100_vit_std", "~model.num_class_embeds", "model.num_classes=100"],
    )


@pytest.mark.imagenet
@pytest.mark.vit
@pytest.mark.standard
def test_sample_cond_imagenet_vit_std():
    run_sample_test("cond_imgnet_std", ["model=imagenet_vit_std"])


@pytest.mark.clevr
@pytest.mark.vit
@pytest.mark.standard
def test_sample_cond_clevr_vit_std():
    run_sample_test("clevr_relative_std", ["model=clevr_vit_std"])


# -----------------------------------------------------------------------------------------
# TRM Diffusion v1 ViT
# -----------------------------------------------------------------------------------------
@pytest.mark.cifar100
@pytest.mark.vit
@pytest.mark.trm
@pytest.mark.v1
def test_sample_cond_cifar_vit_trm():
    run_sample_test(
        "cond_cifar100_trm",
        ["model=cifar100_vit_trm", "~model.core_model.num_class_embeds"],
    )


@pytest.mark.imagenet
@pytest.mark.vit
@pytest.mark.trm
@pytest.mark.v1
def test_sample_cond_imagenet_vit_trm():
    run_sample_test("cond_imgnet_trm", ["model=imagenet_vit_trm"])


@pytest.mark.clevr
@pytest.mark.vit
@pytest.mark.trm
@pytest.mark.v1
def test_sample_cond_clevr_vit_trm():
    run_sample_test("clevr_relative_trm", ["model=clevr_vit_trm"])


# -----------------------------------------------------------------------------------------
# TRM Diffusion v2 UNet
# -----------------------------------------------------------------------------------------
@pytest.mark.cifar100
@pytest.mark.unet
@pytest.mark.trm
@pytest.mark.v2
def test_sample_cond_cifar_unet_trm_v2():
    run_sample_test("cond_cifar100_trm", ["model=unet2d_trm_v2"])


@pytest.mark.imagenet
@pytest.mark.unet
@pytest.mark.trm
@pytest.mark.v2
def test_sample_cond_imagenet_unet_trm_v2():
    run_sample_test("cond_imgnet_trm", ["model=imagenet_condition_unet_trm_v2"])


@pytest.mark.clevr
@pytest.mark.unet
@pytest.mark.trm
@pytest.mark.v2
def test_sample_cond_clevr_unet_trm_v2():
    run_sample_test("clevr_relative_trm", ["model=clevr_condition_unet_trm_v2"])


# -----------------------------------------------------------------------------------------
# TRM Diffusion v2 ViT
# -----------------------------------------------------------------------------------------
@pytest.mark.cifar100
@pytest.mark.vit
@pytest.mark.trm
@pytest.mark.v2
def test_sample_cond_cifar_vit_trm_v2():
    run_sample_test(
        "cond_cifar100_trm",
        ["model=cifar100_vit_trm_v2", "~model.core_model.num_class_embeds"],
    )


@pytest.mark.imagenet
@pytest.mark.vit
@pytest.mark.trm
@pytest.mark.v2
def test_sample_cond_imagenet_vit_trm_v2():
    run_sample_test("cond_imgnet_trm", ["model=imagenet_vit_trm_v2"])


@pytest.mark.clevr
@pytest.mark.vit
@pytest.mark.trm
@pytest.mark.v2
def test_sample_cond_clevr_vit_trm_v2():
    run_sample_test("clevr_relative_trm", ["model=clevr_vit_trm_v2"])


# -----------------------------------------------------------------------------------------
# TRM Diffusion v3 UNet
# -----------------------------------------------------------------------------------------
@pytest.mark.cifar100
@pytest.mark.unet
@pytest.mark.trm
@pytest.mark.v3
def test_sample_cond_cifar_unet_trm_v3():
    run_sample_test(
        "cond_cifar100_trm",
        ["model=unet2d_trm_v2", "model._target_=trm_models.UNetTRMv3"],
    )


@pytest.mark.imagenet
@pytest.mark.unet
@pytest.mark.trm
@pytest.mark.v3
def test_sample_cond_imagenet_unet_trm_v3():
    run_sample_test(
        "cond_imgnet_trm",
        ["model=imagenet_condition_unet_trm_v2", "model._target_=trm_models.UNetTRMv3"],
    )


@pytest.mark.clevr
@pytest.mark.unet
@pytest.mark.trm
@pytest.mark.v3
def test_sample_cond_clevr_unet_trm_v3():
    run_sample_test(
        "clevr_relative_trm",
        ["model=clevr_condition_unet_trm_v2", "model._target_=trm_models.UNetTRMv3"],
    )


# -----------------------------------------------------------------------------------------
# TRM Diffusion v3 ViT/DiT
# -----------------------------------------------------------------------------------------
@pytest.mark.cifar100
@pytest.mark.vit
@pytest.mark.trm
@pytest.mark.v3
def test_sample_cond_cifar_vit_trm_v3():
    run_sample_test(
        "cond_cifar100_trm",
        [
            "model=cifar100_vit_trm_v2",
            "~model.core_model.num_class_embeds",
            "model._target_=trm_models.DiTTRMv3",
        ],
    )


@pytest.mark.imagenet
@pytest.mark.vit
@pytest.mark.trm
@pytest.mark.v3
def test_sample_cond_imagenet_vit_trm_v3():
    run_sample_test(
        "cond_imgnet_trm",
        ["model=imagenet_vit_trm_v2", "model._target_=trm_models.DiTTRMv3"],
    )


@pytest.mark.clevr
@pytest.mark.vit
@pytest.mark.trm
@pytest.mark.v3
def test_sample_cond_clevr_vit_trm_v3():
    run_sample_test(
        "clevr_relative_trm",
        ["model=clevr_vit_trm_v2", "model._target_=trm_models.DiTTRMv3"],
    )


# -----------------------------------------------------------------------------------------
# TRM Diffusion v4
# -----------------------------------------------------------------------------------------
@pytest.mark.cifar100
@pytest.mark.unet
@pytest.mark.trm
@pytest.mark.v4
def test_sample_cond_cifar_unet_trm_v4():
    run_sample_test(
        "cond_cifar100_trm",
        ["model=unet2d_trm_v2", "model._target_=trm_models.UNetTRMv4"],
    )


@pytest.mark.imagenet
@pytest.mark.unet
@pytest.mark.trm
@pytest.mark.v4
def test_sample_cond_imagenet_unet_trm_v4():
    run_sample_test(
        "cond_imgnet_trm",
        ["model=imagenet_condition_unet_trm_v2", "model._target_=trm_models.UNetTRMv4"],
    )


@pytest.mark.clevr
@pytest.mark.unet
@pytest.mark.trm
@pytest.mark.v4
def test_sample_cond_clevr_unet_trm_v4():
    run_sample_test(
        "clevr_relative_trm",
        ["model=clevr_condition_unet_trm_v2", "model._target_=trm_models.UNetTRMv4"],
    )


@pytest.mark.cifar100
@pytest.mark.vit
@pytest.mark.trm
@pytest.mark.v4
def test_sample_cond_cifar_vit_trm_v4():
    run_sample_test(
        "cond_cifar100_trm",
        [
            "model=cifar100_vit_trm_v2",
            "~model.core_model.num_class_embeds",
            "model._target_=trm_models.DiTTRMv4",
        ],
    )


@pytest.mark.imagenet
@pytest.mark.vit
@pytest.mark.trm
@pytest.mark.v4
def test_sample_cond_imagenet_vit_trm_v4():
    run_sample_test(
        "cond_imgnet_trm",
        ["model=imagenet_vit_trm_v2", "model._target_=trm_models.DiTTRMv4"],
    )


@pytest.mark.clevr
@pytest.mark.vit
@pytest.mark.trm
@pytest.mark.v4
def test_sample_cond_clevr_vit_trm_v4():
    run_sample_test(
        "clevr_relative_trm",
        ["model=clevr_vit_trm_v2", "model._target_=trm_models.DiTTRMv4"],
    )


# -----------------------------------------------------------------------------------------
# TRM Diffusion v5
# -----------------------------------------------------------------------------------------
@pytest.mark.cifar100
@pytest.mark.unet
@pytest.mark.trm
@pytest.mark.v5
def test_sample_cond_cifar_unet_trm_v5():
    run_sample_test(
        "cond_cifar100_trm",
        ["model=unet2d_trm_v2", "model._target_=trm_models.UNetTRMv5"],
    )


@pytest.mark.imagenet
@pytest.mark.unet
@pytest.mark.trm
@pytest.mark.v5
def test_sample_cond_imagenet_unet_trm_v5():
    run_sample_test(
        "cond_imgnet_trm",
        ["model=imagenet_condition_unet_trm_v2", "model._target_=trm_models.UNetTRMv5"],
    )


@pytest.mark.clevr
@pytest.mark.unet
@pytest.mark.trm
@pytest.mark.v5
def test_sample_cond_clevr_unet_trm_v5():
    run_sample_test(
        "clevr_relative_trm",
        ["model=clevr_condition_unet_trm_v2", "model._target_=trm_models.UNetTRMv5"],
    )


@pytest.mark.cifar100
@pytest.mark.vit
@pytest.mark.trm
@pytest.mark.v5
def test_sample_cond_cifar_vit_trm_v5():
    run_sample_test(
        "cond_cifar100_trm",
        [
            "model=cifar100_vit_trm_v2",
            "~model.core_model.num_class_embeds",
            "model._target_=trm_models.DiTTRMv5",
        ],
    )


@pytest.mark.imagenet
@pytest.mark.vit
@pytest.mark.trm
@pytest.mark.v5
def test_sample_cond_imagenet_vit_trm_v5():
    run_sample_test(
        "cond_imgnet_trm",
        ["model=imagenet_vit_trm_v2", "model._target_=trm_models.DiTTRMv5"],
    )


@pytest.mark.clevr
@pytest.mark.vit
@pytest.mark.trm
@pytest.mark.v5
def test_sample_cond_clevr_vit_trm_v5():
    run_sample_test(
        "clevr_relative_trm",
        ["model=clevr_vit_trm_v2", "model._target_=trm_models.DiTTRMv5"],
    )


@pytest.mark.clevr
@pytest.mark.vit
@pytest.mark.mask
def test_cond_clevr_mask_dit():
    run_sample_test("clevr_mask_experiment")

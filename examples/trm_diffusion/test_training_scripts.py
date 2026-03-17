import os
import subprocess
import shutil
import pytest


def run_script(script_name, output_dir, extra_args):
    os.makedirs(output_dir, exist_ok=True)
    env = os.environ.copy()
    env["WANDB_MODE"] = "offline"
    env["WANDB_DIR"] = output_dir

    base_cmd = [
        "accelerate",
        "launch",
        "--mixed_precision=fp16",
        "--num_processes=1",
        script_name,
        f"output_dir={output_dir}",
        "train_batch_size=4",
        "eval_batch_size=4",
        "epoch_max_batches_train=1",
        "epoch_max_batches_eval=1",
        "save_model_epochs=1",
        "save_images_epochs=1",
        "ddpm_num_inference_steps=1",
        "checkpointing_steps=1",
        "logger=wandb",
        "gradient_accumulation_steps=1",
        "use_ema=true",
        "optimizer.lr=1e-4",
        "lr_scheduler.warmup_steps=1",
    ]

    cmd_initial = base_cmd + extra_args + ["num_epochs=2"]
    res_initial = subprocess.run(cmd_initial, env=env, capture_output=True, text=True)
    assert res_initial.returncode == 0, f"Initial Run Failed!\n{res_initial.stdout}\n{res_initial.stderr}"

    cmd_resume = base_cmd + extra_args + ["num_epochs=3", "resume_from_checkpoint=latest"]
    res_resume = subprocess.run(cmd_resume, env=env, capture_output=True, text=True)
    assert res_resume.returncode == 0, f"Resume Run Failed!\n{res_resume.stdout}\n{res_resume.stderr}"

    shutil.rmtree(output_dir, ignore_errors=True)


# Standard UNet diffusion
@pytest.mark.cifar100
@pytest.mark.unet
@pytest.mark.standard
def test_unconditional_standard_cifar():
    run_script("train.py", "test-cifar100-standard", ["experiment=uncond_cifar100_std"])


@pytest.mark.cifar100
@pytest.mark.unet
@pytest.mark.standard
def test_conditional_standard_cifar():
    run_script("train.py", "test-cifar100-cond", ["experiment=cond_cifar100_std"])


@pytest.mark.imagenet
@pytest.mark.unet
@pytest.mark.standard
def test_conditional_standard_imagenet():
    run_script("train.py", "test-imagenet-cond", ["experiment=cond_imgnet_std"])


@pytest.mark.clevr
@pytest.mark.unet
@pytest.mark.standard
def test_clevr_standard():
    run_script("train.py", "test-clevr-standard", ["experiment=clevr_relative_std"])


# TRM Diffusion v1 UNet
@pytest.mark.cifar100
@pytest.mark.unet
@pytest.mark.trm
@pytest.mark.v1
def test_unconditional_trm_v1_cifar():
    run_script("train.py", "test-cifar100-small", ["experiment=uncond_cifar100_trm"])


@pytest.mark.cifar100
@pytest.mark.unet
@pytest.mark.trm
@pytest.mark.v1
def test_conditional_trm_v1_cifar():
    run_script("train.py", "test-cifar100-cond-small", ["experiment=cond_cifar100_trm"])


@pytest.mark.imagenet
@pytest.mark.unet
@pytest.mark.trm
@pytest.mark.v1
def test_conditional_trm_v1_imagenet():
    run_script("train.py", "test-imagenet-cond-small", ["experiment=cond_imgnet_trm"])


@pytest.mark.clevr
@pytest.mark.unet
@pytest.mark.trm
@pytest.mark.v1
def test_clevr_trm_v1():
    run_script("train.py", "test-clevr-small", ["experiment=clevr_relative_trm"])


# Standard ViT diffusion
@pytest.mark.cifar100
@pytest.mark.vit
@pytest.mark.standard
def test_cond_cifar_vit_std():
    run_script(
        "train.py",
        "test-cifar100-cond",
        ["experiment=cond_cifar100_std", "model=cifar100_vit_std", "~model.num_class_embeds", "model.num_classes=100"],
    )


@pytest.mark.imagenet
@pytest.mark.vit
@pytest.mark.standard
def test_cond_imagenet_vit_std():
    run_script("train.py", "test-imagenet-cond", ["experiment=cond_imgnet_std", "model=imagenet_vit_std"])


@pytest.mark.clevr
@pytest.mark.vit
@pytest.mark.standard
def test_cond_clevr_vit_std():
    run_script("train.py", "test-clevr-standard", ["experiment=clevr_relative_std", "model=clevr_vit_std"])


# TRM Diffusion v1 ViT
@pytest.mark.cifar100
@pytest.mark.vit
@pytest.mark.trm
@pytest.mark.v1
def test_cond_cifar_vit_trm():
    run_script(
        "train.py",
        "test-cifar100-cond-small",
        ["experiment=cond_cifar100_trm", "model=cifar100_vit_trm", "~model.core_model.num_class_embeds"],
    )


@pytest.mark.imagenet
@pytest.mark.vit
@pytest.mark.trm
@pytest.mark.v1
def test_cond_imagenet_vit_trm():
    run_script("train.py", "test-imagenet-cond-small", ["experiment=cond_imgnet_trm", "model=imagenet_vit_trm"])


@pytest.mark.clevr
@pytest.mark.vit
@pytest.mark.trm
@pytest.mark.v1
def test_cond_clevr_vit_trm():
    run_script("train.py", "test-clevr-small", ["experiment=clevr_relative_trm", "model=clevr_vit_trm"])


# TRM Diffusion v2 UNet
@pytest.mark.cifar100
@pytest.mark.unet
@pytest.mark.trm
@pytest.mark.v2
def test_cond_cifar_unet_trm_v2():
    run_script("train.py", "test-cifar100-cond-small", ["experiment=cond_cifar100_trm", "model=unet2d_trm_v2"])


@pytest.mark.imagenet
@pytest.mark.unet
@pytest.mark.trm
@pytest.mark.v2
def test_cond_imagenet_unet_trm_v2():
    run_script(
        "train.py", "test-imagenet-cond-small", ["experiment=cond_imgnet_trm", "model=imagenet_condition_unet_trm_v2"]
    )


@pytest.mark.clevr
@pytest.mark.unet
@pytest.mark.trm
@pytest.mark.v2
def test_cond_clevr_unet_trm_v2():
    run_script("train.py", "test-clevr-small", ["experiment=clevr_relative_trm", "model=clevr_condition_unet_trm_v2"])


# TRM Diffusion v2 ViT
@pytest.mark.cifar100
@pytest.mark.vit
@pytest.mark.trm
@pytest.mark.v2
def test_cond_cifar_vit_trm_v2():
    run_script(
        "train.py",
        "test-cifar100-cond-small",
        ["experiment=cond_cifar100_trm", "model=cifar100_vit_trm_v2", "~model.core_model.num_class_embeds"],
    )


@pytest.mark.imagenet
@pytest.mark.vit
@pytest.mark.trm
@pytest.mark.v2
def test_cond_imagenet_vit_trm_v2():
    run_script("train.py", "test-imagenet-cond-small", ["experiment=cond_imgnet_trm", "model=imagenet_vit_trm_v2"])


@pytest.mark.clevr
@pytest.mark.vit
@pytest.mark.trm
@pytest.mark.v2
def test_cond_clevr_vit_trm_v2():
    run_script("train.py", "test-clevr-small", ["experiment=clevr_relative_trm", "model=clevr_vit_trm_v2"])


# TRM Diffusion v3 UNet
@pytest.mark.cifar100
@pytest.mark.unet
@pytest.mark.trm
@pytest.mark.v3
def test_cond_cifar_unet_trm_v3():
    run_script(
        "train.py",
        "test-cifar100-cond-v3",
        ["experiment=cond_cifar100_trm", "model=unet2d_trm_v2", "model._target_=trm_models.UNetTRMv3"],
    )


@pytest.mark.imagenet
@pytest.mark.unet
@pytest.mark.trm
@pytest.mark.v3
def test_cond_imagenet_unet_trm_v3():
    run_script(
        "train.py",
        "test-imagenet-cond-v3",
        ["experiment=cond_imgnet_trm", "model=imagenet_condition_unet_trm_v2", "model._target_=trm_models.UNetTRMv3"],
    )


@pytest.mark.clevr
@pytest.mark.unet
@pytest.mark.trm
@pytest.mark.v3
def test_cond_clevr_unet_trm_v3():
    run_script(
        "train.py",
        "test-clevr-v3",
        ["experiment=clevr_relative_trm", "model=clevr_condition_unet_trm_v2", "model._target_=trm_models.UNetTRMv3"],
    )


# TRM Diffusion v3 ViT/DiT
@pytest.mark.cifar100
@pytest.mark.vit
@pytest.mark.trm
@pytest.mark.v3
def test_cond_cifar_vit_trm_v3():
    run_script(
        "train.py",
        "test-cifar100-cond-v3",
        [
            "experiment=cond_cifar100_trm",
            "model=cifar100_vit_trm_v2",
            "~model.core_model.num_class_embeds",
            "model._target_=trm_models.DiTTRMv3",
        ],
    )


@pytest.mark.imagenet
@pytest.mark.vit
@pytest.mark.trm
@pytest.mark.v3
def test_cond_imagenet_vit_trm_v3():
    run_script(
        "train.py",
        "test-imagenet-cond-v3",
        ["experiment=cond_imgnet_trm", "model=imagenet_vit_trm_v2", "model._target_=trm_models.DiTTRMv3"],
    )


@pytest.mark.clevr
@pytest.mark.vit
@pytest.mark.trm
@pytest.mark.v3
def test_cond_clevr_vit_trm_v3():
    run_script(
        "train.py",
        "test-clevr-v3",
        ["experiment=clevr_relative_trm", "model=clevr_vit_trm_v2", "model._target_=trm_models.DiTTRMv3"],
    )


# TRM Diffusion v4
@pytest.mark.cifar100
@pytest.mark.unet
@pytest.mark.trm
@pytest.mark.v4
def test_cond_cifar_unet_trm_v4():
    run_script(
        "train.py",
        "test-cifar100-cond-v4",
        ["experiment=cond_cifar100_trm", "model=unet2d_trm_v2", "model._target_=trm_models.UNetTRMv4"],
    )


@pytest.mark.imagenet
@pytest.mark.unet
@pytest.mark.trm
@pytest.mark.v4
def test_cond_imagenet_unet_trm_v4():
    run_script(
        "train.py",
        "test-imagenet-cond-v4",
        ["experiment=cond_imgnet_trm", "model=imagenet_condition_unet_trm_v2", "model._target_=trm_models.UNetTRMv4"],
    )


@pytest.mark.clevr
@pytest.mark.unet
@pytest.mark.trm
@pytest.mark.v4
def test_cond_clevr_unet_trm_v4():
    run_script(
        "train.py",
        "test-clevr-v4",
        ["experiment=clevr_relative_trm", "model=clevr_condition_unet_trm_v2", "model._target_=trm_models.UNetTRMv4"],
    )


@pytest.mark.cifar100
@pytest.mark.vit
@pytest.mark.trm
@pytest.mark.v4
def test_cond_cifar_vit_trm_v4():
    run_script(
        "train.py",
        "test-cifar100-cond-v4",
        [
            "experiment=cond_cifar100_trm",
            "model=cifar100_vit_trm_v2",
            "~model.core_model.num_class_embeds",
            "model._target_=trm_models.DiTTRMv4",
        ],
    )


@pytest.mark.imagenet
@pytest.mark.vit
@pytest.mark.trm
@pytest.mark.v4
def test_cond_imagenet_vit_trm_v4():
    run_script(
        "train.py",
        "test-imagenet-cond-v4",
        ["experiment=cond_imgnet_trm", "model=imagenet_vit_trm_v2", "model._target_=trm_models.DiTTRMv4"],
    )


@pytest.mark.clevr
@pytest.mark.vit
@pytest.mark.trm
@pytest.mark.v4
def test_cond_clevr_vit_trm_v4():
    run_script(
        "train.py",
        "test-clevr-v4",
        ["experiment=clevr_relative_trm", "model=clevr_vit_trm_v2", "model._target_=trm_models.DiTTRMv4"],
    )


# TRM Diffusion v5
@pytest.mark.cifar100
@pytest.mark.unet
@pytest.mark.trm
@pytest.mark.v5
def test_cond_cifar_unet_trm_v5():
    run_script(
        "train.py",
        "test-cifar100-cond-v5",
        ["experiment=cond_cifar100_trm", "model=unet2d_trm_v2", "model._target_=trm_models.UNetTRMv5"],
    )


@pytest.mark.imagenet
@pytest.mark.unet
@pytest.mark.trm
@pytest.mark.v5
def test_cond_imagenet_unet_trm_v5():
    run_script(
        "train.py",
        "test-imagenet-cond-v5",
        ["experiment=cond_imgnet_trm", "model=imagenet_condition_unet_trm_v2", "model._target_=trm_models.UNetTRMv5"],
    )


@pytest.mark.clevr
@pytest.mark.unet
@pytest.mark.trm
@pytest.mark.v5
def test_cond_clevr_unet_trm_v5():
    run_script(
        "train.py",
        "test-clevr-v5",
        ["experiment=clevr_relative_trm", "model=clevr_condition_unet_trm_v2", "model._target_=trm_models.UNetTRMv5"],
    )


@pytest.mark.cifar100
@pytest.mark.vit
@pytest.mark.trm
@pytest.mark.v5
def test_cond_cifar_vit_trm_v5():
    run_script(
        "train.py",
        "test-cifar100-cond-v5",
        [
            "experiment=cond_cifar100_trm",
            "model=cifar100_vit_trm_v2",
            "~model.core_model.num_class_embeds",
            "model._target_=trm_models.DiTTRMv5",
        ],
    )


@pytest.mark.imagenet
@pytest.mark.vit
@pytest.mark.trm
@pytest.mark.v5
def test_cond_imagenet_vit_trm_v5():
    run_script(
        "train.py",
        "test-imagenet-cond-v5",
        ["experiment=cond_imgnet_trm", "model=imagenet_vit_trm_v2", "model._target_=trm_models.DiTTRMv5"],
    )


@pytest.mark.clevr
@pytest.mark.vit
@pytest.mark.trm
@pytest.mark.v5
def test_cond_clevr_vit_trm_v5():
    run_script(
        "train.py",
        "test-clevr-v5",
        ["experiment=clevr_relative_trm", "model=clevr_vit_trm_v2", "model._target_=trm_models.DiTTRMv5"],
    )


# Ratatouille Decoupled Architecture (Thinker-Painter) UNet
@pytest.mark.cifar100
@pytest.mark.unet
@pytest.mark.ratatouille
def test_cond_cifar_ratatouille_concat():
    run_script(
        "train.py",
        "test-cifar100-rat-concat",
        ["experiment=cond_cifar100_trm", "model=cifar100_ratatouille_concat", "~model.core_model.num_class_embeds"],
    )


@pytest.mark.cifar100
@pytest.mark.unet
@pytest.mark.ratatouille
def test_cond_cifar_ratatouille_control():
    run_script(
        "train.py",
        "test-cifar100-rat-control",
        ["experiment=cond_cifar100_trm", "model=cifar100_ratatouille_control", "~model.core_model.num_class_embeds"],
    )


@pytest.mark.imagenet
@pytest.mark.unet
@pytest.mark.ratatouille
def test_cond_imagenet_ratatouille_control():
    run_script(
        "train.py", "test-imagenet-rat-control", ["experiment=cond_imgnet_trm", "model=imagenet_ratatouille_control"]
    )


@pytest.mark.clevr
@pytest.mark.unet
@pytest.mark.ratatouille
def test_cond_clevr_ratatouille_control():
    run_script(
        "train.py", "test-clevr-rat-control", ["experiment=clevr_relative_trm", "model=clevr_ratatouille_control"]
    )


@pytest.mark.clevr
@pytest.mark.unet
@pytest.mark.ratatouille
def test_cond_clevr_ratatouille_concat():
    run_script("train.py", "test-clevr-rat-concat", ["experiment=clevr_relative_trm", "model=clevr_ratatouille_concat"])


# Ratatouille Decoupled Architecture (Thinker-Painter) ViT / DiT
@pytest.mark.cifar100
@pytest.mark.vit
@pytest.mark.ratatouille
def test_cond_cifar_ratatouille_dit_concat():
    run_script(
        "train.py",
        "test-cifar100-rat-dit-concat",
        [
            "experiment=cond_cifar100_trm",
            "model=cifar100_ratatouille_dit_concat",
            "~model.core_model.num_class_embeds",
            "~model.core_model.num_classes",
        ],
    )


@pytest.mark.cifar100
@pytest.mark.vit
@pytest.mark.ratatouille
def test_cond_cifar_ratatouille_dit_residual():
    run_script(
        "train.py",
        "test-cifar100-rat-dit-residual",
        [
            "experiment=cond_cifar100_trm",
            "model=cifar100_ratatouille_dit_residual",
            "~model.core_model.num_class_embeds",
            "~model.core_model.num_classes",
        ],
    )


@pytest.mark.clevr
@pytest.mark.vit
@pytest.mark.ratatouille
def test_cond_clevr_ratatouille_dit_residual():
    run_script(
        "train.py", "test-clevr-rat-dit-res", ["experiment=clevr_relative_trm", "model=clevr_ratatouille_dit_residual"]
    )

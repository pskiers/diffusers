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
        "accelerate", "launch", "--mixed_precision=fp16", "--num_processes=1",
        script_name, "--output_dir", output_dir,
        "--train_batch_size", "4", "--eval_batch_size", "4",
        "--epoch_max_batches_train", "1", "--epoch_max_batches_eval", "1",
        "--save_model_epochs", "1", "--save_images_epochs", "1",
        "--ddpm_num_inference_steps", "1", "--checkpointing_steps", "1",
        "--logger", "wandb", "--gradient_accumulation_steps", "1",
        "--use_ema", "--learning_rate", "1e-4", "--lr_warmup_steps", "1"
    ]

    # 1. Run Initial Training (2 epochs to ensure checkpoints are written)
    cmd_initial = base_cmd + extra_args + ["--num_epochs", "2"]
    res_initial = subprocess.run(cmd_initial, env=env, capture_output=True, text=True)
    assert res_initial.returncode == 0, f"Initial Run Failed!\n{res_initial.stdout}\n{res_initial.stderr}"

    # 2. Test Checkpoint Resume & EMA Loading (Run for 1 more epoch)
    cmd_resume = base_cmd + extra_args + ["--num_epochs", "3", "--resume_from_checkpoint", "latest"]
    res_resume = subprocess.run(cmd_resume, env=env, capture_output=True, text=True)
    assert res_resume.returncode == 0, f"Resume Run Failed!\n{res_resume.stdout}\n{res_resume.stderr}"

    # 3. Cleanup on success
    shutil.rmtree(output_dir, ignore_errors=True)


# def test_unconditional_standard():
#     run_script("train_unconditional.py", "test-cifar100-standard", [
#         "--dataset_name", "uoft-cs/cifar100",
#         "--resolution", "32", "--center_crop", "--random_flip"
#     ])


# def test_unconditional_small_loop():
#     run_script("train_unconditional_small_loop.py", "test-cifar100-small", [
#         "--dataset_name", "uoft-cs/cifar100",
#         "--resolution", "32", "--center_crop", "--random_flip",
#         "--channels", "128", "128",
#         "--down_block_types", "AttnDownBlock2D", "DownBlock2D",
#         "--up_block_types", "UpBlock2D", "AttnUpBlock2D",
#         "--T", "4", "--n", "6", "--N_supervision", "1"
#     ])


def test_conditional_standard_cifar():
    run_script("train_conditional.py", "test-cifar100-cond", [
        "--dataset_name", "uoft-cs/cifar100",
        "--resolution", "32", "--center_crop", "--random_flip",
        "--num_classes", "100"
    ])


def test_conditional_standard_imagenet():
    run_script("train_conditional.py", "test-imagenet-cond", [
        "--dataset_name", "ILSVRC/imagenet-1k",
        "--resolution", "256", "--center_crop", "--random_flip",
        "--num_classes", "1000",
        "--vae_name", "stabilityai/sdxl-vae",
        "--image_key", "image", "--class_key", "label",
        "--input_channels", "4",
        "--test_split_name", "validation",
        "--cache_dir", "cache_dir"
    ])


def test_conditional_small_loop_cifar():
    run_script("train_conditional_small_loop.py", "test-cifar100-cond-small", [
        "--dataset_name", "uoft-cs/cifar100",
        "--resolution", "32", "--center_crop", "--random_flip",
        "--channels", "128", "128",
        "--down_block_types", "AttnDownBlock2D", "DownBlock2D",
        "--up_block_types", "UpBlock2D", "AttnUpBlock2D",
        "--T", "3", "--n", "6", "--N_supervision", "4",
        "--num_classes", "100"
    ])


def test_conditional_small_loop_imagenet():
    run_script("train_conditional_small_loop.py", "test-imagenet-cond-small", [
        "--dataset_name", "ILSVRC/imagenet-1k",
        "--resolution", "256", "--center_crop", "--random_flip",
        "--channels", "256", "256",
        "--down_block_types", "DownBlock2D", "CrossAttnDownBlock2D",
        "--up_block_types", "CrossAttnUpBlock2D", "UpBlock2D",
        "--T", "3", "--n", "6", "--N_supervision", "4",
        "--num_classes", "1000",
        "--vae_name", "stabilityai/sdxl-vae",
        "--image_key", "image", "--class_key", "label",
        "--input_channels", "4",
        "--test_split_name", "validation",
        "--cache_dir", "cache_dir"
    ])


def test_clevr_standard():
    run_script("train_clevr.py", "test-clevr-standard", [
        "--train_data_dir", "cache_dir",
        "--resolution", "256",
        "--num_classes", "1000",
        "--vae_name", "stabilityai/sdxl-vae",
        "--input_channels", "4",
        "--test_split_name", "validation",
        "--cache_dir", "cache_dir",
        "--dataset_mode", "absolute"
    ])


def test_clevr_small_loop():
    run_script("train_clevr_small_loop.py", "test-clevr-small", [
        "--train_data_dir", "cache_dir",
        "--resolution", "256",
        "--channels", "128", "128",
        "--down_block_types", "CrossAttnDownBlock2D", "DownBlock2D",
        "--up_block_types", "UpBlock2D", "CrossAttnUpBlock2D",
        "--T", "3", "--n", "6", "--N_supervision", "4",
        "--num_classes", "1000",
        "--vae_name", "stabilityai/sdxl-vae",
        "--image_key", "image", "--class_key", "label",
        "--input_channels", "4",
        "--test_split_name", "validation",
        "--cache_dir", "cache_dir",
        "--dataset_mode", "relative"
    ])

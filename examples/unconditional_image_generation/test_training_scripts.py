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
        script_name,
        f"output_dir={output_dir}",
        "train_batch_size=4", "eval_batch_size=4",
        "epoch_max_batches_train=1", "epoch_max_batches_eval=1",
        "save_model_epochs=1", "save_images_epochs=1",
        "ddpm_num_inference_steps=1", "checkpointing_steps=1",
        "logger=wandb", "gradient_accumulation_steps=1",
        "use_ema=true", "learning_rate=1e-4", "lr_warmup_steps=1"
    ]

    cmd_initial = base_cmd + extra_args + ["num_epochs=2"]
    res_initial = subprocess.run(cmd_initial, env=env, capture_output=True, text=True)
    assert res_initial.returncode == 0, f"Initial Run Failed!\n{res_initial.stdout}\n{res_initial.stderr}"

    cmd_resume = base_cmd + extra_args + ["num_epochs=3", "resume_from_checkpoint=latest"]
    res_resume = subprocess.run(cmd_resume, env=env, capture_output=True, text=True)
    assert res_resume.returncode == 0, f"Resume Run Failed!\n{res_resume.stdout}\n{res_resume.stderr}"

    shutil.rmtree(output_dir, ignore_errors=True)


def test_unconditional_standard():
    run_script("train.py", "test-cifar100-standard", ["experiment=uncond_cifar100_std"])

def test_unconditional_small_loop():
    run_script("train.py", "test-cifar100-small", ["experiment=uncond_cifar100_trm"])

def test_conditional_standard_cifar():
    run_script("train.py", "test-cifar100-cond", ["experiment=cond_cifar100_std"])

def test_conditional_standard_imagenet():
    run_script("train.py", "test-imagenet-cond", ["experiment=cond_imgnet_std"])

def test_conditional_small_loop_cifar():
    run_script("train.py", "test-cifar100-cond-small", ["experiment=cond_cifar100_trm"])

def test_conditional_small_loop_imagenet():
    run_script("train.py", "test-imagenet-cond-small", ["experiment=cond_imgnet_trm"])

def test_clevr_standard():
    run_script("train.py", "test-clevr-standard", ["experiment=clevr_relative_std"])

def test_clevr_small_loop():
    run_script("train.py", "test-clevr-small", ["experiment=clevr_relative_trm"])

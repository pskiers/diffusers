# TRM Diffusion
## Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:

**Important**

To make sure you can successfully run the latest versions of the example scripts, we highly recommend **installing from source** and keeping the install up to date as we update the example scripts frequently and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:
```bash
git clone https://github.com/pskiers/diffusers.git
cd diffusers
git checkout pskiers/trm-diffusion
pip install .
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.4cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
```

Then cd in the example folder  and run
```bash
pip install -r requirements.txt
```


Initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

And login to [Wandb](https://wandb.ai/) with:
```bash
wandb login
```

### Optional: closed-loop eval simulators (PushT / BlockPush / ToolHang)

Not required for training — `models/dp_eval_callbacks.py` gracefully skips
(logs a warning, returns `{}`) any callback whose simulator isn't installed.
Only needed for the periodic/standalone closed-loop coverage/success-rate
metrics on those three tasks.

```bash
# PushT
pip install gym-pusht pymunk shapely

# ToolHang — pinned to match upstream real-stanford/diffusion_policy's own
# conda_environment.yaml as closely as possible (current robosuite releases
# have drifted API-wise from the version they benchmarked against):
pip install "robosuite @ https://github.com/cheng-chi/robosuite/archive/277ab9588ad7a4f4b55cf75508b44aa67ec171f0.tar.gz"
pip install robomimic==0.2.0
# Simpler but unverified-compatible alternative: pip install robosuite robomimic

# BlockPush — `block_pushing` is not on PyPI; it lives inside
# google-research/ibc and must be put on PYTHONPATH:
git clone https://github.com/google-research/ibc.git third_party/ibc
export PYTHONPATH="$PYTHONPATH:$(pwd)/third_party/ibc/environments"
```

## Training

### CIFAR-100 Unconditional
```bash
# Standard Diffusion Unconditional
accelerate launch --mixed_precision="fp16" --num_processes=1 train.py experiment=uncond_cifar100_std

# TRM Diffusion Unconditional
accelerate launch --mixed_precision="fp16" --num_processes=1 train.py experiment=uncond_cifar100_trm
```

### CIFAR-100 Conditional
```bash
# Standard Conditional
accelerate launch --mixed_precision="fp16" --num_processes=1 train.py experiment=cond_cifar100_std

# Small Loop (TRM) Conditional
accelerate launch --mixed_precision="fp16" --num_processes=1 train.py experiment=cond_cifar100_trm
```

### ImageNet
```bash
# Standard Conditional
accelerate launch --mixed_precision="fp16" --num_processes=1 train.py experiment=cond_imgnet_std

# Small Loop (TRM) Conditional
accelerate launch --mixed_precision="fp16" --num_processes=1 train.py experiment=cond_imgnet_trm
```

### Clevr
```bash
# Standard CLEVR
accelerate launch --mixed_precision="fp16" --num_processes=1 train.py experiment=clevr_relative_std

# Small Loop (TRM) CLEVR
accelerate launch --mixed_precision="fp16" --num_processes=1 train.py experiment=clevr_relative_trm
```

## Sampling


### CIFAR-100 Unconditional
```bash
# Standard Diffusion Unconditional
accelerate launch --mixed_precision="fp16" sample.py \
  experiment=uncond_cifar100_std \
  checkpoint_path="cifar100-standard-long/checkpoint-<STEP>" \
  num_samples=10000 \
  sample_batch_size=250

# TRM Diffusion Unconditional
accelerate launch --mixed_precision="fp16" sample.py \
  experiment=uncond_cifar100_trm \
  checkpoint_path="cifar100-size-small-nsup1-long/checkpoint-<STEP>" \
  num_samples=10000 \
  sample_batch_size=5000
```

### CIFAR-100 Conditional
```bash
# Standard Conditional
accelerate launch --mixed_precision="fp16" sample.py \
  experiment=cond_cifar100_std \
  checkpoint_path="cifar100-conditional-standard/checkpoint-<STEP>" \
  num_samples=10000 \
  sample_batch_size=250

# Small Loop (TRM) Conditional
accelerate launch --mixed_precision="fp16" sample.py \
  experiment=cond_cifar100_trm \
  checkpoint_path="cifar100-conditional-hrm-nsup-4-T-3-n-6/checkpoint-<STEP>" \
  num_samples=10000 \
  sample_batch_size=5000
```

### ImageNet
```bash
# Standard Conditional
accelerate launch --mixed_precision="fp16" sample.py \
  experiment=cond_imgnet_std \
  checkpoint_path="imagenet-conditional-standard-big/checkpoint-<STEP>" \
  num_samples=10000 \
  sample_batch_size=64

# Small Loop (TRM) Conditional
accelerate launch --mixed_precision="fp16" sample.py \
  experiment=cond_imgnet_trm \
  checkpoint_path="imagenet-conditional-hrm-nsup-4-T-3-n-6-fix-ch256/checkpoint-<STEP>" \
  num_samples=10000 \
  sample_batch_size=64
```

### Clevr
```bash
# Standard CLEVR
accelerate launch --mixed_precision="fp16" sample.py \
  experiment=clevr_relative_std \
  checkpoint_path="clevr-standard-att-early-cross-att-relative-fix/checkpoint-<STEP>" \
  num_samples=10000 \
  sample_batch_size=64

# Small Loop (TRM) CLEVR
accelerate launch --mixed_precision="fp16" sample.py \
  experiment=clevr_relative_trm \
  checkpoint_path="clevr-hrm-nsup-4-T-3-n-6-cross-att-pred-eps-relative-fix/checkpoint-<STEP>" \
  num_samples=10000 \
  sample_batch_size=64
```
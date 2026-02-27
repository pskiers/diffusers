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

#!/bin/bash
# TEMPORARY (Claude): discrete token-per-cell TRM (train_discrete_trm.py) on titan. Resumable: rerun to continue.
# Usage: train_discrete_titan.sh <gpu (nvidia-smi index)> [out_dir]
set -uo pipefail
cd ~/diffusers_interp/examples/trm_diffusion
source ~/trm_helios_venv/bin/activate
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD:$PWD/experiments:$HOME/diffusers_interp/stubs" CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1
OUT=${2:-runs/claude_discrete_trm}; mkdir -p $OUT
python experiments/train_discrete_trm.py --train data/discrete/train.npz --test_dir data/discrete --test_sizes 5,8,10,11,13,16 \
  --out_dir $OUT --steps 40000 --keep_steps 4000,10000,20000,30000 >> $OUT/train_stdout.txt 2>&1

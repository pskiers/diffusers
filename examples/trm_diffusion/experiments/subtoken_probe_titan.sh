#!/bin/bash
# TEMPORARY (Claude): interp_subtoken_probe.py (or SCRIPT=<file in experiments/>) on titan. Usage: subtoken_probe_titan.sh <gpu (nvidia-smi index)> <out_dir> [extra hydra overrides...]
set -euo pipefail
cd ~/diffusers_interp/examples/trm_diffusion
GPU=$1; OUT=$2; shift 2
source ~/trm_helios_venv/bin/activate
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD:$PWD/experiments:$HOME/diffusers_interp/stubs" CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
mkdir -p $OUT
python experiments/${SCRIPT:-interp_subtoken_probe.py} experiment=amaze_thinker_v2_controlnet \
  painter.checkpoint=runs/pt_maze_final_painter/checkpoint_final.pt \
  +interprebability_config.checkpoint=runs/pt_maze_final_thinker/checkpoint_final.pt \
  +interprebability_config.out_dir=$OUT hydra.run.dir=$OUT/hydra "$@"

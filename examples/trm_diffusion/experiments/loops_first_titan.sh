#!/bin/bash
# TEMPORARY (Claude): interp_loops_first.py (or SCRIPT=<file in experiments/>) on titan. Usage: loops_first_titan.sh <gpu> <plain|path> <out_dir> [extra hydra overrides...]
# env PLAIN_CKPT / PATH_CKPT override the checkpoints (default: the originals)
# GPU numbers as in nvidia-smi (PCI order; the CUDA default orders the fastest card first)
set -euo pipefail
cd ~/diffusers_interp/examples/trm_diffusion
GPU=$1; KIND=$2; OUT=$3; shift 3
source ~/trm_helios_venv/bin/activate
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD:$PWD/experiments:$HOME/diffusers_interp/stubs" CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
COMMON="experiment=amaze_thinker_v2_controlnet painter.checkpoint=runs/pt_maze_final_painter/checkpoint_final.pt
  +interprebability_config.data_parquet=data/amaze/maze/square/n8_test.parquet +interprebability_config.out_dir=$OUT"
if [ "$KIND" = plain ]; then
  M="+interprebability_config.model_kind=plain +interprebability_config.plain_checkpoint=${PLAIN_CKPT:-runs/claude_plain_trm/checkpoint_final.pt} +interprebability_config.checkpoint=none"
else
  M="+interprebability_config.model_kind=path +interprebability_config.checkpoint=${PATH_CKPT:-runs/pt_maze_final_thinker/checkpoint_final.pt}"
fi
python experiments/${SCRIPT:-interp_loops_first.py} $COMMON $M hydra.run.dir=$OUT/hydra_$KIND "$@"

#!/bin/bash
# TEMPORARY (Claude): interp_compare_states.py on titan. Usage: states_titan.sh <gpu (nvidia-smi index)>
set -uo pipefail
GPU=$1
cd ~/diffusers_interp/examples/trm_diffusion
export PLAIN_CKPTS=plain:runs/claude_plain_trm/checkpoint_final.pt,plain4k:runs/claude_plain_trm/checkpoint_step4000.pt,noisy:runs/claude_plain_trm_noisy/checkpoint_final.pt
for spec in "8 random 0" "11 longest 10"; do
  set -- $spec; OUT=runs/claude_states/n$1_$2; mkdir -p $OUT
  SCRIPT=interp_compare_states.py bash experiments/loops_first_titan.sh $GPU path $OUT ++interprebability_config.data_parquet=data/amaze/maze/square/n$1_test.parquet +interprebability_config.maze_size=$1 +interprebability_config.side_pick=$2 +interprebability_config.min_side=$3 > $OUT/log.txt 2>&1
done
echo ALLDONE > runs/claude_states/done

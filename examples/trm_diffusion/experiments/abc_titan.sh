#!/bin/bash
# TEMPORARY (Claude): mismatched x_t / conditioning (interp_loops_xt.py), z_H update decomposition (interp_loops_zh_decomp.py),
# encoder probes + CKA across models (interp_compare_models.py). Usage: abc_titan.sh <gpu (nvidia-smi index)> <part: A|B>
set -uo pipefail
cd ~/diffusers_interp/examples/trm_diffusion
GPU=$1; PART=$2
B=runs/claude_abc
export PLAIN_CKPTS=plain:runs/claude_plain_trm/checkpoint_final.pt,plain4k:runs/claude_plain_trm/checkpoint_step4000.pt,noisy:runs/claude_plain_trm_noisy/checkpoint_final.pt,mse:runs/claude_plain_trm_mse/checkpoint_final.pt
for spec in "11 longest 10" "8 random 0"; do
  set -- $spec; N=$1; PICK=$2; MIN=$3; OUT=$B/n${N}_${PICK}; mkdir -p $OUT
  ARGS="++interprebability_config.data_parquet=data/amaze/maze/square/n${N}_test.parquet +interprebability_config.maze_size=$N +interprebability_config.side_pick=$PICK +interprebability_config.min_side=$MIN"
  if [ "$PART" = A ]; then
    SCRIPT=interp_compare_models.py bash experiments/loops_first_titan.sh $GPU path $OUT $ARGS > $OUT/log_compare.txt 2>&1
    SCRIPT=interp_loops_zh_decomp.py bash experiments/loops_first_titan.sh $GPU plain $OUT $ARGS > $OUT/log_zh_decomp_plain.txt 2>&1
  else
    SCRIPT=interp_loops_xt.py bash experiments/loops_first_titan.sh $GPU path $OUT $ARGS > $OUT/log_xt.txt 2>&1
    SCRIPT=interp_loops_zh_decomp.py bash experiments/loops_first_titan.sh $GPU path $OUT $ARGS > $OUT/log_zh_decomp_path.txt 2>&1
  fi
done
echo ALLDONE > $B/done_$PART

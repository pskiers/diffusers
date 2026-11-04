#!/bin/bash
# TEMPORARY (Claude): strategy of the loop-trained models, same analyses as for the originals (runs/claude_loops_first,
# helios runs/claude_loops2): interp_loops_first.py (first guess / pruning dynamics, update split, attention range, heads)
# and check_loops.py (drawn = shortest path vs dead-end-filling result). Waits for PID $3 (the eval on that GPU) first.
# Usage: loops_strategy_titan.sh <gpu> <plain|path> <pid to wait for>
set -uo pipefail
cd ~/diffusers_interp/examples/trm_diffusion
GPU=$1; KIND=$2; WAIT=$3; OUT=runs/claude_loops_strategy
mkdir -p $OUT
while kill -0 $WAIT 2>/dev/null; do sleep 30; done
export PLAIN_CKPT=runs/claude_plain_trm_loops/checkpoint_last.pt PATH_CKPT=runs/claude_thinker_loops/checkpoint_final.pt
BS=$([ "$KIND" = plain ] && echo 132 || echo 16)
bash experiments/loops_first_titan.sh $GPU $KIND $OUT > $OUT/log_first_$KIND.txt 2>&1
SCRIPT=check_loops.py bash experiments/loops_first_titan.sh $GPU $KIND $OUT/check +interprebability_config.num_boards=100 \
  +interprebability_config.batch_size=$BS > $OUT/log_check_$KIND.txt 2>&1
echo done > $OUT/done_$KIND

#!/bin/bash
# Separate Python environment for the T2I-CoReBench VLM judge
# (eval/corebench_judge.py): Qwen3-VL needs newer vLLM/torch/transformers than
# training. Versions as tested (vLLM 0.31 runs on CUDA 13: NVIDIA driver >= 580).
#
#   bash scripts/setup_judge_env.sh /path/to/judge-env
#   export COREBENCH_JUDGE_PYTHON=/path/to/judge-env/bin/python
#
# The judge downloads its weights on first use (Qwen3-VL-32B-Thinking: ~66 GB)
# into the Hugging Face cache — point HF_HOME at a disk with room.
set -euo pipefail
ENV_DIR=${1:?usage: setup_judge_env.sh /path/to/judge-env}
python3 -m venv "$ENV_DIR"
"$ENV_DIR/bin/pip" install --upgrade pip
"$ENV_DIR/bin/pip" install "vllm==0.31.0" "qwen-vl-utils==0.0.14"
"$ENV_DIR/bin/python" -c "import vllm, qwen_vl_utils; print('judge env ready: vllm', vllm.__version__)"

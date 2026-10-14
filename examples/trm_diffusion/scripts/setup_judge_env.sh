#!/bin/bash
# Separate Python environment for the T2I-CoReBench VLM judge
# (eval/corebench_judge.py): Qwen3-VL needs newer vLLM/torch/transformers than
# training. Versions as tested (vLLM 0.31 runs on CUDA 13: NVIDIA driver >= 580).
#
#   bash scripts/setup_judge_env.sh /path/to/judge-env
#   PYTHON=/path/to/python3.11+ bash scripts/setup_judge_env.sh /path/to/judge-env   # another interpreter
#   export COREBENCH_JUDGE_PYTHON=/path/to/judge-env/bin/python
#
# Python >= 3.11: early 3.10 releases (e.g. 3.10.4) have an inspect.getsource bug
# (it stops at a lambda inside a decorator) that breaks Triton kernels vLLM imports
# during its kernel warmup ("'NoneType' object has no attribute 'start'").
#
# The judge downloads its weights on first use (Qwen3-VL-32B-Thinking: ~66 GB)
# into the Hugging Face cache — point HF_HOME at a disk with room.
set -euo pipefail
ENV_DIR=${1:?usage: setup_judge_env.sh /path/to/judge-env}
"${PYTHON:-python3}" -c "import sys; sys.exit(sys.version_info < (3, 11))" \
  || { echo "judge env needs Python >= 3.11 (see above)" >&2; exit 1; }
"${PYTHON:-python3}" -m venv "$ENV_DIR"
"$ENV_DIR/bin/pip" install --upgrade pip
"$ENV_DIR/bin/pip" install "vllm==0.31.0" "qwen-vl-utils==0.0.14"
"$ENV_DIR/bin/python" -c "import vllm, qwen_vl_utils; print('judge env ready: vllm', vllm.__version__)"

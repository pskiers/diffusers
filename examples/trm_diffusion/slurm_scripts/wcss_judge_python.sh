#!/bin/bash
# COREBENCH_JUDGE_PYTHON for WCSS: the judge env ($JUDGE_ENV) is built on the
# Python/3.12.3 module (3.10.4 breaks vLLM) and needs it loaded to run, while the
# training process that starts the judge has the 3.10.4 module loaded. The module
# changes only apply to this judge process, not to the training that started it.
source /usr/local/sbin/modules.sh
module purge >/dev/null 2>&1
module load Python/3.12.3-GCCcore-13.3.0 >/dev/null 2>&1
exec "${JUDGE_ENV:?JUDGE_ENV is not set}/bin/python" "$@"

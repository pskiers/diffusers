#!/bin/bash -l
# Vendor the AMAZE fine-tuning glue code into third_party/amaze/ and clone the
# Bagel & Janus base model repos where the SFT scripts expect them.
# Run ONCE on a login node (needs internet), like the data generator setup.
#
#   bash third_party/amaze/setup_ft_code.sh
#
# Env: AMAZE_REPO (default the public spatigen/amaze), SKIP_BASE=1 to skip
#      cloning the large Bagel/Janus base repos.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"        # .../third_party/amaze
AMAZE_REPO="${AMAZE_REPO:-https://github.com/spatigen/amaze.git}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo ">> cloning ${AMAZE_REPO}"
git clone --depth 1 "${AMAZE_REPO}" "${TMP}/amaze"

# The sft/, data/ and infer/ glue is COMMITTED to this repo and locally modified
# (e.g. sft/janus/sft.py + sft/bagel/sft.py carry our validation / wandb changes).
# Do NOT clobber it on re-runs. Set FORCE_VENDOR=1 to re-copy upstream (overwrites edits).
if [[ "${FORCE_VENDOR:-0}" == "1" || ! -f "${HERE}/sft/janus/sft.py" ]]; then
  echo ">> copying sft/, data/maze_dataset.py and infer/ glue into ${HERE}"
  mkdir -p "${HERE}/sft" "${HERE}/data" "${HERE}/infer"
  cp -r "${TMP}/amaze/sft/." "${HERE}/sft/"
  cp    "${TMP}/amaze/data/maze_dataset.py" "${HERE}/data/"
  cp -r "${TMP}/amaze/infer/." "${HERE}/infer/"
else
  echo ">> glue already present (committed + locally modified) -> keeping it."
  echo "   (set FORCE_VENDOR=1 to overwrite with upstream)"
fi

if [[ "${SKIP_BASE:-0}" != "1" ]]; then
  echo ">> cloning base model repos (large)"
  [[ -d "${HERE}/sft/bagel/Bagel" ]] || git clone --depth 1 https://github.com/ByteDance-Seed/Bagel.git "${HERE}/sft/bagel/Bagel"
  [[ -d "${HERE}/sft/janus/Janus" ]] || git clone --depth 1 https://github.com/deepseek-ai/Janus.git "${HERE}/sft/janus/Janus"
else
  echo ">> SKIP_BASE=1 -> not cloning Bagel/Janus base repos"
fi

echo "Done."
echo "  glue code : ${HERE}/{sft,infer,data}"
echo "  base repos: ${HERE}/sft/bagel/Bagel , ${HERE}/sft/janus/Janus"

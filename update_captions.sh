#!/usr/bin/env bash
# ==============================================================================
# Krea 2 VLM Caption Refinement Runner (Google Gemma 4 12B)
# Tailored for RTX PRO 6000 (96GB VRAM) on RunPod
# Upgrades dataset captions to 550-1024 tokens with exhaustive bondage taxonomy
# ==============================================================================
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Activate Virtual Environment if available
if [ -f ".venv/bin/activate" ]; then
    # shellcheck source=/dev/null
    source .venv/bin/activate
fi

# 2. Environment optimizations
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_USE_V1=0

# 3. Parse dataset target and additional arguments
TARGET="${1:-restrained_elegance}"
EXTRA_ARGS=()

if [[ "$1" == -* ]]; then
    # First argument is a flag (e.g. --force), default target to restrained_elegance
    TARGET="restrained_elegance"
    EXTRA_ARGS=("$@")
elif [ -n "$1" ]; then
    shift
    EXTRA_ARGS=("$@")
fi

# Determine whether target is a directory path or a dataset name
DATASET_FLAG="--dataset-name"
if [ -d "$TARGET" ] || [[ "$TARGET" == *"/"* ]] || [[ "$TARGET" == *"\\"* ]]; then
    DATASET_FLAG="--dataset-dir"
fi

TEMPERATURE="0.75"
MAX_TOKENS="340"
MIN_TOKENS="150"
PARALLEL_SUB_BATCH="16"
TIERS="30,40,30"

echo "========================================================================"
echo " Starting Krea 2 Caption Refinement (Google Gemma 4 12B)"
echo " Target Target   : ${TARGET} (${DATASET_FLAG})"
echo " GPU VRAM Target : RTX PRO 6000 (96GB VRAM)"
echo " Engine          : Google Gemma 4 12B (google/gemma-4-12B-it)"
echo " Output Mode     : Pure Raw Text (Zero Token Waste, Programmatic JSON)"
echo " Tier Mix        : 30% Tags | 40% Short (max 150 tok) | 30% Dense (max 340 tok)"
echo " Sampling Temp   : ${TEMPERATURE}"
echo " Parallel Batch  : ${PARALLEL_SUB_BATCH} images concurrent"
echo " Taxonomy Focus  : Model Position | Bondage Type | Equipment | Rigging"
echo "========================================================================"

python3 scripts/update_dataset_captions.py \
    "${DATASET_FLAG}" "${TARGET}" \
    --temperature "${TEMPERATURE}" \
    --max-tokens "${MAX_TOKENS}" \
    --min-tokens "${MIN_TOKENS}" \
    --parallel-sub-batch "${PARALLEL_SUB_BATCH}" \
    --tiers "${TIERS}" \
    "${EXTRA_ARGS[@]}"

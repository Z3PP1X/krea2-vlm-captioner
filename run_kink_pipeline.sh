#!/usr/bin/env bash
# ==============================================================================
# Kink Collection: 9 Channels Automated Pipeline Execution Script
# Tailored for RTX PRO 6000 (96GB VRAM, 141GB RAM, 16 vCPUs)
# ==============================================================================
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Activate Virtual Environment if available
if [ -f ".venv/bin/activate" ]; then
    # shellcheck source=/dev/null
    source .venv/bin/activate
fi

# 2. Optimize environment & ensure dependencies
pip install -r requirements.txt -U "transformers>=4.49.0" accelerate imagehash beautifulsoup4 -q 2>/dev/null || true
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_USE_V1=0
export TOKENIZERS_PARALLELISM=false

echo "========================================================================"
echo " Starting Kink Collection End-to-End Pipeline (Stages 1-5)"
echo " GPU Target : RTX PRO 6000 (96GB VRAM)"
echo " Model      : Google Gemma 4 12B Multimodal (google/gemma-4-12B-it)"
echo " Engine     : Native Hugging Face AutoModelForMultimodalLM (bfloat16)"
echo " Batch Size : 32"
echo " Workers    : 16"
echo " Min Res    : 256px"
echo " Dataset    : kink_collection -> /app/ai-toolkit/datasets/kink_collection"
echo "========================================================================"

# Run Python orchestrator, passing through any extra flags (e.g. --skip-crawl, --batch-size)
python3 scripts/run_kink_collection.py "$@"

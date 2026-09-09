#!/usr/bin/env bash
# ==============================================================================
# Alternative: High-Throughput vLLM Server on RunPod
# ==============================================================================
# Use this on RunPod with an RTX 4090 (24GB) or A6000 / A100 (48/80GB)
# for continuous batching and 5x-10x faster tagging than sequential Ollama.
set -e

pip install vllm Pillow requests tqdm

MODEL_ID="${1:-Qwen/Qwen2.5-VL-7B-Instruct}"
PORT="${2:-8000}"

echo "Starting vLLM OpenAI-compatible server with model: $MODEL_ID on port $PORT"
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --port "$PORT" \
    --trust-remote-code \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --enforce-eager

# Client command:
# python3 caption_images.py -i /workspace/images -t 'restrained_elegance' --backend openai --url http://localhost:8000/v1 -m "$MODEL_ID" -w 4

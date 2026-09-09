#!/usr/bin/env bash
# ==============================================================================
# RunPod Environment Setup Script for Krea 2 VLM Captioner
# ==============================================================================
set -e

echo "=== [1/4] Updating apt and installing dependencies ==="
apt-get update -y && apt-get install -y curl git python3-pip

echo "=== [2/4] Installing Python requirements ==="
pip install --upgrade pip 2>/dev/null || true
pip install -r requirements.txt --break-system-packages 2>/dev/null || pip install -r requirements.txt

echo "=== [3/4] Checking / Installing Ollama ==="
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "Ollama is already installed."
fi

# Start Ollama service in background if not already running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama daemon in background..."
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    sleep 3
fi

echo "=== [4/4] Pulling Vision Model ==="
# Default vision model (can be replaced with your model tag, e.g. gemma:latest, llama3.2-vision, or gemma4:32b)
MODEL_NAME="${1:-gemma:latest}"
echo "Pulling model '$MODEL_NAME' (this may take a few minutes depending on network)..."
ollama pull "$MODEL_NAME"

echo ""
echo "=============================================================================="
echo " Setup complete! Ollama is running and model '$MODEL_NAME' is ready."
echo ""
echo " Example Usage:"
echo "   python3 caption_images.py -i /workspace/my_images -t 'restrained_elegance' -m '$MODEL_NAME'"
echo "=============================================================================="

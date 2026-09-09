#!/usr/bin/env bash
# ==============================================================================
# RunPod Environment Setup Script for Krea 2 LoRA Pipeline
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
chmod +x "${SCRIPT_DIR}/install.sh"
exec "${SCRIPT_DIR}/install.sh" --gpu "$@"


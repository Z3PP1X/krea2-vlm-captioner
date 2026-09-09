#!/usr/bin/env bash
# ==============================================================================
# Complete Installation & Virtual Environment Setup Script
# Krea 2 LoRA End-to-End Data Pipeline
# ==============================================================================
set -eo pipefail

# Text formatting
BOLD="\033[1m"
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
RESET="\033[0m"

log_info()    { echo -e "${BLUE}${BOLD}[INFO]${RESET} $1"; }
log_success() { echo -e "${GREEN}${BOLD}[SUCCESS]${RESET} $1"; }
log_warn()    { echo -e "${YELLOW}${BOLD}[WARN]${RESET} $1"; }
log_error()   { echo -e "${RED}${BOLD}[ERROR]${RESET} $1"; }

# Default parameters
VENV_PATH=".venv"
FORCE_GPU=false
FORCE_CPU=false
RECREATE_VENV=false
RUN_TESTS=true
AI_TOOLKIT_PATH="/app/ai-toolkit"

# Parse CLI arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            FORCE_GPU=true
            shift
            ;;
        --cpu)
            FORCE_CPU=true
            shift
            ;;
        --recreate)
            RECREATE_VENV=true
            shift
            ;;
        --venv-path)
            VENV_PATH="$2"
            shift 2
            ;;
        --no-test)
            RUN_TESTS=false
            shift
            ;;
        -h|--help)
            echo "Usage: ./install.sh [options]"
            echo ""
            echo "Options:"
            echo "  --gpu             Force GPU installation (CUDA PyTorch + vLLM)"
            echo "  --cpu             Force CPU-only installation"
            echo "  --recreate        Delete existing virtual environment and create a fresh one"
            echo "  --venv-path PATH  Specify custom venv directory (default: .venv)"
            echo "  --no-test         Skip running pytest verification at the end"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *)
            log_warn "Unknown argument '$1', ignoring."
            shift
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${BOLD}======================================================================${RESET}"
echo -e "${BOLD}     Krea 2 LoRA Pipeline: Automated Environment Installation         ${RESET}"
echo -e "${BOLD}======================================================================${RESET}"

# ------------------------------------------------------------------------------
# 1. System Package & Dependency Checks
# ------------------------------------------------------------------------------
log_info "[1/7] Checking system dependencies..."

# In container environments (RunPod/Docker), install system packages if missing
if command -v apt-get &> /dev/null && [ "$(id -u)" -eq 0 ]; then
    log_info "Updating apt packages and verifying system tools (git, curl, libgl1)..."
    apt-get update -y -qq
    apt-get install -y -qq git curl ffmpeg libsm6 libxext6 libgl1 python3-venv python3-pip > /dev/null 2>&1 || true
fi

# ------------------------------------------------------------------------------
# 2. Python Binary Detection
# ------------------------------------------------------------------------------
log_info "[2/7] Detecting suitable Python binary..."

PYTHON_BIN=""
for candidate in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$candidate" &> /dev/null; then
        VER=$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "0.0")
        MAJOR=$(echo "$VER" | cut -d. -f1)
        MINOR=$(echo "$VER" | cut -d. -f2)
        if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 10 ]; then
            PYTHON_BIN="$candidate"
            log_info "Found supported Python version: $($candidate --version) ($PYTHON_BIN)"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    log_error "No Python >= 3.10 found on PATH. Please install Python 3.10, 3.11, or 3.12."
    exit 1
fi

# ------------------------------------------------------------------------------
# 3. Virtual Environment Creation
# ------------------------------------------------------------------------------
log_info "[3/7] Setting up Python Virtual Environment at '${VENV_PATH}'..."

if [ "$RECREATE_VENV" = true ] && [ -d "$VENV_PATH" ]; then
    log_warn "Removing existing virtual environment as requested by --recreate..."
    rm -rf "$VENV_PATH"
fi

if [ ! -d "$VENV_PATH" ]; then
    log_info "Creating new virtual environment using $PYTHON_BIN..."
    "$PYTHON_BIN" -m venv "$VENV_PATH" || {
        log_error "Failed to create virtual environment. Ensure 'python3-venv' is installed."
        exit 1
    }
else
    log_info "Reusing existing virtual environment at '${VENV_PATH}'."
fi

# Activate virtual environment
# shellcheck source=/dev/null
source "${VENV_PATH}/bin/activate"

log_info "Upgrading pip, setuptools, and wheel inside venv..."
pip install --upgrade pip setuptools wheel -q

# ------------------------------------------------------------------------------
# 4. Hardware Detection & GPU Acceleration Setup
# ------------------------------------------------------------------------------
log_info "[4/7] Detecting hardware configuration..."

HAS_NVIDIA=false
if [ "$FORCE_CPU" = false ]; then
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        HAS_NVIDIA=true
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 || echo "NVIDIA GPU")
        log_success "NVIDIA GPU detected: ${GPU_NAME}"
    elif [ "$FORCE_GPU" = true ]; then
        HAS_NVIDIA=true
        log_warn "GPU forced via --gpu flag, proceeding with CUDA dependencies."
    fi
fi

if [ "$HAS_NVIDIA" = true ]; then
    log_info "Installing PyTorch with CUDA support and vLLM acceleration..."
    # Ensure numpy<2 is installed to maintain C-ABI compatibility across PyTorch & vLLM
    pip install "numpy<2" -q
    
    # Install PyTorch
    log_info "Installing PyTorch (CUDA)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -q || {
        log_warn "cu124 wheel failed, falling back to standard PyPI torch..."
        pip install torch torchvision torchaudio -q
    }

    # Install GPU requirements (vLLM, Transformers, Accelerate)
    log_info "Installing vLLM and multimodal dependencies..."
    pip install -r requirements-gpu.txt -q
else
    log_info "Configuring CPU-only environment..."
    pip install "numpy<2" -q
    pip install -r requirements.txt -q
fi

# ------------------------------------------------------------------------------
# 5. Pipeline Editable Installation
# ------------------------------------------------------------------------------
log_info "[5/7] Installing Krea 2 Pipeline CLI in editable mode ('pip install -e .')..."
pip install -e . -q

# Create global symlink if running as root on RunPod/Linux so 'pipeline' works everywhere
if [ "$(id -u)" -eq 0 ] && [ -w /usr/local/bin ]; then
    ln -sf "${PWD}/${VENV_PATH}/bin/pipeline" /usr/local/bin/pipeline
    log_success "Symlinked '${PWD}/${VENV_PATH}/bin/pipeline' -> /usr/local/bin/pipeline"
fi

# ------------------------------------------------------------------------------
# 6. Initialize Pipeline Directory Hierarchy
# ------------------------------------------------------------------------------
log_info "[6/7] Initializing project folder structure..."
mkdir -p data/raw data/qc data/processed data/export logs config

# If AI-Toolkit directory exists (e.g. on RunPod /app/ai-toolkit), ensure datasets dir is ready
if [ -d "$AI_TOOLKIT_PATH" ]; then
    mkdir -p "${AI_TOOLKIT_PATH}/datasets"
    log_info "Verified AI-Toolkit dataset directory: ${AI_TOOLKIT_PATH}/datasets"
fi

# ------------------------------------------------------------------------------
# 7. Verification & Self-Test
# ------------------------------------------------------------------------------
if [ "$RUN_TESTS" = true ]; then
    log_info "[7/7] Running self-test test suite..."
    python -m pytest tests/ -q || {
        log_warn "Some unit tests did not pass. Check test logs."
    }
else
    log_info "[7/7] Skipping test suite (--no-test specified)."
fi

# ------------------------------------------------------------------------------
# Completion Banner & Instructions
# ------------------------------------------------------------------------------
echo ""
echo -e "${BOLD}======================================================================${RESET}"
echo -e "${GREEN}${BOLD}             Installation Completed Successfully!                     ${RESET}"
echo -e "${BOLD}======================================================================${RESET}"
echo ""
echo -e "To activate the environment in your shell, run:"
echo -e "  ${BOLD}${BLUE}source ${VENV_PATH}/bin/activate${RESET}"
echo ""
echo -e "Quick Start Commands:"
echo -e "  ${BOLD}pipeline status${RESET}                                        # Show pipeline status"
echo -e "  ${BOLD}pipeline crawl --url \"https://scrolller.com/...\"${RESET}      # Crawl dataset"
echo -e "  ${BOLD}pipeline qc${RESET}                                            # Run QC & deduplication"
echo -e "  ${BOLD}pipeline downscale${RESET}                                     # Downscale to max 2048px"
echo -e "  ${BOLD}pipeline caption -m gemma4 -t \"trigger_name\"${RESET}          # Multimodal captioning (Gemma 4)"
echo -e "  ${BOLD}pipeline export --dataset-name \"my_dataset\"${RESET}           # Export to AI-Toolkit"
echo ""
echo -e "${BOLD}======================================================================${RESET}"

# Krea 2 LoRA Data Pipeline & Tooling 🚀

A production-grade, modular, idempotent, and resumable end-to-end data pipeline and training tooling tailored specifically for training LoRAs on the **Krea 2 (K2)** generative foundation model (single-stream MMDiT conditioned via Qwen3-VL text encoder).

Engineered for modern high-VRAM hardware (**NVIDIA L40S 48GB** and **RTX 5090 32GB**) with primary integration for [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit) and companion support for `musubi-tuner`.

---

## 📑 Table of Contents

- [Architectural Overview](#-architectural-overview)
- [8-Stage Pipeline Design](#-8-stage-pipeline-design)
- [Manifest & State Architecture](#-manifest--state-architecture)
- [Quick Start & Installation](#-quick-start--installation)
- [CLI Reference](#-cli-reference)
- [Training Runbook & Tooling](#-training-runbook--tooling)
- [Documentation Index](#-documentation-index)
- [Testing & Verification](#-testing--verification)

---

## 🏛 Architectural Overview

```
[Sites / Sources]
       │ (HTTP Range Sniffer > 0.7 MP)
       ▼
[Stufe 1: Crawl & Pre-Filter] ────► data/raw/
       │
       ▼
[Stufe 2: QC & Deduplication] ────► sRGB, EXIF Strip, Laplacian Sharpness, pHash Deduplication
       │
       ▼
[Stufe 3: DiT Downscale]      ────► Max 2048px, Multiples of 16 (data/processed/)
       │
       ▼
[Stufe 4: Qwen-VL Inference]  ────► vLLM Offline Batching, Guided JSON, Mandatory Screening Gate
       │
       ▼
[Stufe 5: Balance & Export]   ────► Dynamic Repeats, Distribution Report, AI-Toolkit / Musubi Configs
       │
       ▼
[Stufen 6-8: Train & Validate]───► Krea 2 RAW Training, Validation Grids, Sequential Character Stack
```

---

## 🔄 8-Stage Pipeline Design

### 1. Stufe 1 – Crawling mit Vorfilter
- **HTTP Range Sniffer**: Reads image header bytes before download; skips any image under 0.7 MP (~1024x700) with zero bandwidth waste.
- **Compliance & Rate Limiting**: Honors `robots.txt`, detects `§ 44b UrhG` (EU TDM reservation) opt-outs, and enforces domain-level rate limiting with exponential backoff.
- **Extractors**: Native XenForo forum thread extractor and dbNaked high-res scene extractor.

### 2. Stufe 2 – Quality Control & Deduplication
- **Sanitization**: Strict sRGB color profile normalization and lossless EXIF metadata stripping.
- **Sharpness Gate**: Laplacian variance metric (threshold $\ge 100$) rejects blurry and low-detail frames.
- **Watermark & Noise**: Edge-band luminance variance detection flags embedded site watermarks and banners.
- **Perceptual Deduplication**: 64-bit DCT pHash clustering with Hamming distance $\le 6$ groups near-duplicate frames into equivalence classes, retaining only the sharpest instance.

### 3. Stufe 3 – Downscaling & Multiples of 16
- **DiT & VAE Alignment**: Ensures both dimensions are exact multiples of 16 to prevent latent boundary artifacts during MMDiT patchification.
- **Interpolation**: High-quality Lanczos resampling with strict aspect-ratio preservation down to a maximum bounding box of 2048px.

### 4. Stufe 4 – Qwen-VL Captioning & Screening Gate
- **High-Throughput Inference**: Offline batching via `vLLM` using `Qwen/Qwen2.5-VL-7B-Instruct` (or Qwen3-VL) with Guided JSON schema decoding (`config/vocabulary.yaml`).
- **Non-Negotiable Compliance Gate**: Immediately rejects and logs any image failing:
  - `subject_age_estimate < 25` or `uncertain_age == True`
  - `watermark_detected == True` or text overlay
  - `quality == "low"`
- **Prompt Assembly**: Assembles 7-layer narrative captions (`Trigger` -> `Medium` -> `Subject` -> `Pose` -> `Wardrobe` -> `Environment` -> `Lighting` -> `Optics`). Supports `style` mode (omits style descriptors to let weights absorb aesthetic) and `subject` mode (describes style fully to isolate the subject).
- **Inspection HTML**: Generates `inspection_report.html` for human audit of captions, tags, and confidence scores.

### 5. Stufe 5 – Datensatz strukturieren & Export
- **Dynamic Repeat Balancing**: Analyzes tag distribution across scenes/identities and assigns inverse-frequency repeat weights (clamped to $\le 4.0\times$) to prevent over-represented concepts from dominating.
- **Distribution Analysis**: Flags dominance alerts whenever any single scene or subject exceeds 35% of total dataset exposure; generates `distribution_report.md`.
- **Toolkit Formats**: Exports directory structures and configurations directly for `ai-toolkit` (`ai_toolkit_dataset.yaml`) and `musubi-tuner` (`dataset.toml`).

### 6. Stufe 6 – Trainings-Tooling (AI-Toolkit)
- **Krea 2 RAW LoRA Template**: Standardized configuration for single-stream MMDiT on Krea 2 RAW (undistilled base model) with `fp8` base quantization, LoRA rank 128 / alpha 128, learning rate `5e-5`, and full gradient checkpointing.
- **Curated 2–3k Subset Extractor**: Selects the top 2,500 highest-quality, balanced samples for rapid A/B convergence benchmarking against the full dataset.

### 7. Stufe 7 – Validierung & Benchmarking
- **Systematic Validation Matrix**: Generates a 3-tier prompt grid (`trigger_only`, `trigger_with_core_tags`, `trigger_with_scene`) crossed against negative prompts (`none` vs. `standard`).
- **HTML Contact Sheet**: Generates `contact_sheet.html` with interactive slider comparison across checkpoints (e.g. 500, 1000, 1500, 2000 steps).

### 8. Stufe 8 – Charakter-LoRAs im selben Universum
- **Sequential Training Stack**: Freezes the base style LoRA and trains secondary character/subject LoRAs on top of the stylized latent space.
- **Weight Matrix Grid**: Automates evaluation matrix across style weights ($0.6 - 1.0$) and character weights ($0.6 - 1.0$) to verify aesthetic consistency without facial identity collapse.

---

## 🗃 Manifest & State Architecture

All pipeline stages are coordinated through an atomic, append-only JSONL manifest: `data/manifest.jsonl`.

- **Idempotency**: Every stage checks if an image is already processed before executing. Re-running any stage only processes newly added or pending items.
- **Non-Destructive Rejection**: Rejected items are never deleted; their stage status is marked `rejected` alongside a structured `reject_reason` (e.g., `screening_age_under_25`, `low_sharpness_42.1`, `duplicate_of_<id>`).
- **Real-Time Monitoring**: Run `pipeline status` at any time to inspect counts across all 8 stages.

---

## ⚡ Quick Start & Installation

### Local / Development Setup (CPU / Verification)
```bash
git clone https://github.com/Z3PP1X/krea2-vlm-captioner.git
cd krea2-vlm-captioner

# Install package and standard dependencies
pip install -e .

# Run test suite to verify installation
python -m pytest
```

### Production GPU Setup (NVIDIA L40S / RTX 5090 on RunPod)
```bash
# Install with vLLM & PyTorch GPU acceleration
pip install -e ".[gpu]"

# Or install flash-infer/vllm dependencies directly
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install vllm
```

---

## 💻 CLI Reference

The unified CLI entrypoint is `pipeline` (or `python -m pipeline.cli`):

```bash
# 1. Check pipeline manifest status
pipeline status

# 2. Stufe 1: Crawl with HTTP Range pre-filtering
pipeline crawl --source-type xenforo --url "https://forum.example.com/threads/123" --pages 1-5
pipeline crawl --source-type dbnaked --url "https://dbnaked.com/bdsm/channels/example" --pages 1

# 3. Stufe 2: Quality control, EXIF strip & pHash deduplication
pipeline qc

# 4. Stufe 3: Downscale to max 2048px (multiples of 16)
pipeline downscale --max-dim 2048

# 5. Stufe 4: Qwen-VL Guided JSON Captioning & Screening Gate
pipeline caption --model Qwen/Qwen2.5-VL-7B-Instruct --batch-size 16 --mode style

# 6. Stufe 5: Balance repeats and export datasets
pipeline export --max-factor 4.0 --target-repeats 10

# 7. Stufe 6: Generate AI-Toolkit Krea 2 RAW training config
pipeline train-config --name "krea2_style_lora" --dataset-yaml "data/export/ai_toolkit_dataset.yaml"

# 8. Stufe 7: Generate systematic validation prompt grid & contact sheet
pipeline validate --lora-name "krea2_style_lora"

# 9. Stufe 8: Generate sequential character LoRA configs
pipeline character-config --style-lora "output/krea2_style_lora/krea2_style_lora.safetensors" --character-name "eva"

# Or execute Stages 1 to 5 end-to-end:
pipeline run-all
```

---

## 🚀 Training Runbook & Tooling

Complete execution instructions for training on RunPod are provided in [docs/TRAINING.md](docs/TRAINING.md):

- **Network Volume Base Model Caching**: Persist Krea 2 RAW weights on `/workspace` so new pods spin up in seconds.
- **AI-Toolkit Execution**: Step-by-step launch commands, monitoring with TensorBoard / WandB, and checkpointing.
- **VRAM Optimization**:
  - **L40S (48 GB)**: `fp8` base model, rank 128 / alpha 128, batch size 2, gradient accumulation 2.
  - **RTX 5090 (32 GB)**: `fp8` base model, rank 128 / alpha 128, batch size 1, gradient accumulation 4, gradient checkpointing enabled.
- **Curated 2-3k vs Full Set Protocol**: Initial 1,500-step run on the curated subset to lock in learning rates and loss stability before scaling to the full dataset.

---

## 📚 Documentation Index

- [docs/GAP_ANALYSIS.md](docs/GAP_ANALYSIS.md): Comprehensive baseline evaluation of the legacy repository vs. the target architecture.
- [docs/MIGRATION_PLAN.md](docs/MIGRATION_PLAN.md): Detailed 3-phase technical migration blueprint covering all 8 stages.
- [docs/TRAINING.md](docs/TRAINING.md): End-to-end RunPod operator runbook, hardware profiles, AI-Toolkit and musubi configurations, and cost calculations.
- [config/pipeline.yaml](config/pipeline.yaml): Centralized configuration for all pipeline parameters.
- [config/vocabulary.yaml](config/vocabulary.yaml): Controlled vocabulary schema for Pydantic and Guided JSON decoding.

---

## 🧪 Testing & Verification

The repository includes a comprehensive unit test suite covering every stage:

```bash
python -m pytest -v
```

```text
tests/test_manifest.py           3 passed
tests/test_stage1_prefilter.py   6 passed
tests/test_stage2_qc.py          3 passed
tests/test_stage3_downscale.py   3 passed
tests/test_stage4_assembly.py    3 passed
tests/test_stage4_screening.py   4 passed
tests/test_stage5_export.py      3 passed
tests/test_stage6_7_8.py         4 passed
====================== 29 passed in 0.89s ======================
```

---

## 📄 License
MIT License.

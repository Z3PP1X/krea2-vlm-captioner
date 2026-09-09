# Krea 2 VLM Image Captioner & Crawler 🚀

An end-to-end pipeline that crawls image sets from XenForo threads and dbNaked galleries/channels, groups them into set folders with `genres.txt` metadata, and batch-captions every image using a Vision-Language Model (Gemma, PaliGemma, Qwen-VL) running on RunPod.

Tailored specifically for the **Krea 2 (K2)** generative foundation model, utilizing its **Qwen3-VL text encoder** and **7-layer natural language narrative prompting system**.

---

## 🌟 Key Features

1. **Multi-Site Image Crawler (`crawler.py`)**:
   - **XenForo Native Support** (`xxx-files.org`, etc.): Automatically downloads original-sized image sets into separate folders and writes a `genres.txt` containing forum tags and metadata for each set.
   - **dbNaked Native Support** (`dbnaked.com`): Automatically scrapes channels, galleries, and direct scenes, downloads full 1600x1600 resolution images into scene folders, and generates `genres.txt` containing categories, tags, and performer models.
2. **Automated `genres.txt` Context Injection**: When captioning an image, the captioner automatically detects `genres.txt` in that set directory and feeds the set title and tags (`Set Context & Tags: Shibari, Rope Bondage, Corset...`) into the VLM prompt. This eliminates hallucinations and grounds materials, wardrobe, and actions.
3. **Trigger Word Prepending**: Guarantees your custom trigger token (e.g., `restrained_elegance`, `shibori`) is placed as token 0 before `Photograph of...` for optimal Krea 2 LoRA attention.
4. **Clean Plain Text Output**: Extracts the dense 7-layer narrative caption directly into `<image>.txt` alongside each image—ready for immediate LoRA training in Kohya, AI-Toolkit, or OneTrainer.
5. **Unified 1-Command Pipeline**: Run the crawler and immediately auto-tag all crawled sets with a single CLI command (`crawl_and_tag.py`).

---

## 📁 Repository Structure

```
krea2-vlm-captioner/
├── crawl_and_tag.py          # Unified end-to-end pipeline (Auto-detects XenForo or dbNaked)
├── dbnaked_crawler.py        # dbNaked high-res (1600x1600) channel & gallery crawler
├── crawler.py                # XenForo forum thread image crawler
├── caption_images.py         # Krea 2 VLM Batch Captioner (reads images + genres.txt -> .txt)
├── setup_runpod.sh           # 1-click RunPod setup script (Ollama + model pull)
├── vllm_server_runpod.sh     # vLLM continuous batching startup script
├── requirements.txt          # Python dependencies
├── .gitignore                # Excludes raw images, venvs, and temp files
└── README.md                 # Documentation & quick start guide
```

---

## ⚡ Quick Start on RunPod

### 1. Setup on RunPod
Inside your RunPod terminal (e.g. `/workspace`):
```bash
cd /workspace
git clone https://github.com/Z3PP1X/krea2-vlm-captioner.git
cd krea2-vlm-captioner

chmod +x setup_runpod.sh
./setup_runpod.sh gemma:latest   # or your preferred VLM (e.g. gemma4:32b)
```

### 2. Run the Unified Crawl & Tag Job

**XenForo Thread:**
```bash
python3 crawl_and_tag.py --pages 27 -t "restrained_elegance" -m "gemma:latest"
```

**dbNaked Channel:**
```bash
python3 crawl_and_tag.py --url "https://dbnaked.com/bdsm/channels/thetrainingofo.com?media=pictures" --pages 1 -t "restrained_elegance"
```

## 🔄 How the Crawl & Tag Flow Works

```
[Forum Thread] 
      │
      ▼
1. XenForo Crawler
      ├── Downloads full-res images into set folders
      └── Writes 'genres.txt' (Tags: Latex, Bondage, Corset / Title: Set 112)
      │
      ▼
2. Folder Structure
      └── downloads/
            └── Models Tied Gallery 112/
                  ├── genres.txt
                  ├── image_01.jpg
                  └── image_02.jpg
      │
      ▼
3. Krea 2 VLM Captioner
      ├── Reads image_01.jpg
      ├── Reads genres.txt context: "Tags: Latex, Bondage; Set: Models Tied 112"
      ├── Injects Trigger Token: "restrained_elegance"
      └── Feeds prompt + image to Gemma / VLM
      │
      ▼
4. Output Dataset
      └── downloads/
            └── Models Tied Gallery 112/
                  ├── genres.txt
                  ├── image_01.jpg
                  ├── image_01.txt    <-- Pure 7-layer narrative caption
                  ├── image_02.jpg
                  └── image_02.txt    <-- Pure 7-layer narrative caption
```

### Example Generated `.txt` File:
```text
restrained_elegance Photograph of a slender woman with fair skin and straight blonde hair, kneeling gracefully on all fours with an arched back on a black glossy reflective floor. She is wearing polished stainless steel wrist cuffs connected to a taut shiny silver chain. The background is a pitch-black studio void creating stark contrast. Focused directional studio spotlighting creates sculptural highlights along her back. Crisp 50mm lens focus with shallow depth of field.
```

---

## 💻 CLI Commands & Options

### 1. `crawl_and_tag.py` (Unified Pipeline)
```bash
python crawl_and_tag.py [CRAWLER OPTIONS] [CAPTIONER OPTIONS]
```

* `--pages` / `-p`: Pages to crawl (`27`, `11-15`, `all`).
* `-t` / `--trigger`: LoRA trigger word (e.g. `restrained_elegance`, `shibori`).
* `-o` / `--output-dir`: Output folder (default: `./downloads`).
* `-m` / `--model`: Model name in Ollama / vLLM (default: `gemma4:32b`).
* `-u` / `--url-vlm`: Inference URL (default: `http://localhost:11434`).
* `--skip-crawl`: Skip downloading; only caption existing folders in `--output-dir`.
* `--skip-caption`: Only crawl and download without running the captioner.
* `--save-json`: Save `<image>.json` with full 7-layer scene breakdown alongside `<image>.txt`.

### 2. `caption_images.py` (Standalone Captioning)
Run captioning independently on any folder or pre-existing downloads:
```bash
# Tag an existing crawler downloads folder recursively:
python caption_images.py -i ./downloads -r -t "restrained_elegance"

# Tag a single set directory:
python caption_images.py -i "./downloads/Models Tied Gallery 112" -t "shibori"
```

### 3. `dbnaked_crawler.py` (dbNaked Channel & Gallery Crawler)
Crawl any dbNaked channel or studio directly:
```bash
# Crawl page 1 of Infernal Restraints channel:
python dbnaked_crawler.py --url "https://dbnaked.com/bdsm/channels/infernalrestraints.com" -p 1 -o ./downloads

# Crawl pages 1 to 3 with 6 worker threads:
python dbnaked_crawler.py --url "https://dbnaked.com/bdsm/channels/infernalrestraints.com" -p 1-3 -w 6 -o ./downloads
```

### 4. `crawler.py` (Standalone XenForo Forum Crawler)
Run the XenForo crawler independently:
```bash
python crawler.py --pages 27 -o ./downloads
```

---

## 🚀 High-Speed Batching with vLLM on RunPod

For maximum throughput with continuous batching:
1. Launch vLLM server:
   ```bash
   chmod +x vllm_server_runpod.sh
   ./vllm_server_runpod.sh Qwen/Qwen2.5-VL-7B-Instruct 8000
   ```
2. Run pipeline with `--backend openai`:
   ```bash
   python3 crawl_and_tag.py \
     --pages 27 \
     -t "restrained_elegance" \
     --backend openai \
     --url-vlm http://localhost:8000/v1 \
     -m "Qwen/Qwen2.5-VL-7B-Instruct" \
     --caption-workers 4
   ```

---

## 📄 License
MIT License.

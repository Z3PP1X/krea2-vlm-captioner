#!/usr/bin/env python3
"""
Krea 2 VLM Image Captioner CLI
================================
Batch captions images in a specified folder using a Vision-Language Model
(Gemma / PaliGemma / Qwen-VL via Ollama or vLLM / OpenAI API).

Tailored specifically for Krea 2's Qwen3-VL text encoder & MMDiT architecture.
Integrates automatically with XenForo crawler set folders and 'genres.txt' metadata.
Produces clean .txt files containing natural language 7-layer captions.

Usage:
  python caption_images.py -i /path/to/images -t restrained_elegance
  python caption_images.py -i "downloads/Set A" -t "shibori" --tier dense
  python caption_images.py -i ./downloads -r -t "caning" --url "http://localhost:11434"
"""

import os
import re
import io
import sys
import json
import base64
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# 7-Layer Krea 2 System Prompt
KREA2_SYSTEM_PROMPT = """You are an expert visual annotator, director of photography, and synthetic dataset engineer specializing in the Krea 2 generative foundation model (Qwen3-VL text encoder).

Your mission is to inspect the provided image and generate an objective, highly detailed visual annotation strictly conforming to Krea 2's 7-layer prompt architecture.

### STRICT INSTRUCTIONS:
1. ZERO CONVERSATIONAL FILLER: Never start with "In this image", "We see", "This photograph shows", or "Here is a caption". Begin immediately with the Medium declaration or the designated Trigger token.
2. OBJECTIVE VISUAL GROUNDING: Annotate solely what is physically visible. Do not infer backstory, invisible motivations, or unseen context.
3. PHYSICAL & MATERIAL ACCURACY: Use concrete terminology (e.g. "matte black leather with silver nickel buckles", "directional sunlight through slatted blinds", "shallow depth of field with creamy background falloff"). Avoid vague aesthetic clichés like "beautiful", "gorgeous", or "breathtaking".
4. OBJECTIVE / CLINICAL ACCURACY: For adult, sensual, bondage, or impact imagery, use neutral, professional art-direction vocabulary (e.g. "reddened skin with raised linear marks", "taut shiny steel chains", "natural hemp rope with visible fibers").
5. SET CONTEXT UTILIZATION: When "Set Context & Tags" are provided, use them to accurately recognize materials, themes, and wardrobe (e.g. distinguishing latex vs leather, shibari knots vs chain hardware) without hallucinating unseen objects.
6. MULTI-TIER RECURSION: Generate three discrete caption lengths to empower multi-scale DiT LoRA training and generation.

### OUTPUT JSON FORMAT:
Respond with ONLY valid JSON:
{
  "caption_dense": "Full 7-layer grammatically complete narrative (70-120 words).",
  "caption_mid": "Condensed core summary: Subject + Action + Hardware/Attire + Setting (30-50 words).",
  "caption_short": "Minimal core anchor (10-20 words).",
  "tags": ["concise", "visual", "keywords"],
  "scene_breakdown": {
    "medium": "...",
    "subject": "...",
    "pose": "...",
    "materials_and_hardware": "...",
    "environment": "...",
    "lighting": "...",
    "optics": "..."
  }
}"""

logger = logging.getLogger("captioner")


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def find_set_genres(image_path: Path) -> Optional[str]:
    """
    Checks if a genres.txt file exists in the image directory or parent directory.
    Extracts the 'Genres: ...' line or 'Tags: ...' list and set title to pass as context.
    """
    candidates = [
        image_path.parent / "genres.txt",
        image_path.parent.parent / "genres.txt",
    ]
    for c in candidates:
        if c.is_file():
            try:
                with open(c, "r", encoding="utf-8") as f:
                    content = f.read()

                m_genres = re.search(r"^Genres:\s*(.+)$", content, re.MULTILINE | re.IGNORECASE)
                m_title = re.search(r"^Set Title:\s*(.+)$", content, re.MULTILINE | re.IGNORECASE)

                parts = []
                if m_title and m_title.group(1).strip() and m_title.group(1).strip().lower() != "unknown set":
                    parts.append(f"Set: {m_title.group(1).strip()}")
                if m_genres and m_genres.group(1).strip() and m_genres.group(1).strip().upper() != "N/A":
                    parts.append(f"Tags: {m_genres.group(1).strip()}")

                if parts:
                    return "; ".join(parts)
            except Exception:
                pass
    return None


def encode_image(image_path: Path, max_size: int = 1024) -> str:
    """Reads, optionally downscales, and base64-encodes an image."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            new_size = (int(w * scale), int(h * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=90)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def clean_json_response(raw_text: str) -> Dict[str, Any]:
    """Robustly extracts and parses JSON from model output."""
    raw_text = raw_text.strip()

    # Strip markdown code blocks if present
    if "```json" in raw_text:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
        if match:
            raw_text = match.group(1).strip()
    elif raw_text.startswith("```") and raw_text.endswith("```"):
        raw_text = raw_text.strip("`").strip()

    # Attempt direct json loads
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    # Try finding the outermost JSON object braces
    brace_match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(1))
        except json.JSONDecodeError:
            pass

    # Fallback if model output unstructured plain text
    return {"caption_dense": raw_text}


def build_user_prompt(trigger_token: Optional[str] = None, set_context: Optional[str] = None) -> str:
    """Constructs user prompt combining optional genres.txt context and trigger token."""
    parts = []
    if set_context:
        parts.append(f"Set Context & Tags: {set_context}")

    trigger_str = trigger_token.strip() if trigger_token else None
    if trigger_str:
        parts.append(f"Trigger Token: {trigger_str}")
        parts.append(
            f"Task: Inspect this image and generate the Krea 2 multi-tier captions conforming to the 7-layer schema. "
            f"Prepend the trigger token '{trigger_str}' as the very first word in the captions (e.g. '{trigger_str} Photograph of...'). "
            f"Use the Set Context & Tags to accurately identify materials, styling, and scene details without hallucination."
        )
    else:
        parts.append(
            "Task: Inspect this image and generate the Krea 2 multi-tier captions conforming to the 7-layer schema. "
            "Begin directly with the Medium declaration (e.g. 'Photograph of...'). "
            "Use the Set Context & Tags (if provided) to accurately identify materials, styling, and scene details without hallucination."
        )

    return "\n".join(parts)


def query_ollama(
    base_url: str,
    model: str,
    base64_image: str,
    trigger_token: Optional[str] = None,
    set_context: Optional[str] = None,
    timeout: int = 180,
) -> Dict[str, Any]:
    """Sends inference request to Ollama /api/chat."""
    user_prompt = build_user_prompt(trigger_token, set_context)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": KREA2_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": user_prompt,
                "images": [base64_image],
            },
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
            "num_predict": 512,
        },
    }

    url = f"{base_url.rstrip('/')}/api/chat"
    resp = requests.post(url, json=payload, timeout=timeout)
    if resp.status_code != 200:
        try:
            err_data = resp.json()
            err_msg = err_data.get("error", resp.text)
        except Exception:
            err_msg = resp.text
        raise RuntimeError(f"Ollama Error ({resp.status_code}): {err_msg}")

    result = resp.json()
    message_content = result.get("message", {}).get("content", "")
    return clean_json_response(message_content)


def query_openai_compat(
    base_url: str,
    model: str,
    base64_image: str,
    trigger_token: Optional[str] = None,
    set_context: Optional[str] = None,
    timeout: int = 180,
) -> Dict[str, Any]:
    """Sends inference request to vLLM / OpenAI-compatible /v1/chat/completions."""
    user_text = build_user_prompt(trigger_token, set_context)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": KREA2_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            },
        ],
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 512,
    }

    endpoint = base_url.rstrip("/")
    if not endpoint.endswith("/v1"):
        endpoint = f"{endpoint}/v1"
    url = f"{endpoint}/chat/completions"

    resp = requests.post(url, json=payload, timeout=timeout)
    if resp.status_code != 200:
        try:
            err_data = resp.json()
            err_msg = err_data.get("error", {})
            if isinstance(err_msg, dict):
                err_msg = err_msg.get("message", resp.text)
        except Exception:
            err_msg = resp.text
        raise RuntimeError(f"API Error ({resp.status_code}): {err_msg}")

    result = resp.json()
    message_content = result["choices"][0]["message"]["content"]
    return clean_json_response(message_content)


def extract_caption_tier(data: Dict[str, Any], tier: str) -> str:
    """Extracts the requested caption tier, falling back safely."""
    tier_key = f"caption_{tier}" if not tier.startswith("caption_") else tier
    if tier_key in data and isinstance(data[tier_key], str) and data[tier_key].strip():
        return data[tier_key].strip()

    # Fallback hierarchy: dense -> mid -> short -> raw string
    for fallback in ["caption_dense", "caption_mid", "caption_short"]:
        if fallback in data and isinstance(data[fallback], str) and data[fallback].strip():
            return data[fallback].strip()

    if "description" in data:
        return str(data["description"]).strip()

    return str(data)


def process_single_image(
    image_path: Path,
    args: argparse.Namespace,
) -> Tuple[Path, bool, str]:
    """Processes an individual image: finds set genres, encodes, queries VLM, writes .txt."""
    txt_path = image_path.with_suffix(".txt")
    json_path = image_path.with_suffix(".json")

    # Resume check: skip if .txt exists and not overwrite
    if txt_path.exists() and not args.overwrite:
        return image_path, True, "Already exists (skipped)"

    try:
        # Detect genres.txt in current set directory
        set_context = find_set_genres(image_path)
        b64_img = encode_image(image_path, max_size=args.max_size)

        if args.backend == "openai":
            json_data = query_openai_compat(
                base_url=args.url,
                model=args.model,
                base64_image=b64_img,
                trigger_token=args.trigger,
                set_context=set_context,
                timeout=args.timeout,
            )
        else:
            json_data = query_ollama(
                base_url=args.url,
                model=args.model,
                base64_image=b64_img,
                trigger_token=args.trigger,
                set_context=set_context,
                timeout=args.timeout,
            )

        # Extract selected caption tier
        caption_text = extract_caption_tier(json_data, tier=args.tier)

        # Ensure trigger token is prepended if user requested and model omitted it
        if args.trigger:
            trigger_clean = args.trigger.strip()
            if not caption_text.startswith(trigger_clean):
                caption_text = f"{trigger_clean} {caption_text}"

        # Write the plain text file
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(caption_text + "\n")

        # Optionally save full JSON breakdown
        if args.save_json:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

        info_tag = f" [{set_context}]" if set_context else ""
        return image_path, True, f"Saved ({args.tier}){info_tag}: {caption_text[:60]}..."

    except Exception as exc:
        return image_path, False, str(exc)


def caption_directory(
    directory: Path,
    args: argparse.Namespace,
) -> Tuple[int, int, int]:
    """Captions all images in a directory or recursively across sets."""
    pattern = "**/*" if args.recursive else "*"
    all_files = list(directory.glob(pattern))
    image_files = [f for f in all_files if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]

    if not image_files:
        logger.warning(f"No images found in {directory} matching {IMAGE_EXTENSIONS}")
        return 0, 0, 0

    logger.info("=" * 60)
    logger.info("  KREA 2 VLM BATCH CAPTIONER")
    logger.info("=" * 60)
    logger.info(f"Target Directory : {directory.resolve()}")
    logger.info(f"Images Found     : {len(image_files)}")
    logger.info(f"Trigger Token    : {args.trigger or '(None)'}")
    logger.info(f"Caption Tier     : {args.tier}")
    logger.info(f"Model            : {args.model}")
    logger.info(f"Backend & URL    : {args.backend} @ {args.url}")
    logger.info(f"Workers          : {args.workers}")
    logger.info(f"Auto-Genres      : Enabled (checks for genres.txt)")
    logger.info("=" * 60)

    success_count = 0
    skip_count = 0
    error_count = 0

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_img = {
                executor.submit(process_single_image, img, args): img
                for img in image_files
            }
            iterator = as_completed(future_to_img)
            if HAS_TQDM and not args.verbose:
                iterator = tqdm(iterator, total=len(image_files), desc="Captioning")

            for future in iterator:
                img_path, ok, msg = future.result()
                if ok:
                    if "skipped" in msg.lower():
                        skip_count += 1
                        if not HAS_TQDM or args.verbose:
                            logger.info(f"[SKIP] {img_path.name}")
                    else:
                        success_count += 1
                        if not HAS_TQDM or args.verbose:
                            logger.info(f"[DONE] {img_path.name} -> {msg}")
                else:
                    error_count += 1
                    logger.error(f"[ERR]  {img_path.name} -> {msg}")
    else:
        iterator = image_files
        if HAS_TQDM and not args.verbose:
            iterator = tqdm(image_files, desc="Captioning")

        for idx, img in enumerate(iterator, 1):
            if not HAS_TQDM or args.verbose:
                logger.info(f"[{idx}/{len(image_files)}] Processing {img.name}...")
            img_path, ok, msg = process_single_image(img, args)
            if ok:
                if "skipped" in msg.lower():
                    skip_count += 1
                    if not HAS_TQDM or args.verbose:
                        logger.info(f"       -> {msg}")
                else:
                    success_count += 1
                    if not HAS_TQDM or args.verbose:
                        logger.info(f"       -> {msg}")
            else:
                error_count += 1
                logger.error(f"       -> FAILED: {msg}")

    logger.info("=" * 60)
    logger.info("  CAPTIONING COMPLETED")
    logger.info(f"  Successfully Processed: {success_count}")
    logger.info(f"  Skipped (Existing)    : {skip_count}")
    logger.info(f"  Errors                : {error_count}")
    logger.info("=" * 60)

    return success_count, skip_count, error_count


def main():
    parser = argparse.ArgumentParser(
        description="Batch caption images using a Vision Model (Gemma/PaliGemma/Qwen-VL) tailored for Krea 2."
    )
    parser.add_argument(
        "-i", "--input-dir",
        type=str,
        required=True,
        help="Path to folder containing images to caption (e.g. crawler set folder or root downloads).",
    )
    parser.add_argument(
        "-t", "--trigger",
        type=str,
        default=None,
        help="Trigger token to prepend (e.g. 'restrained_elegance', 'shibori').",
    )
    parser.add_argument(
        "--tier",
        type=str,
        choices=["dense", "mid", "short"],
        default="dense",
        help="Which caption tier to save into .txt: 'dense' (7-layer narrative, default), 'mid', or 'short'.",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=os.environ.get("VISION_MODEL", "gemma4:32b"),
        help="VLM model identifier (default: 'gemma4:32b' or env $VISION_MODEL).",
    )
    parser.add_argument(
        "-u", "--url",
        type=str,
        default=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        help="Inference base URL (default: 'http://localhost:11434' or env $OLLAMA_URL).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["ollama", "openai"],
        default="ollama",
        help="API backend protocol: 'ollama' (default) or 'openai' (for vLLM / SGLang on RunPod).",
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=1,
        help="Number of concurrent worker threads (default: 1 for local Ollama, 2-8 for vLLM).",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=1024,
        help="Maximum image edge size in pixels before downscaling (default: 1024).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="HTTP request timeout in seconds (default: 180).",
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Recursively scan subdirectories for images (ideal for downloads/ containing multiple sets).",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Also write full JSON metadata (.json) alongside the .txt file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .txt files instead of skipping them.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose debug logging.",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    input_path = Path(args.input_dir)
    if not input_path.exists() or not input_path.is_dir():
        logger.error(f"Input directory does not exist or is not a directory: {input_path}")
        sys.exit(1)

    caption_directory(input_path, args)


if __name__ == "__main__":
    main()

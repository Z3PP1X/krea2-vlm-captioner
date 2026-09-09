"""Stufe 4 – Qwen-VL Offline-Batch Captioning und Verbindliches Screening."""

from __future__ import annotations

import os
import re
import random
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image

from pipeline.manifest import Manifest, ManifestEntry
from pipeline.captioning.schema import (
    load_vocabulary,
    get_vllm_json_schema,
    build_system_prompt,
)
from pipeline.captioning.screening import evaluate_screening
from pipeline.captioning.assembly import assemble_caption
from pipeline.captioning.reporter import generate_html_sample_report
from pipeline.captioning.vllm_engine import QwenVLEngine

logger = logging.getLogger("pipeline.stage4_caption")


def check_evasion_or_invalid(
    data: Dict[str, Any],
    min_desc_chars: int = 25,
    evasion_patterns: Optional[List[str]] = None,
) -> Tuple[bool, str]:
    """Validates that model output is not empty, evasive, or missing required fields."""
    if not isinstance(data, dict):
        return True, "Output is not a valid JSON dictionary"

    # Check required fields
    required = ["style", "location", "pose", "description"]
    missing = [f for f in required if f not in data]
    if missing:
        return True, f"Missing required JSON fields: {missing}"

    desc = str(data.get("description", "")).strip()
    if len(desc) < min_desc_chars:
        return True, f"Description too short ({len(desc)} < {min_desc_chars} chars)"

    if evasion_patterns:
        for pat in evasion_patterns:
            if re.search(pat, desc):
                return True, f"Matched evasion pattern: {pat}"

    return False, "ok"


def run_stage4(args: Any, config: Dict[str, Any]) -> int:
    """Executes Stage 4: Qwen-VL Offline-Batch Captioning & Screening."""
    general_cfg = config.get("general", {})
    cap_cfg = config.get("stage4_caption", {})
    screening_cfg = cap_cfg.get("screening", {})
    retry_cfg = cap_cfg.get("retry", {})

    manifest_path = general_cfg.get("manifest_path", "data/manifest.jsonl")
    manifest = Manifest(manifest_path)

    vocab_path = general_cfg.get("vocabulary_path", "config/vocabulary.yaml")
    vocab = load_vocabulary(vocab_path)
    json_schema = get_vllm_json_schema(vocab)
    system_prompt = build_system_prompt(vocab)

    batch_size = int(cap_cfg.get("batch_size", 16))
    trigger_word = cap_cfg.get("trigger_word", "restrained_elegance")
    caption_mode = cap_cfg.get("caption_mode", "style")
    template = cap_cfg.get("caption_template", "{trigger} {style} {location} {description} {lighting_camera}")

    min_age = int(screening_cfg.get("min_subject_age", 25))
    reject_uncertain_age = screening_cfg.get("reject_uncertain_age", True)
    reject_watermark = screening_cfg.get("reject_watermark", True)
    reject_text = screening_cfg.get("reject_text", True)
    reject_low_quality = screening_cfg.get("reject_low_quality", True)

    max_retries = int(retry_cfg.get("max_retries", 2))
    temp_step = float(retry_cfg.get("temp_step", 0.1))
    min_desc_chars = int(retry_cfg.get("min_description_chars", 25))
    evasion_patterns = retry_cfg.get("evasion_patterns", [])

    sample_size = getattr(args, "sample", None)

    logger.info("=" * 60)
    logger.info("  STUFE 4: QWEN-VL CAPTIONING & VERBINDLICHES SCREENING")
    logger.info("=" * 60)
    logger.info(f"VLM Engine / Model    : {cap_cfg.get('model_name', 'Qwen-VL')}")
    logger.info(f"Trigger Token         : {trigger_word}")
    logger.info(f"Caption Mode          : {caption_mode} (style omitted in text: {caption_mode == 'style'})")
    logger.info(f"Screening Gate        : Enforced (Min Age: {min_age}, Reject Uncertain: {reject_uncertain_age})")
    logger.info(f"Batch Size            : {batch_size}")
    logger.info(f"Sample Mode           : {sample_size if sample_size else 'Full Dataset'}")
    logger.info("=" * 60)

    # 1. Gather eligible items
    eligible = [
        e for e in manifest
        if e.stages_status.get("stage3_downscale") == "done"
        and e.stages_status.get("stage4_caption") in ["pending", None]
    ]

    if sample_size and sample_size < len(eligible):
        logger.info(f"Selecting random sample of {sample_size} from {len(eligible)} eligible images.")
        eligible = random.sample(eligible, sample_size)

    logger.info(f"Total images to caption: {len(eligible)}")
    if not eligible:
        logger.info("No images pending captioning.")
        return 0

    # 2. Initialize VLM Engine
    engine = QwenVLEngine(
        model_name=cap_cfg.get("model_name", "Qwen/Qwen2.5-VL-7B-Instruct"),
        gpu_memory_utilization=float(cap_cfg.get("vllm_gpu_memory_utilization", 0.85)),
        max_model_len=int(cap_cfg.get("max_model_len", 4096)),
        temperature=float(cap_cfg.get("temperature", 0.2)),
        max_tokens=int(cap_cfg.get("max_tokens", 250)),
    )

    captioned_count = 0
    screened_out_count = 0
    failed_count = 0
    sample_records: List[Dict[str, Any]] = []

    # Process in batches
    for batch_start in range(0, len(eligible), batch_size):
        batch_entries = eligible[batch_start : batch_start + batch_size]
        image_paths = []
        user_prompts = []

        for entry in batch_entries:
            p = entry.get_processed_file() or entry.get_raw_file() or Path(entry.processed_path or entry.raw_path or "")
            image_paths.append(p)
            context = f"Context: {entry.context_title}. Tags: {', '.join(entry.context_tags)}" if entry.context_tags else ""
            user_prompts.append(
                f"Inspect this image objectively and output the structured JSON conforming to the schema. {context}"
            )

        # Generate outputs
        try:
            results = engine.generate_batch(
                image_paths=image_paths,
                user_prompts=user_prompts,
                system_prompt=system_prompt,
                json_schema=json_schema,
            )
        except Exception as exc:
            logger.error(f"vLLM Batch Generation Error: {exc}")
            for entry in batch_entries:
                entry.update_stage("stage4_caption", "failed", reasons=[f"vllm_error_{type(exc).__name__}"])
            continue

        # Inspect and evaluate each result
        for entry, img_path, res_data in zip(batch_entries, image_paths, results):
            # Check for evasion or invalid output with retry attempt
            is_evasive, ev_reason = check_evasion_or_invalid(res_data, min_desc_chars, evasion_patterns)
            attempt = 0

            while is_evasive and attempt < max_retries:
                attempt += 1
                logger.info(f"Retrying image {img_path.name} (Attempt {attempt}/{max_retries}) due to: {ev_reason}")
                try:
                    retry_results = engine.generate_batch(
                        image_paths=[img_path],
                        user_prompts=[f"Please provide an objective, factual visual description without refusal. Context: {entry.context_title}"],
                        system_prompt=system_prompt,
                        json_schema=json_schema,
                        seed=42 + attempt * 100,
                        temperature=float(cap_cfg.get("temperature", 0.2)) + (attempt * temp_step),
                    )
                    res_data = retry_results[0]
                    is_evasive, ev_reason = check_evasion_or_invalid(res_data, min_desc_chars, evasion_patterns)
                except Exception as e:
                    logger.warning(f"Retry generation failed: {e}")
                    break

            if is_evasive:
                logger.warning(f"Image {img_path.name} failed captioning after {attempt} retries: {ev_reason}")
                entry.update_stage("stage4_caption", "failed", reasons=[f"evasion_or_invalid_{ev_reason}"])
                failed_count += 1
                continue

            # 3. Mandatory Compliance & Screening Gate
            passed_screen, screen_reasons = evaluate_screening(
                res_data,
                min_age=min_age,
                reject_uncertain_age=reject_uncertain_age,
                reject_watermark=reject_watermark,
                reject_text=reject_text,
                reject_low_quality=reject_low_quality,
            )

            if not passed_screen:
                logger.info(f"[SCREENING REJECT] {img_path.name}: {screen_reasons}")
                entry.caption_data = res_data
                entry.update_stage("stage4_caption", "rejected_screening", reasons=screen_reasons)
                screened_out_count += 1
                if sample_size:
                    sample_records.append({
                        "image_path": str(img_path),
                        "caption": "(Rejected at Screening)",
                        "status": "rejected_screening",
                        "json_data": res_data,
                        "reasons": screen_reasons,
                    })
                continue

            # 4. Caption Assembly
            caption_text = assemble_caption(
                res_data,
                trigger_word=trigger_word,
                caption_mode=caption_mode,
                template=template,
            )

            # Write .txt alongside image
            txt_path = img_path.with_suffix(".txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(caption_text + "\n")

            rel_txt_path = os.path.relpath(txt_path, Path(general_cfg.get("manifest_path", "data")).parent).replace("\\", "/")

            entry.caption_data = res_data
            entry.caption_text = caption_text
            entry.caption_path = rel_txt_path
            entry.update_stage("stage4_caption", "captioned")
            entry.stages_status["stage5_export"] = "pending"
            captioned_count += 1

            if sample_size:
                sample_records.append({
                    "image_path": str(img_path),
                    "caption": caption_text,
                    "status": "captioned",
                    "json_data": res_data,
                    "reasons": [],
                })

            logger.info(f"[CAPTIONED] {img_path.name} -> {caption_text[:70]}...")

        # Periodic manifest save
        manifest.save()

    manifest.save()

    # Generate sample HTML report if requested
    if sample_size and sample_records:
        report_path = Path(general_cfg.get("reports_dir", "data/reports")) / "caption_sample_report.html"
        generated_report = generate_html_sample_report(sample_records, output_path=report_path)
        logger.info(f"Sample inspection HTML report generated: {generated_report.resolve()}")

    logger.info("=" * 60)
    logger.info("  STUFE 4 ERGEBNIS")
    logger.info(f"  Erfolgreich Gecaptioned    : {captioned_count}")
    logger.info(f"  Screening Ausgeschlossen   : {screened_out_count}")
    logger.info(f"  Inferenz-Fehler / Abbrüche : {failed_count}")
    logger.info("=" * 60)

    return 0

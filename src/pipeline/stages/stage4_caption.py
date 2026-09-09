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
    system_prompt = build_system_prompt(vocab, raw_caption_mode=True)

    raw_model = getattr(args, "model", None) or cap_cfg.get("model_name", "Qwen/Qwen2.5-VL-3B-Instruct")
    model_aliases = {
        "3.8": "Qwen/Qwen2.5-VL-3B-Instruct",
        "3.8b": "Qwen/Qwen2.5-VL-3B-Instruct",
        "qwen3.8": "Qwen/Qwen2.5-VL-3B-Instruct",
        "qwen3.8b": "Qwen/Qwen2.5-VL-3B-Instruct",
        "qwen-3.8": "Qwen/Qwen2.5-VL-3B-Instruct",
        "qwen-3.8b": "Qwen/Qwen2.5-VL-3B-Instruct",
        "qwen-vl-3.8": "Qwen/Qwen2.5-VL-3B-Instruct",
        "qwen-vl-3.8b": "Qwen/Qwen2.5-VL-3B-Instruct",
        "krea2": "Qwen/Qwen2.5-VL-3B-Instruct",
        "7b": "Qwen/Qwen2.5-VL-7B-Instruct",
        "qwen7b": "Qwen/Qwen2.5-VL-7B-Instruct",
        "qwen-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
        "27b": "Qwen/Qwen3.8-27B",
        "qwen27b": "Qwen/Qwen3.8-27B",
        # Gemma 4 & PaliGemma Multimodal Family
        "gemma": "google/gemma-4-E4B-it",
        "gemma4": "google/gemma-4-E4B-it",
        "gemma-4": "google/gemma-4-E4B-it",
        "gemma4-e4b": "google/gemma-4-E4B-it",
        "gemma-4-e4b": "google/gemma-4-E4B-it",
        "gemma4-e2b": "google/gemma-4-E2B-it",
        "gemma-4-e2b": "google/gemma-4-E2B-it",
        "gemma4-12b": "google/gemma-4-12B-it",
        "gemma-4-12b": "google/gemma-4-12B-it",
        "gemma4-26b": "google/gemma-4-26B-it",
        "gemma-4-26b": "google/gemma-4-26B-it",
        "gemma4-31b": "google/gemma-4-31B-it",
        "gemma-4-31b": "google/gemma-4-31B-it",
        "paligemma": "google/paligemma2-3b-pt-448",
        "paligemma2": "google/paligemma2-3b-pt-448",
    }
    model_name = model_aliases.get(raw_model.lower().strip(), raw_model)
    batch_size = getattr(args, "batch_size", None) or int(cap_cfg.get("batch_size", 16))
    trigger_word = getattr(args, "trigger", None) or cap_cfg.get("trigger_word", "restrained_elegance")
    caption_mode = getattr(args, "mode", None) or cap_cfg.get("caption_mode", "style")
    template = cap_cfg.get("caption_template", "{trigger} {style} {location} {description} {lighting_camera}")

    check_age = bool(screening_cfg.get("check_age", False))
    min_age = int(screening_cfg.get("min_subject_age", 0)) if check_age else 0
    reject_uncertain_age = bool(screening_cfg.get("reject_uncertain_age", False)) if check_age else False

    no_screening = bool(getattr(args, "no_screening", False))
    ignore_watermarks = bool(getattr(args, "ignore_watermarks", False))

    reject_watermark = False if (no_screening or ignore_watermarks) else bool(screening_cfg.get("reject_watermark", False))
    reject_text = False if (no_screening or ignore_watermarks) else bool(screening_cfg.get("reject_text", False))
    reject_low_quality = False if no_screening else bool(screening_cfg.get("reject_low_quality", True))

    max_retries = int(retry_cfg.get("max_retries", 2))
    temp_step = float(retry_cfg.get("temp_step", 0.1))
    min_desc_chars = int(retry_cfg.get("min_description_chars", 25))
    evasion_patterns = retry_cfg.get("evasion_patterns", [])

    sample_size = getattr(args, "sample", None)

    logger.info("=" * 60)
    logger.info("  STUFE 4: QWEN-VL CAPTIONING & SCREENING")
    logger.info("=" * 60)
    logger.info(f"VLM Engine / Model    : {model_name}")
    logger.info(f"Trigger Token         : {trigger_word}")
    logger.info(f"Caption Mode          : {caption_mode} (style omitted in text: {caption_mode == 'style'})")
    gate_status = "Disabled (--no-screening)" if no_screening else f"Active (Watermarks: {'Reject' if reject_watermark else 'Allow'}, Text: {'Reject' if reject_text else 'Allow'})"
    logger.info(f"Screening Gate        : {gate_status}")
    logger.info(f"Batch Size            : {batch_size}")
    logger.info(f"Sample Mode           : {sample_size if sample_size else 'Full Dataset'}")
    logger.info("=" * 60)

    force = bool(getattr(args, "force", False))
    retry_failed = bool(getattr(args, "retry_failed", True))
    refine_mode = bool(getattr(args, "refine_captions", False) or getattr(args, "refine", False))

    if refine_mode:
        logger.info("Refinement Mode Active: Auditing and expanding existing captions into 400+ token Krea 2 narratives.")

    # 1. Gather eligible items
    # Check if stage 3 downscaling was run
    downscale_done_count = sum(1 for e in manifest if e.stages_status.get("stage3_downscale") == "done")
    recheck_screening = (not reject_watermark or not reject_text or no_screening)

    if refine_mode:
        # In refinement mode, select all entries that have passed QC or are already captioned
        eligible = [
            e for e in manifest
            if e.stages_status.get("stage4_caption") == "captioned"
            or e.stages_status.get("stage3_downscale") == "done"
            or e.stages_status.get("stage2_qc") == "passed"
        ]
    elif downscale_done_count > 0:
        eligible = [
            e for e in manifest
            if e.stages_status.get("stage3_downscale") == "done"
            and (
                force
                or e.stages_status.get("stage4_caption") in ["pending", None]
                or (retry_failed and e.stages_status.get("stage4_caption") == "failed")
                or (
                    recheck_screening
                    and e.stages_status.get("stage4_caption") == "rejected_screening"
                    and any("watermark" in r or "text" in r for r in e.rejection_reasons)
                )
            )
        ]
    else:
        # Fallback: if Stage 3 downscaling was not run yet, accept QC-passed images directly
        qc_passed_entries = [
            e for e in manifest
            if e.stages_status.get("stage2_qc") == "passed"
            and (
                force
                or e.stages_status.get("stage4_caption") in ["pending", None]
                or (retry_failed and e.stages_status.get("stage4_caption") == "failed")
                or (
                    recheck_screening
                    and e.stages_status.get("stage4_caption") == "rejected_screening"
                    and any("watermark" in r or "text" in r for r in e.rejection_reasons)
                )
            )
        ]
        if qc_passed_entries:
            logger.info("Notice: Stage 3 Downscale has not been run yet. Proceeding with raw QC-passed images.")
            eligible = qc_passed_entries
        else:
            eligible = []

    if sample_size and sample_size < len(eligible):
        logger.info(f"Selecting random sample of {sample_size} from {len(eligible)} eligible images.")
        eligible = random.sample(eligible, sample_size)

    logger.info(f"Total images to caption: {len(eligible)}")
    if not eligible:
        total_m = len(manifest)
        qc_passed = sum(1 for e in manifest if e.stages_status.get("stage2_qc") == "passed")
        downscale_done = sum(1 for e in manifest if e.stages_status.get("stage3_downscale") == "done")
        captioned = sum(1 for e in manifest if e.stages_status.get("stage4_caption") == "captioned")
        screened_out = sum(1 for e in manifest if e.stages_status.get("stage4_caption") == "rejected_screening")
        failed_caption = sum(1 for e in manifest if e.stages_status.get("stage4_caption") == "failed")

        logger.info("No images pending captioning. Manifest status diagnostics:")
        logger.info(f"  - Total entries in manifest    : {total_m}")
        logger.info(f"  - Stage 2 QC Passed            : {qc_passed}")
        logger.info(f"  - Stage 3 Downscaled           : {downscale_done}")
        logger.info(f"  - Stage 4 Already Captioned    : {captioned}")
        logger.info(f"  - Stage 4 Rejected Screening   : {screened_out}")
        logger.info(f"  - Stage 4 Failed Previous Runs : {failed_caption}")
        if captioned > 0:
            logger.info("Tip: All eligible images are already captioned. Use 'pipeline caption --force' to re-caption.")
        elif qc_passed == 0:
            logger.info("Tip: No images have passed Stage 2 QC yet. Run 'pipeline qc' first.")
        return 0

    # 2. Initialize VLM Engine
    max_model_len = getattr(args, "max_model_len", None) or int(cap_cfg.get("max_model_len", 12288))
    temperature = float(getattr(args, "temperature", None) or cap_cfg.get("temperature", 0.75))
    max_tokens = int(getattr(args, "max_tokens", None) or cap_cfg.get("max_tokens", 750))
    m_lower = model_name.lower()

    if "gemma-4" in m_lower or "gemma4" in m_lower:
        from pipeline.captioning.hf_gemma4_engine import Gemma4HfEngine
        parallel_sub_batch = getattr(args, "parallel_sub_batch", None) or min(batch_size, 16)
        engine = Gemma4HfEngine(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            parallel_sub_batch_size=int(parallel_sub_batch),
        )
    else:
        engine = QwenVLEngine(
            model_name=model_name,
            gpu_memory_utilization=float(cap_cfg.get("vllm_gpu_memory_utilization", 0.85)),
            max_model_len=max_model_len,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    try:
        engine._ensure_model_loaded()
    except RuntimeError as exc:
        logger.error(f"Cannot initialize Stage 4 engine: {exc}")
        return 1

    captioned_count = 0
    screened_out_count = 0
    failed_count = 0
    sample_records: List[Dict[str, Any]] = []

    # Process in batches
    total_eligible = len(eligible)
    total_batches = (total_eligible + batch_size - 1) // batch_size
    logger.info(f"\nStarting Stage 4 processing: {total_eligible} images across {total_batches} batches (batch size: {batch_size}, target: >= 400 tokens, temp: {temperature})...")

    for batch_idx, batch_start in enumerate(range(0, total_eligible, batch_size), start=1):
        batch_entries = eligible[batch_start : batch_start + batch_size]
        batch_end = min(batch_start + batch_size, total_eligible)
        logger.info(f"\n--- Batch {batch_idx}/{total_batches} [Images {batch_start + 1}-{batch_end} of {total_eligible}] ---")
        image_paths = []
        user_prompts = []

        for entry in batch_entries:
            p = entry.get_processed_file() or entry.get_raw_file() or Path(entry.processed_path or entry.raw_path or "")
            image_paths.append(p)
            context = f"Context: {entry.context_title}. Tags: {', '.join(entry.context_tags)}" if entry.context_tags else ""

            # Check for existing caption for refinement
            existing_caption = ""
            txt_p = p.with_suffix(".txt")
            if txt_p.exists():
                try:
                    with open(txt_p, "r", encoding="utf-8") as f_old:
                        existing_caption = f_old.read().strip()
                except Exception:
                    pass
            if not existing_caption and entry.caption_text:
                existing_caption = entry.caption_text.strip()

            if refine_mode and existing_caption:
                user_prompts.append(
                    f"Existing Draft Caption: \"{existing_caption}\"\n"
                    f"Refinement Task: Audit and elevate this draft caption into an exhaustive Krea 2 visual narrative. "
                    f"Correct any visual errors or inaccurate terms. The output caption MUST be at least 550 tokens long and up to 1024 tokens "
                    f"(approx. 380-500+ words in detailed natural English), exhaustively detailing: "
                    f"(1) Model Position & posture geometry (kneeling, arched spine, limb angles, physical contact points, muscle tension), "
                    f"(2) Bondage Type (shibari rope, chain restraint, leather bondage, suspension, predicament), "
                    f"(3) Bondage Equipment & materials (5-8mm hemp/jute rope, welded steel chains, chrome handcuffs, O-rings, carabiners, spreader bars, leather collars), "
                    f"(4) Bondage Position & anatomical rigging (takate-kote box tie, hishime chest harness, wrist/ankle placement, skin bite indentations), "
                    f"(5) Studio environment/flooring, (6) Chiaroscuro lighting/specular highlights, and (7) Camera optics/DoF. "
                    f"Output raw caption text only. Do not output JSON. {context}"
                )
            else:
                user_prompts.append(
                    f"Task: Inspect this image and write an exhaustive Krea 2 visual narrative. "
                    f"The output caption MUST be at least 550 tokens long and up to 1024 tokens (approx. 380-500+ words in detailed natural English), "
                    f"exhaustively detailing: "
                    f"(1) Model Position & posture geometry (kneeling, arched spine, limb angles, physical contact points, muscle tension), "
                    f"(2) Bondage Type (shibari rope, chain restraint, leather bondage, suspension, predicament), "
                    f"(3) Bondage Equipment & materials (5-8mm hemp/jute rope, welded steel chains, chrome handcuffs, O-rings, carabiners, spreader bars, leather collars), "
                    f"(4) Bondage Position & anatomical rigging (takate-kote box tie, hishime chest harness, wrist/ankle placement, skin bite indentations), "
                    f"(5) Studio environment/flooring, (6) Chiaroscuro lighting/specular highlights, and (7) Camera optics/DoF. "
                    f"Output raw caption text only. Do not output JSON. {context}"
                )

        # Generate outputs
        try:
            results = engine.generate_batch(
                image_paths=image_paths,
                user_prompts=user_prompts,
                system_prompt=system_prompt,
                json_schema=json_schema,
                raw_text_mode=True,
            )
        except Exception as exc:
            logger.warning(f"Batch generation exception: {exc}. Retrying batch items individually...")
            results = []
            for img_p, u_p, b_entry in zip(image_paths, user_prompts, batch_entries):
                try:
                    single_res = engine.generate_batch(
                        image_paths=[img_p],
                        user_prompts=[u_p],
                        system_prompt=system_prompt,
                        json_schema=json_schema,
                        raw_text_mode=True,
                    )
                    results.append(single_res[0])
                except Exception as single_exc:
                    logger.error(f"Single image {img_p.name} failed: {single_exc}")
                    b_entry.update_stage("stage4_caption", "failed", reasons=[f"vllm_error_{type(single_exc).__name__}"])
                    results.append(None)

        # Inspect and evaluate each result
        for entry, img_path, res_data in zip(batch_entries, image_paths, results):
            if res_data is None:
                failed_count += 1
                continue
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

            passed_screen, screen_reasons = evaluate_screening(
                res_data,
                min_age=min_age,
                check_age=check_age,
                reject_uncertain_age=reject_uncertain_age,
                reject_watermark=reject_watermark,
                reject_text=reject_text,
                reject_low_quality=reject_low_quality,
            )

            if not passed_screen:
                logger.info(f"[SCREENING REJECT] {img_path.name}: {screen_reasons}")
                entry.rejection_reasons = [r for r in entry.rejection_reasons if not r.startswith("vllm_error")]
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
            item_trigger = entry.trigger_word
            if not item_trigger:
                if trigger_word and "{category}" in trigger_word and entry.category:
                    item_trigger = trigger_word.replace("{category}", entry.category)
                elif trigger_word:
                    item_trigger = trigger_word
                elif entry.category:
                    item_trigger = f"kink, {entry.category}"
                else:
                    item_trigger = "restrained_elegance"

            caption_text = assemble_caption(
                res_data,
                trigger_word=item_trigger,
                caption_mode=caption_mode,
                template=template,
            )

            # Write .txt alongside image (with backup if refining)
            txt_path = img_path.with_suffix(".txt")
            if refine_mode and txt_path.exists():
                bak_path = img_path.with_suffix(".txt.bak")
                if not bak_path.exists():
                    try:
                        import shutil
                        shutil.copy2(txt_path, bak_path)
                    except Exception:
                        pass

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(caption_text + "\n")

            rel_txt_path = os.path.relpath(txt_path, Path(general_cfg.get("manifest_path", "data")).parent).replace("\\", "/")

            entry.rejection_reasons = [
                r for r in entry.rejection_reasons
                if not (r.startswith("vllm_error") or r.startswith("screening_"))
            ]
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

            tag_label = "[REFINED]" if refine_mode else "[CAPTIONED]"
            w_count = len(caption_text.split())
            approx_tok = int(w_count * 1.35)
            logger.info(f"{tag_label} {img_path.name} ({w_count} words, ~{approx_tok} tokens) -> {caption_text[:75]}...")

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

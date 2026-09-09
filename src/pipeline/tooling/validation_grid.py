"""Stufe 7 – Validierungs-Tooling: Prompt-Grid Generator und Kontaktbogen-Builder."""

from __future__ import annotations

import html
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

from pipeline.captioning.schema import load_vocabulary

logger = logging.getLogger("pipeline.validation_grid")


def generate_prompt_grid(
    vocab: Dict[str, List[str]],
    trigger_word: str = "restrained_elegance",
) -> List[Dict[str, Any]]:
    """Builds an exhaustive testing grid of standalone subconcepts and systematic combinations."""
    grid = []
    grid_idx = 1

    styles = [s for s in vocab.get("styles", []) if s != "other"]
    locations = [loc for loc in vocab.get("locations", []) if loc != "other"]
    poses = [p for p in vocab.get("poses", []) if p != "other"]

    # 1. Standalone Styles
    for s in styles:
        s_clean = s.replace("_", " ")
        grid.append({
            "id": f"style_{s}",
            "type": "standalone_style",
            "prompt": f"{trigger_word} A photograph showcasing {s_clean} aesthetic in studio lighting.",
            "style": s,
        })

    # 2. Standalone Poses
    for p in poses:
        p_clean = p.replace("_", " ")
        grid.append({
            "id": f"pose_{p}",
            "type": "standalone_pose",
            "prompt": f"{trigger_word} Photograph of a slender subject in {p_clean}, neutral studio backdrop.",
            "pose": p,
        })

    # 3. Systematic Combinations (Style x Location x Pose)
    for s in styles[:4]:
        for loc in locations[:3]:
            for p in poses[:3]:
                s_clean = s.replace("_", " ")
                loc_clean = loc.replace("_", " ")
                p_clean = p.replace("_", " ")
                grid.append({
                    "id": f"combo_{s}_{loc}_{p}",
                    "type": "triplet_combination",
                    "prompt": f"{trigger_word} {s_clean} aesthetic photograph set in a {loc_clean}. The subject is posed in {p_clean}. Directional rim lighting.",
                    "style": s,
                    "location": loc,
                    "pose": p,
                })

    return grid


def generate_contact_sheet_html(
    grid_items: List[Dict[str, Any]],
    output_path: str | Path = "data/validation/contact_sheet.html",
    weights: List[float] = [0.0, 0.7, 0.85, 1.0],
    models: List[str] = ["Krea 2 RAW (52 Steps)", "Krea 2 Turbo (8 Steps)"],
) -> Path:
    """Generates an HTML contact sheet matrix comparing prompts across LoRA scale weights."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    rows_html = []
    for item in grid_items[:20]:  # preview sample rows
        prompt_text = item["prompt"]
        cols_html = []
        for w in weights:
            # Placeholder thumbnail cell representing rendered checkpoint image
            cols_html.append(f"""
            <td class="render-cell">
                <div class="render-box">
                    <span class="weight-label">LoRA Scale {w:.2f}</span>
                    <div class="placeholder-img">Image @ {w:.2f}</div>
                </div>
            </td>
            """)

        rows_html.append(f"""
        <tr>
            <td class="prompt-cell">
                <strong>{html.escape(item['id'])}</strong><br>
                <small>{html.escape(prompt_text)}</small>
            </td>
            {''.join(cols_html)}
        </tr>
        """)

    table_body = "\n".join(rows_html)
    header_cols = "".join([f"<th>LoRA Scale {w:.2f}</th>" for w in weights])

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Krea 2 LoRA Checkpoint Validation Contact Sheet</title>
    <style>
        body {{ font-family: -apple-system, sans-serif; background: #0f172a; color: #f8fafc; padding: 24px; }}
        h1 {{ color: #38bdf8; margin-bottom: 4px; }}
        p {{ color: #94a3b8; margin-bottom: 24px; }}
        table {{ width: 100%; border-collapse: collapse; background: #1e293b; border-radius: 8px; overflow: hidden; }}
        th, td {{ border: 1px solid #334155; padding: 12px; text-align: left; vertical-align: top; }}
        th {{ background: #020617; color: #38bdf8; font-weight: 600; text-align: center; }}
        .prompt-cell {{ width: 28%; font-size: 0.85rem; color: #cbd5e1; }}
        .render-cell {{ text-align: center; width: 18%; }}
        .render-box {{ background: #020617; border-radius: 6px; padding: 8px; }}
        .placeholder-img {{ height: 160px; background: #1e293b; display: flex; align-items: center; justify-content: center; border-radius: 4px; color: #64748b; font-size: 0.8rem; }}
        .weight-label {{ font-size: 0.75rem; color: #38bdf8; display: block; margin-bottom: 4px; font-weight: 600; }}
    </style>
</head>
<body>
    <h1>Krea 2 LoRA Checkpoint Validation Contact Sheet</h1>
    <p>Comparing Prompt Responses across LoRA Weights (0.0 to 1.0) on Krea 2 RAW & Turbo.</p>
    <table>
        <thead>
            <tr>
                <th>Test Prompt</th>
                {header_cols}
            </tr>
        </thead>
        <tbody>
            {table_body}
        </tbody>
    </table>
</body>
</html>
"""
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    return out_file


def run_validation_grid(args: Any, config: Dict[str, Any]) -> int:
    """Executes Stage 7: Generates validation prompt grid and contact sheet template."""
    general_cfg = config.get("general", {})
    cap_cfg = config.get("stage4_caption", {})

    vocab_path = general_cfg.get("vocabulary_path", "config/vocabulary.yaml")
    vocab = load_vocabulary(vocab_path)
    trigger = cap_cfg.get("trigger_word", "restrained_elegance")

    val_dir = Path("data/validation")
    val_dir.mkdir(parents=True, exist_ok=True)

    grid = generate_prompt_grid(vocab, trigger_word=trigger)
    grid_file = val_dir / "prompt_grid.json"
    with open(grid_file, "w", encoding="utf-8") as f:
        json.dump(grid, f, indent=2, ensure_ascii=False)

    sheet_file = val_dir / "contact_sheet.html"
    generate_contact_sheet_html(grid, output_path=sheet_file)

    logger.info("=" * 60)
    logger.info("  STUFE 7 ERGEBNIS: VALIDIERUNGS-GRID GENERIERT")
    logger.info(f"  Prompts im Grid        : {len(grid)}")
    logger.info(f"  JSON Prompt Grid       : {grid_file.resolve()}")
    logger.info(f"  Kontaktbogen (HTML)    : {sheet_file.resolve()}")
    logger.info("=" * 60)

    return 0

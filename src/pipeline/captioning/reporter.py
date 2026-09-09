"""HTML and Markdown report generation for captioning sample inspection."""

from __future__ import annotations

import os
import html
import json
from pathlib import Path
from typing import List, Dict, Any


def generate_html_sample_report(
    sample_records: List[Dict[str, Any]],
    output_path: str | Path = "data/reports/caption_sample_report.html",
    title: str = "Krea 2 LoRA Dataset — Captioning & Screening Sample Review",
) -> Path:
    """Generates a self-contained, responsive HTML inspection page with image cards and JSON breakdown."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    cards_html = []
    for item in sample_records:
        img_path = item.get("image_path", "")
        caption = item.get("caption", "")
        status = item.get("status", "unknown")
        json_data = item.get("json_data", {})
        reasons = item.get("reasons", [])

        status_badge = "badge-pass" if status == "captioned" else "badge-reject"
        badge_text = "PASSED" if status == "captioned" else f"REJECTED: {', '.join(reasons)}"

        # Use relative path from report to image
        try:
            rel_img = os.path.relpath(img_path, out_file.parent).replace("\\", "/")
        except Exception:
            rel_img = img_path

        card = f"""
        <div class="card">
            <div class="img-box">
                <img src="{html.escape(rel_img)}" alt="Sample Image" loading="lazy">
            </div>
            <div class="content">
                <span class="badge {status_badge}">{html.escape(badge_text)}</span>
                <h3>{html.escape(Path(img_path).name)}</h3>
                <p class="caption"><strong>Caption:</strong> {html.escape(caption)}</p>
                <details>
                    <summary>View Structured JSON Breakdown</summary>
                    <pre><code>{html.escape(json.dumps(json_data, indent=2, ensure_ascii=False))}</code></pre>
                </details>
            </div>
        </div>
        """
        cards_html.append(card)

    cards_combined = "\n".join(cards_html)

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: #0f172a;
            color: #f8fafc;
            margin: 0;
            padding: 24px;
        }}
        h1 {{ margin-bottom: 8px; font-weight: 700; color: #38bdf8; }}
        p.subtitle {{ color: #94a3b8; margin-bottom: 24px; }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
            gap: 20px;
        }}
        .card {{
            background: #1e293b;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
            border: 1px solid #334155;
            display: flex;
            flex-direction: column;
        }}
        .img-box {{
            height: 240px;
            background: #020617;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }}
        .img-box img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
        }}
        .content {{
            padding: 16px;
            flex: 1;
            display: flex;
            flex-direction: column;
        }}
        .badge {{
            display: inline-block;
            align-self: flex-start;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            margin-bottom: 8px;
        }}
        .badge-pass {{ background: #065f46; color: #34d399; }}
        .badge-reject {{ background: #991b1b; color: #f87171; }}
        h3 {{ margin: 0 0 8px; font-size: 1rem; color: #e2e8f0; }}
        p.caption {{
            font-size: 0.875rem;
            line-height: 1.4;
            color: #cbd5e1;
            background: #0f172a;
            padding: 8px 12px;
            border-radius: 6px;
            border-left: 3px solid #38bdf8;
            margin-bottom: 12px;
        }}
        details {{
            margin-top: auto;
            font-size: 0.75rem;
            color: #94a3b8;
        }}
        summary {{
            cursor: pointer;
            padding: 4px 0;
            user-select: none;
        }}
        pre {{
            background: #020617;
            padding: 8px;
            border-radius: 6px;
            overflow-x: auto;
            color: #a5f3fc;
            margin: 6px 0 0;
        }}
    </style>
</head>
<body>
    <h1>{html.escape(title)}</h1>
    <p class="subtitle">Sample Size: {len(sample_records)} images inspected with Qwen-VL 7-Layer Narrative Schema and Mandatory Age Gate.</p>
    <div class="grid">
        {cards_combined}
    </div>
</body>
</html>
"""
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html_doc)

    return out_file

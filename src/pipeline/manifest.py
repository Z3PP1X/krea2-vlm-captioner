"""Central Manifest Manager for Krea 2 LoRA Data Pipeline.

Provides atomic, idempotent JSON-Lines storage tracking image state across all stages.
"""

from __future__ import annotations

import os
import json
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator
from pydantic import BaseModel, Field


class ManifestEntry(BaseModel):
    """Represents a single image item and its full lifecycle status in the pipeline."""

    image_id: str = Field(description="Unique hash or identifier for the image")
    source: str = Field(default="unknown", description="Source platform (e.g. xenforo, dbnaked, manual)")
    source_url: str = Field(description="Remote URL of the image file")
    page_url: Optional[str] = Field(default=None, description="Remote gallery/thread page URL")
    context_title: Optional[str] = Field(default=None, description="Set/Scene title")
    context_tags: List[str] = Field(default_factory=list, description="Original tags or genre metadata")

    # File locations
    raw_path: Optional[str] = Field(default=None, description="Relative path to raw downloaded file")
    processed_path: Optional[str] = Field(default=None, description="Relative path to downscaled/cleaned image")
    caption_path: Optional[str] = Field(default=None, description="Relative path to paired .txt caption file")

    # Image Metrics
    sha256: Optional[str] = Field(default=None, description="SHA256 checksum of raw file")
    phash: Optional[str] = Field(default=None, description="Perceptual hash string (pHash)")
    phash_cluster_id: Optional[str] = Field(default=None, description="ID of deduplication cluster")
    width: Optional[int] = Field(default=None, description="Image width in pixels")
    height: Optional[int] = Field(default=None, description="Image height in pixels")
    aspect_ratio: Optional[float] = Field(default=None, description="Aspect ratio (width / height)")
    megapixels: Optional[float] = Field(default=None, description="Resolution in megapixels")

    # Lifecycle State per Stage
    # Allowed states: pending, downloaded, passed, rejected, rejected_screening, done, failed, skipped
    stages_status: Dict[str, str] = Field(
        default_factory=lambda: {
            "stage1_crawl": "pending",
            "stage2_qc": "pending",
            "stage3_downscale": "pending",
            "stage4_caption": "pending",
            "stage5_export": "pending",
        }
    )

    # QC and Screening Metrics
    qc_metrics: Dict[str, Any] = Field(default_factory=dict, description="Laplacian variance, scores, flags")
    rejection_reasons: List[str] = Field(default_factory=list, description="Non-destructive rejection notes")

    # VLM Annotation & Output
    caption_data: Optional[Dict[str, Any]] = Field(default=None, description="Structured JSON output from Qwen-VL")
    caption_text: Optional[str] = Field(default=None, description="Final assembled text prompt")

    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def update_stage(self, stage: str, status: str, reasons: Optional[List[str]] = None) -> None:
        """Updates the status of a specific stage and registers timestamp."""
        self.stages_status[stage] = status
        if reasons:
            for r in reasons:
                if r not in self.rejection_reasons:
                    self.rejection_reasons.append(r)
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def is_rejected(self) -> bool:
        """Returns True if item was rejected at QC or screening."""
        return (
            self.stages_status.get("stage2_qc") == "rejected"
            or self.stages_status.get("stage4_caption") == "rejected_screening"
            or len(self.rejection_reasons) > 0
        )

    def get_raw_file(self) -> Optional[Path]:
        """Resolves existing raw file path robustly."""
        return resolve_manifest_path(self.raw_path)

    def get_processed_file(self) -> Optional[Path]:
        """Resolves existing processed file path robustly."""
        return resolve_manifest_path(self.processed_path)


def resolve_manifest_path(path_str: Optional[str | Path]) -> Optional[Path]:
    """Resolves a manifest file path robustly against cwd and data directory."""
    if not path_str:
        return None
    p = Path(path_str)
    if p.exists():
        return p
    # Try with 'data/' prefix
    data_p = Path("data") / p
    if data_p.exists():
        return data_p
    # Try if path starts with 'data/' but current dir already is 'data/'
    parts = p.parts
    if parts and parts[0] == "data":
        sub_p = Path(*parts[1:])
        if sub_p.exists():
            return sub_p
    return None


class Manifest:
    """Manages reading, querying, and atomically saving manifest entries."""

    def __init__(self, manifest_path: str | Path):
        self.path = Path(manifest_path)
        self.entries: Dict[str, ManifestEntry] = {}
        self._url_to_id: Dict[str, str] = {}
        self._lock = threading.RLock()
        self.load()

    def load(self) -> None:
        """Loads entries from the JSONL manifest file if it exists."""
        with self._lock:
            self.entries.clear()
            self._url_to_id.clear()
            if not self.path.exists():
                return

            with open(self.path, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        entry = ManifestEntry(**data)
                        self.entries[entry.image_id] = entry
                        if entry.source_url:
                            self._url_to_id[entry.source_url] = entry.image_id
                    except Exception as exc:
                        # Non-fatal error; logs and skips malformed line
                        print(f"Warning: Corrupt line {line_idx} in {self.path}: {exc}")

    def save(self) -> None:
        """Atomically persists all entries to the JSONL manifest file."""
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to temporary file in the same directory, then rename atomically
            temp_file = tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                delete=False,
                suffix=".tmp"
            )
            try:
                for entry in self.entries.values():
                    temp_file.write(entry.model_dump_json() + "\n")
                temp_file.flush()
                os.fsync(temp_file.fileno())
                temp_file.close()

                # Atomic replace
                os.replace(temp_file.name, self.path)
            except Exception:
                if os.path.exists(temp_file.name):
                    os.remove(temp_file.name)
                raise

    def add_or_update(self, entry: ManifestEntry) -> None:
        """Inserts or updates an entry and records URL mapping."""
        with self._lock:
            entry.updated_at = datetime.now(timezone.utc).isoformat()
            self.entries[entry.image_id] = entry
            if entry.source_url:
                self._url_to_id[entry.source_url] = entry.image_id

    def get(self, image_id: str) -> Optional[ManifestEntry]:
        """Gets entry by its unique image_id."""
        with self._lock:
            return self.entries.get(image_id)

    def get_by_url(self, url: str) -> Optional[ManifestEntry]:
        """Gets entry by its source URL."""
        with self._lock:
            image_id = self._url_to_id.get(url)
            return self.entries.get(image_id) if image_id else None

    def __iter__(self) -> Iterator[ManifestEntry]:
        with self._lock:
            return iter(list(self.entries.values()))

    def __len__(self) -> int:
        with self._lock:
            return len(self.entries)

    def filter_by_stage(self, stage: str, status: str) -> List[ManifestEntry]:
        """Returns all entries matching a given status in a stage."""
        return [
            e for e in self.entries.values()
            if e.stages_status.get(stage) == status
        ]

    def summary(self) -> Dict[str, Any]:
        """Calculates summary statistics across all stages and rejections."""
        total = len(self.entries)
        stage_counts: Dict[str, Dict[str, int]] = {}
        for entry in self.entries.values():
            for stage, status in entry.stages_status.items():
                if stage not in stage_counts:
                    stage_counts[stage] = {}
                stage_counts[stage][status] = stage_counts[stage].get(status, 0) + 1

        rejection_counts: Dict[str, int] = {}
        for entry in self.entries.values():
            for reason in entry.rejection_reasons:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

        return {
            "total_items": total,
            "stages": stage_counts,
            "rejections": rejection_counts,
        }

"""Automated integration of exported datasets and training configs into ostris/ai-toolkit."""

from __future__ import annotations

import os
import shutil
import sqlite3
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

logger = logging.getLogger("pipeline.export.ai_toolkit")


def find_ai_toolkit_dir(custom_path: Optional[str | Path] = None) -> Optional[Path]:
    """Probes standard locations or custom path to locate the AI-Toolkit installation."""
    candidates = []
    if custom_path:
        candidates.append(Path(custom_path))

    env_path = os.environ.get("AI_TOOLKIT_PATH")
    if env_path:
        candidates.append(Path(env_path))

    # Standard RunPod & local relative paths
    candidates.extend([
        Path("/app/ai-toolkit"),
        Path("../ai-toolkit"),
        Path("./ai-toolkit"),
    ])

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            # Verify it looks like ai-toolkit (has run.py or toolkit/ or config/)
            if resolved.is_dir() and (
                (resolved / "run.py").exists()
                or (resolved / "toolkit").exists()
                or (resolved / "config").exists()
                or (resolved / "datasets").exists()
                or (resolved / "ui").exists()
            ):
                return resolved
        except (OSError, PermissionError):
            continue

    return None


def link_dataset_folder(ai_toolkit_dir: Path, dataset_name: str, images_dir: Path) -> Optional[Path]:
    """Creates a symlink inside <ai_toolkit_dir>/datasets/<dataset_name> pointing to images_dir."""
    target_datasets_dir = ai_toolkit_dir / "datasets"
    target_datasets_dir.mkdir(parents=True, exist_ok=True)

    link_path = target_datasets_dir / dataset_name
    src_path = images_dir.resolve()

    if link_path.is_symlink():
        try:
            if link_path.resolve() == src_path:
                logger.debug(f"Symlink already points to {src_path}: {link_path}")
                return link_path
            link_path.unlink()
        except OSError:
            pass
    elif link_path.exists():
        logger.info(f"Dataset target path already exists as regular directory: {link_path}")
        return link_path

    try:
        os.symlink(str(src_path), str(link_path), target_is_directory=True)
        logger.info(f"Successfully symlinked dataset: {link_path} -> {src_path}")
        return link_path
    except (OSError, PermissionError) as exc:
        if os.name == "nt":
            try:
                import _winapi
                _winapi.CreateJunction(str(src_path), str(link_path))
                logger.info(f"Successfully created directory junction: {link_path} -> {src_path}")
                return link_path
            except Exception as junction_exc:
                logger.warning(f"Could not create junction on Windows ({junction_exc}).")
        logger.warning(f"Could not create symlink ({exc}).")
        return None


def register_dataset_in_sqlite(
    ai_toolkit_dir: Path,
    dataset_name: str,
    folder_path: Path,
    caption_ext: str = "txt",
) -> Optional[Path]:
    """Registers dataset into AI-Toolkit's SQLite DB (aitk_db.db) if available."""
    db_candidates = [
        ai_toolkit_dir / "aitk_db.db",
        ai_toolkit_dir / "ui" / "aitk_db.db",
        ai_toolkit_dir / "ui" / "prisma" / "aitk_db.db",
    ]

    target_db = None
    for db in db_candidates:
        if db.exists():
            target_db = db
            break

    if not target_db:
        logger.debug("No aitk_db.db found in AI-Toolkit directory. Skipping SQLite registration.")
        return None

    try:
        conn = sqlite3.connect(target_db)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND lower(name)='dataset';")
        row = cursor.fetchone()
        if not row:
            conn.close()
            return None

        table_name = row[0]
        cursor.execute(f"PRAGMA table_info({table_name});")
        cols = {col[1]: col for col in cursor.fetchall()}

        folder_str = str(folder_path.resolve()).replace("\\", "/")
        cursor.execute(f"SELECT id FROM {table_name} WHERE name = ? OR folder_path = ?", (dataset_name, folder_str))
        if cursor.fetchone():
            logger.info(f"Dataset '{dataset_name}' already registered in {target_db.name}")
            conn.close()
            return target_db

        now = datetime.now(timezone.utc).isoformat()
        insert_data: Dict[str, Any] = {}

        if "id" in cols:
            insert_data["id"] = str(uuid.uuid4())
        if "name" in cols:
            insert_data["name"] = dataset_name
        if "folder_path" in cols:
            insert_data["folder_path"] = folder_str
        elif "folderPath" in cols:
            insert_data["folderPath"] = folder_str
        if "caption_ext" in cols:
            insert_data["caption_ext"] = caption_ext
        elif "captionExt" in cols:
            insert_data["captionExt"] = caption_ext
        if "created_at" in cols:
            insert_data["created_at"] = now
        elif "createdAt" in cols:
            insert_data["createdAt"] = now
        if "updated_at" in cols:
            insert_data["updated_at"] = now
        elif "updatedAt" in cols:
            insert_data["updatedAt"] = now

        col_keys = list(insert_data.keys())
        placeholders = ", ".join(["?"] * len(col_keys))
        sql = f"INSERT INTO {table_name} ({', '.join(col_keys)}) VALUES ({placeholders})"
        cursor.execute(sql, [insert_data[k] for k in col_keys])
        conn.commit()
        conn.close()
        logger.info(f"Successfully registered dataset '{dataset_name}' in SQLite DB ({target_db})")
        return target_db
    except Exception as exc:
        logger.warning(f"Failed to register dataset in SQLite DB ({exc}). Continuing without DB entry.")
        return None


def install_training_config(
    ai_toolkit_dir: Path,
    images_dir: Path,
    trigger_word: str = "restrained_elegance",
    template_path: Optional[Path] = None,
    filename: str = "ai_toolkit_krea2_raw.yaml",
) -> Optional[Path]:
    """Generates and installs a ready-to-train YAML job into <ai_toolkit_dir>/config/."""
    target_config_dir = ai_toolkit_dir / "config"
    target_config_dir.mkdir(parents=True, exist_ok=True)

    dest_file = target_config_dir / filename
    src_template = template_path or Path("templates/ai_toolkit_krea2_raw.yaml")

    if src_template.exists():
        with open(src_template, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {
            "job": "extension",
            "config": {
                "name": "krea2_style_lora_raw",
                "process": [{
                    "type": "sd_trainer",
                    "training_folder": "output/krea2_style_lora_raw",
                    "device": "cuda:0",
                    "trigger_word": trigger_word,
                    "datasets": [{
                        "folder_path": "",
                        "caption_ext": "txt",
                    }],
                }],
            },
        }

    # Fix dataset folder path to absolute path
    abs_img_dir = str(images_dir.resolve()).replace("\\", "/")
    try:
        processes = cfg.get("config", {}).get("process", [])
        for proc in processes:
            for ds in proc.get("datasets", []):
                ds["folder_path"] = abs_img_dir
                ds["caption_ext"] = "txt"
    except Exception:
        pass

    with open(dest_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    logger.info(f"Installed training config: {dest_file}")
    return dest_file


def integrate_with_ai_toolkit(
    images_dir: Path,
    dataset_name: str = "restrained_elegance",
    trigger_word: str = "restrained_elegance",
    custom_ai_toolkit_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Main orchestrator for AI-Toolkit integration."""
    result: Dict[str, Any] = {
        "ai_toolkit_found": False,
        "ai_toolkit_dir": None,
        "linked": False,
        "link_path": None,
        "db_registered": False,
        "db_path": None,
        "config_installed": False,
        "config_path": None,
    }

    ai_toolkit_dir = find_ai_toolkit_dir(custom_ai_toolkit_dir)
    if not ai_toolkit_dir:
        logger.info("AI-Toolkit installation directory not detected. Automatic integration skipped.")
        return result

    result["ai_toolkit_found"] = True
    result["ai_toolkit_dir"] = str(ai_toolkit_dir)

    # 1. Symlink dataset
    link_path = link_dataset_folder(ai_toolkit_dir, dataset_name, images_dir)
    if link_path:
        result["linked"] = True
        result["link_path"] = str(link_path)

    # 2. Register in SQLite DB (if UI DB exists)
    folder_to_register = link_path or images_dir
    db_path = register_dataset_in_sqlite(ai_toolkit_dir, dataset_name, folder_to_register)
    if db_path:
        result["db_registered"] = True
        result["db_path"] = str(db_path)

    # 3. Install ready-to-run training configuration
    installed_cfg = install_training_config(ai_toolkit_dir, images_dir, trigger_word=trigger_word)
    if installed_cfg:
        result["config_installed"] = True
        result["config_path"] = str(installed_cfg)

    return result

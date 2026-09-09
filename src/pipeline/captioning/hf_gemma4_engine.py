"""Hugging Face Transformers Multimodal Engine for Google Gemma 4.

Google DeepMind released Gemma 4 (E2B, E4B, 12B, 31B) with native multimodal support
using AutoModelForMultimodalLM and AutoProcessor in transformers.
"""

from __future__ import annotations

import io
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image

logger = logging.getLogger("pipeline.captioning.hf_gemma4")


class Gemma4HfEngine:
    """Manages high-fidelity multimodal captioning with Google Gemma 4 using Transformers."""

    def __init__(
        self,
        model_name: str = "google/gemma-4-12B-it",
        device_map: str = "auto",
        temperature: float = 0.2,
        max_tokens: int = 350,
    ):
        self.model_name = model_name
        self.device_map = device_map
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._model = None
        self._processor = None

    def _ensure_model_loaded(self):
        """Lazy-loads Gemma 4 model and multimodal processor."""
        if self._model is not None:
            return

        import torch
        try:
            from transformers import AutoProcessor, AutoModelForMultimodalLM
        except ImportError:
            try:
                from transformers import AutoProcessor, AutoModelForCausalLM as AutoModelForMultimodalLM
            except ImportError as e:
                raise RuntimeError(
                    f"Transformers could not be loaded: {e}. "
                    "Please run 'pip install -U transformers torch accelerate'."
                ) from e

        logger.info("=" * 60)
        logger.info(f" Loading Google Gemma 4 Multimodal Engine: {self.model_name}")
        logger.info(f" Device Map: {self.device_map} | Precision: bfloat16 / auto")
        logger.info("=" * 60)

        self._processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        self._model = AutoModelForMultimodalLM.from_pretrained(
            self.model_name,
            dtype="auto",
            device_map=self.device_map,
            trust_remote_code=True,
        )
        logger.info("Google Gemma 4 loaded successfully into GPU memory.")

    def generate_batch(
        self,
        image_paths: List[Path],
        user_prompts: List[str],
        system_prompt: str,
        json_schema: Dict[str, Any],
        seed: int = 42,
        temperature: Optional[float] = None,
    ) -> List[Optional[Dict[str, Any]]]:
        """Generates structured JSON captions for a batch of images using Gemma 4."""
        self._ensure_model_loaded()
        import torch
        import time

        results: List[Optional[Dict[str, Any]]] = []
        schema_json = json.dumps(json_schema, indent=2)

        augmented_system = (
            f"{system_prompt}\n\n"
            f"You MUST output ONLY a valid JSON object strictly conforming to this schema:\n"
            f"{schema_json}\n"
            f"Do not write explanations, introductions, or markdown codeblocks. Output only the JSON."
        )

        total_imgs = len(image_paths)
        for idx, (img_path, u_prompt) in enumerate(zip(image_paths, user_prompts), start=1):
            t0 = time.time()
            logger.info(f" -> [{idx}/{total_imgs}] Captioning '{img_path.name}' with Gemma 4 12B...")
            try:
                with Image.open(img_path) as pil_img:
                    img_rgb = pil_img.convert("RGB")

                messages = [
                    {"role": "system", "content": augmented_system},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img_rgb},
                            {"type": "text", "text": f"{u_prompt}\nOutput raw JSON only:"},
                        ],
                    },
                ]

                # Format prompt using Gemma 4 native chat template
                inputs = self._processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    add_generation_prompt=True,
                ).to(self._model.device)

                input_len = inputs["input_ids"].shape[-1]
                temp = self.temperature if temperature is None else temperature
                do_sample = temp > 0.05

                gen_kwargs: Dict[str, Any] = {
                    "max_new_tokens": self.max_tokens,
                    "do_sample": do_sample,
                }
                if do_sample:
                    gen_kwargs["temperature"] = temp

                with torch.inference_mode():
                    outputs = self._model.generate(**inputs, **gen_kwargs)

                raw_text = self._processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

                # Extract and parse JSON
                m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
                if m:
                    clean_json = m.group(1)
                else:
                    m2 = re.search(r"(\{.*\})", raw_text, re.DOTALL)
                    clean_json = m2.group(1) if m2 else raw_text

                parsed = json.loads(clean_json)
                results.append(parsed)
                elapsed = time.time() - t0
                summary_desc = parsed.get("description", "")[:60] if isinstance(parsed, dict) else ""
                logger.info(f"    ✓ [{idx}/{total_imgs}] Generated in {elapsed:.1f}s: {summary_desc}...")

            except Exception as exc:
                elapsed = time.time() - t0
                logger.warning(f"    ✗ [{idx}/{total_imgs}] Generation failed for {img_path.name} after {elapsed:.1f}s: {exc}")
                results.append(None)

        return results

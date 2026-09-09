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
        temperature: float = 0.75,
        max_tokens: int = 1024,
        parallel_sub_batch_size: int = 8,
    ):
        self.model_name = model_name
        self.device_map = device_map
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.parallel_sub_batch_size = parallel_sub_batch_size
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
        logger.info(f" Parallel GPU Sub-Batch Size: {self.parallel_sub_batch_size}")
        logger.info("=" * 60)

        self._processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        # For causal multimodal autoregressive generation, left-padding is required
        if hasattr(self._processor, "tokenizer") and self._processor.tokenizer is not None:
            self._processor.tokenizer.padding_side = "left"
            if self._processor.tokenizer.pad_token_id is None:
                self._processor.tokenizer.pad_token_id = self._processor.tokenizer.eos_token_id

        self._model = AutoModelForMultimodalLM.from_pretrained(
            self.model_name,
            dtype="auto",
            device_map=self.device_map,
            trust_remote_code=True,
        )
        logger.info("Google Gemma 4 loaded successfully into GPU memory.")

    def _generate_parallel_chunk(
        self,
        image_paths: List[Path],
        user_prompts: List[str],
        augmented_system: str,
        temperature: Optional[float] = None,
        raw_text_mode: bool = True,
    ) -> List[Optional[Dict[str, Any]]]:
        """Runs true parallel GPU tensor generation on a chunk of images."""
        import torch

        conversations = []
        prompt_suffix = "\nOutput raw caption text only:" if raw_text_mode else "\nOutput raw JSON only:"
        for img_path, u_prompt in zip(image_paths, user_prompts):
            with Image.open(img_path) as pil_img:
                img_rgb = pil_img.convert("RGB")

            conversations.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_rgb},
                        {"type": "text", "text": f"{augmented_system}\n\n{u_prompt}{prompt_suffix}"},
                    ],
                }
            ])

        inputs = self._processor.apply_chat_template(
            conversations,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True,
        ).to(self._model.device)

        input_len = inputs["input_ids"].shape[1]
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

        results: List[Optional[Dict[str, Any]]] = []
        for i, img_path in enumerate(image_paths):
            try:
                raw_text = self._processor.decode(outputs[i][input_len:], skip_special_tokens=True).strip()
                if raw_text_mode:
                    clean_text = raw_text.strip()
                    clean_text = re.sub(r"^```(?:markdown|text)?\s*", "", clean_text)
                    clean_text = re.sub(r"\s*```$", "", clean_text).strip()
                    if clean_text.startswith('"') and clean_text.endswith('"'):
                        clean_text = clean_text[1:-1].strip()
                    w_count = len(clean_text.split())
                    results.append({
                        "caption_dense": clean_text,
                        "description": clean_text,
                        "word_count": w_count,
                        "approx_tokens": int(w_count * 1.35),
                    })
                else:
                    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
                    clean_json = m.group(1) if m else None
                    if not clean_json:
                        m2 = re.search(r"(\{.*\})", raw_text, re.DOTALL)
                        clean_json = m2.group(1) if m2 else raw_text
                    parsed = json.loads(clean_json)
                    results.append(parsed)
            except Exception as parse_err:
                logger.warning(f"Parse failed for {img_path.name}: {parse_err}")
                results.append(None)

        return results

    def _generate_single(
        self,
        img_path: Path,
        u_prompt: str,
        augmented_system: str,
        temperature: Optional[float] = None,
        raw_text_mode: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Fallback for generating a single image safely."""
        import torch
        with Image.open(img_path) as pil_img:
            img_rgb = pil_img.convert("RGB")

        prompt_suffix = "\nOutput raw caption text only:" if raw_text_mode else "\nOutput raw JSON only:"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_rgb},
                    {"type": "text", "text": f"{augmented_system}\n\n{u_prompt}{prompt_suffix}"},
                ],
            }
        ]

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
        if raw_text_mode:
            clean_text = raw_text.strip()
            clean_text = re.sub(r"^```(?:markdown|text)?\s*", "", clean_text)
            clean_text = re.sub(r"\s*```$", "", clean_text).strip()
            if clean_text.startswith('"') and clean_text.endswith('"'):
                clean_text = clean_text[1:-1].strip()
            w_count = len(clean_text.split())
            return {
                "caption_dense": clean_text,
                "description": clean_text,
                "word_count": w_count,
                "approx_tokens": int(w_count * 1.35),
            }

        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
        clean_json = m.group(1) if m else None
        if not clean_json:
            m2 = re.search(r"(\{.*\})", raw_text, re.DOTALL)
            clean_json = m2.group(1) if m2 else raw_text

        return json.loads(clean_json)

    def generate_batch(
        self,
        image_paths: List[Path],
        user_prompts: List[str],
        system_prompt: str,
        json_schema: Optional[Dict[str, Any]] = None,
        seed: int = 42,
        temperature: Optional[float] = None,
        raw_text_mode: bool = True,
    ) -> List[Optional[Dict[str, Any]]]:
        """Generates structured captions using parallel GPU batches with automatic fallback.
        
        When raw_text_mode=True (default), Gemma outputs pure descriptive text to maximize
        token capacity for 400+ token descriptions without wasting tokens on JSON syntax.
        """
        self._ensure_model_loaded()
        import time

        results: List[Optional[Dict[str, Any]]] = []

        if raw_text_mode or not json_schema:
            augmented_system = system_prompt
        else:
            schema_json = json.dumps(json_schema, indent=2)
            augmented_system = (
                f"{system_prompt}\n\n"
                f"You MUST output ONLY a valid JSON object strictly conforming to this schema:\n"
                f"{schema_json}\n"
                f"Do not write explanations, introductions, or markdown codeblocks. Output only the JSON."
            )

        total_imgs = len(image_paths)
        chunk_size = max(1, self.parallel_sub_batch_size)

        for chunk_start in range(0, total_imgs, chunk_size):
            chunk_paths = image_paths[chunk_start : chunk_start + chunk_size]
            chunk_prompts = user_prompts[chunk_start : chunk_start + chunk_size]
            chunk_end = min(chunk_start + chunk_size, total_imgs)
            t0 = time.time()

            # Attempt true parallel GPU tensor generation
            try:
                logger.info(f" -> Sub-batch [{chunk_start + 1}-{chunk_end}/{total_imgs}]: Parallel GPU inference on {len(chunk_paths)} images...")
                chunk_results = self._generate_parallel_chunk(
                    image_paths=chunk_paths,
                    user_prompts=chunk_prompts,
                    augmented_system=augmented_system,
                    temperature=temperature,
                    raw_text_mode=raw_text_mode,
                )
                elapsed = time.time() - t0
                per_img = elapsed / max(1, len(chunk_paths))
                logger.info(f"    ✓ Sub-batch [{chunk_start + 1}-{chunk_end}/{total_imgs}] done in {elapsed:.1f}s ({per_img:.2f}s/img)")
                results.extend(chunk_results)

            except Exception as batch_exc:
                logger.warning(f"Parallel sub-batch note ({batch_exc}). Gracefully executing sequentially for this chunk...")
                for idx, (img_path, u_prompt) in enumerate(zip(chunk_paths, chunk_prompts), start=chunk_start + 1):
                    t_seq = time.time()
                    try:
                        single_res = self._generate_single(
                            img_path=img_path,
                            u_prompt=u_prompt,
                            augmented_system=augmented_system,
                            temperature=temperature,
                            raw_text_mode=raw_text_mode,
                        )
                        results.append(single_res)
                        elapsed_seq = time.time() - t_seq
                        desc = single_res.get("description", "")[:50] if isinstance(single_res, dict) else ""
                        logger.info(f"    ✓ [{idx}/{total_imgs}] Sequential generated in {elapsed_seq:.1f}s: {desc}...")
                    except Exception as single_exc:
                        logger.warning(f"    ✗ [{idx}/{total_imgs}] Generation failed for {img_path.name}: {single_exc}")
                        results.append(None)

        return results

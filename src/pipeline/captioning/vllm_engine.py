"""vLLM Offline-Batch Engine with Guided Decoding for Qwen-VL.

Features lazy-loading of GPU dependencies (vllm, torch) to allow import on CPU machines.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image

logger = logging.getLogger("pipeline.captioning.vllm_engine")


class QwenVLEngine:
    """Manages high-throughput offline batch inference with Qwen-VL using vLLM."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 4096,
        temperature: float = 0.2,
        max_tokens: int = 250,
    ):
        self.model_name = model_name
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._llm = None

    def _ensure_model_loaded(self):
        """Lazy-loads vLLM engine on first inference invocation."""
        if self._llm is not None:
            return

        import os
        # Disable flashinfer sampler to avoid nvcc JIT compilation in containers without full CUDA toolkit
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

        try:
            from vllm import LLM
            logger.info(f"Loading vLLM Offline Engine with {self.model_name} (GPU util: {self.gpu_memory_utilization})...")
            self._llm = LLM(
                model=self.model_name,
                trust_remote_code=True,
                max_model_len=self.max_model_len,
                gpu_memory_utilization=self.gpu_memory_utilization,
                limit_mm_per_prompt={"image": 1},
            )
        except ImportError as exc:
            raise RuntimeError(
                f"vLLM could not be imported: {exc}. "
                "If numpy version conflict, downgrade with 'pip install \"numpy<2\"'."
            ) from exc

    def generate_batch(
        self,
        image_paths: List[Path],
        user_prompts: List[str],
        system_prompt: str,
        json_schema: Dict[str, Any],
        seed: int = 42,
        temperature: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Processes a batch of images through vLLM using Guided Decoding with assistant prefill '{'."""
        self._ensure_model_loaded()

        temp = self.temperature if temperature is None else temperature
        sampling_kwargs: Dict[str, Any] = {
            "temperature": temp,
            "max_tokens": self.max_tokens,
            "seed": seed,
        }

        # Support both modern vLLM (StructuredOutputsParams) and legacy (GuidedDecodingParams)
        try:
            from vllm.sampling_params import StructuredOutputsParams
            sampling_kwargs["structured_outputs"] = StructuredOutputsParams(json=json_schema)
        except ImportError:
            try:
                from vllm.sampling_params import GuidedDecodingParams
                sampling_kwargs["guided_decoding"] = GuidedDecodingParams(json=json_schema)
            except ImportError:
                sampling_kwargs["structured_outputs"] = {"json": json_schema}

        sampling_params = SamplingParams(**sampling_kwargs)

        inputs = []
        for img_path, u_prompt in zip(image_paths, user_prompts):
            with Image.open(img_path) as pil_img:
                img_copy = pil_img.convert("RGB")

            # Format Qwen-VL chat prompt with assistant prefill '{'
            prompt_str = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{u_prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n{{"
            )
            inputs.append({
                "prompt": prompt_str,
                "multi_modal_data": {"image": img_copy},
            })

        logger.info(f"Submitting batch of {len(inputs)} requests to vLLM offline engine...")
        outputs = self._llm.generate(inputs, sampling_params=sampling_params)

        results = []
        for out in outputs:
            generated_text = out.outputs[0].text.strip()
            # If assistant prefilled '{', prepend it if model continued inside the object
            if not generated_text.startswith("{"):
                full_json_str = "{" + generated_text
            else:
                full_json_str = generated_text

            try:
                parsed = json.loads(full_json_str)
            except Exception as e:
                logger.warning(f"Failed to decode vLLM JSON output: {full_json_str[:100]}... Error: {e}")
                parsed = {"raw_output": generated_text, "error": str(e)}

            results.append(parsed)

        return results

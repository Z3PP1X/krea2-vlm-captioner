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
        # Disable experimental v1 engine to ensure stable multimodal inference
        os.environ.setdefault("VLLM_USE_V1", "0")

        # Compatibility patch for transformers Qwen2Tokenizer vs vLLM tokenizer cache
        try:
            from transformers.tokenization_utils_base import PreTrainedTokenizerBase
            if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
                PreTrainedTokenizerBase.all_special_tokens_extended = property(
                    lambda self: getattr(self, "all_special_tokens", [])
                )
            import transformers
            for tok_name in ["PreTrainedTokenizer", "Qwen2Tokenizer"]:
                tok_cls = getattr(transformers, tok_name, None)
                if tok_cls is not None and not hasattr(tok_cls, "all_special_tokens_extended"):
                    setattr(tok_cls, "all_special_tokens_extended", property(lambda self: getattr(self, "all_special_tokens", [])))
        except Exception as tok_err:
            logger.debug(f"Tokenizer compatibility patch skipped: {tok_err}")

        try:
            from vllm import LLM

            # Hotfix for Gemma 4 RoPE scaling parameter schema mismatch in vLLM
            try:
                import vllm.transformers_utils.config as vllm_cfg
                _orig_dict = getattr(vllm_cfg, "patch_rope_scaling_dict", None)
                if _orig_dict is not None:
                    def _safe_patch_dict(rope_scaling):
                        if not isinstance(rope_scaling, dict):
                            return
                        if "rope_type" not in rope_scaling:
                            rope_scaling["rope_type"] = rope_scaling.get("type", "default")
                        try:
                            _orig_dict(rope_scaling)
                        except ValueError:
                            pass
                    vllm_cfg.patch_rope_scaling_dict = _safe_patch_dict

                _orig_patch = getattr(vllm_cfg, "patch_rope_scaling", None)
                if _orig_patch is not None:
                    def _safe_patch(cfg):
                        try:
                            _orig_patch(cfg)
                        except ValueError:
                            pass
                    vllm_cfg.patch_rope_scaling = _safe_patch
            except Exception as patch_e:
                logger.debug(f"vLLM rope patch ignored: {patch_e}")

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

    def build_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """Formats model-family specific multimodal chat prompt with assistant '{' prefill."""
        m_lower = self.model_name.lower()
        if "gemma-4" in m_lower or "gemma4" in m_lower:
            # Gemma 4 native turn format
            return (
                f"<|turn>system\n{system_prompt}<turn|>\n"
                f"<|turn>user\n<|image|>{user_prompt}<turn|>\n"
                f"<|turn>model\n{{"
            )
        elif "paligemma" in m_lower or "gemma" in m_lower:
            # PaliGemma / Gemma 2 turn format
            return (
                f"<start_of_turn>user\n<image>{system_prompt}\n{user_prompt}<end_of_turn>\n"
                f"<start_of_turn>model\n{{"
            )
        else:
            # Default Qwen-VL chat format
            return (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{user_prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n{{"
            )

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

        from vllm import SamplingParams

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

            prompt_str = self.build_prompt(system_prompt, u_prompt)
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

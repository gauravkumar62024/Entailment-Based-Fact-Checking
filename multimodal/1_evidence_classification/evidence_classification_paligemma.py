# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Unified Multimodal Evidence Classification System
# Hybrid: vLLM for Qwen2VL/Idefics3, HF Transformers for PaliGemma

# Supported Modality Combinations:
# CLAIM: text + image, text only, image only
# EVIDENCE: text + image, text only, image only, none

# Supported Models:
# - PaliGemma (HF, 4-bit)
# - Qwen2VL (vLLM)
# - Idefics3 (vLLM)

# Supported Datasets:
# - Factify
# - MOCHEG
# - MR2
# - VERITE  <-- added
# """

# import os
# import re
# import csv
# import json
# import argparse
# import hashlib
# import sys
# from io import BytesIO
# from typing import List, Dict, Any, Optional, Tuple
# from dataclasses import dataclass
# from enum import Enum

# import requests
# from PIL import Image
# from tqdm import tqdm
# import torch
# import numpy as np
# import pandas as pd

# from transformers import AutoProcessor
# # vLLM imports
# try:
#     from vllm import LLM, SamplingParams
#     VLLM_AVAILABLE = True
# except ImportError:
#     VLLM_AVAILABLE = False
#     print("[WARN] vLLM not available")

# # HF Transformers imports for PaliGemma
# from transformers import (
#     PaliGemmaProcessor,
#     PaliGemmaForConditionalGeneration,
#     BitsAndBytesConfig,
# )

# csv.field_size_limit(sys.maxsize)


# # ============================================================================
# # CONFIGURATION & ENUMS
# # ============================================================================

# class ModalityType(Enum):
#     """Supported modality combinations"""
#     TEXT_IMAGE = "text_image"
#     TEXT_ONLY = "text_only"
#     IMAGE_ONLY = "image_only"
#     NONE = "none"


# @dataclass
# class ClaimData:
#     """Structured claim representation"""
#     claim_id: str
#     text: Optional[str] = None
#     ocr: Optional[str] = None
#     image_url: Optional[str] = None
#     image: Optional[Image.Image] = None
#     modality: ModalityType = ModalityType.NONE

#     def get_full_text(self) -> str:
#         """Combine text and OCR"""
#         parts = []
#         if self.text:
#             parts.append(self.text.strip())
#         if self.ocr:
#             parts.append(self.ocr.strip())
#         return " ".join(parts)

#     def has_text(self) -> bool:
#         return bool(self.text or self.ocr)

#     def has_image(self) -> bool:
#         return self.image is not None


# @dataclass
# class EvidenceData:
#     """Structured evidence representation"""
#     evidence_id: str
#     text: Optional[str] = None
#     ocr: Optional[str] = None
#     image_url: Optional[str] = None
#     image: Optional[Image.Image] = None
#     sentences: Optional[List[str]] = None
#     modality: ModalityType = ModalityType.NONE

#     def get_full_text(self) -> str:
#         """Combine text and OCR"""
#         parts = []
#         if self.text:
#             parts.append(self.text.strip())
#         if self.ocr:
#             parts.append(self.ocr.strip())
#         return " ".join(parts)

#     def has_text(self) -> bool:
#         return bool(self.text or self.ocr or self.sentences)

#     def has_image(self) -> bool:
#         return self.image is not None


# MODEL_CONFIG = {
#     # PaliGemma - Uses HF Transformers (4-bit quantized)
#     "paligemma": {
#         "model_id": "google/paligemma2-3b-pt-448",
#         "backend": "hf",  # Use Hugging Face Transformers
#         "max_new_tokens": 256,
#         "image_size": 448,
#         "low_res_size": 224,
#         "max_model_len": 8192,
#         "use_chat_template": False,
#         "quantization": "4bit",  # 4-bit quantization
#     },

#     # Qwen2VL - Uses vLLM
#     "qwen2vl": {
#         "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
#         "backend": "vllm",
#         "max_new_tokens": 256,
#         "image_size": 768,
#         "max_model_len": 10000,
#         "visual_token_count": 256,
#         "use_chat_template": True,
#     },

#     # Idefics3 - Uses vLLM
#     "idefics3": {
#         "model_id": "leon-se/Idefics3-8B-Llama3-bnb_nf4",  # or HuggingFaceM4/Idefics3-8B-Llama3
#         "backend": "vllm",
#         "max_new_tokens": 256,
#         "image_size": 768,
#         "max_model_len": 4096,  # increased from 2048 to avoid prompt length errors
#         "visual_token_count": 256,
#         "use_chat_template": True,
#     },

#     # AWQ Quantized models (vLLM)
#     "qwen2vl-awq": {
#         "model_id": "Qwen/Qwen2-VL-7B-Instruct-AWQ",
#         "backend": "vllm",
#         "max_new_tokens": 256,
#         "image_size": 768,
#         "max_model_len": 8192,
#         "visual_token_count": 256,
#         "use_chat_template": True,
#         "quantization": "awq",
#         "dtype": "half",
#     },
#     "idefics3-awq": {
#         "model_id": "ronantakizawa/idefics3-8b-llama3-awq",
#         "backend": "vllm",
#         "max_new_tokens": 256,
#         "image_size": 768,
#         "max_model_len": 8192,
#         "visual_token_count": 256,
#         "use_chat_template": True,
#         "quantization": "awq",
#         "dtype": "half",
#     },
# }


# # ============================================================================
# # UTILITY FUNCTIONS
# # ============================================================================

# def is_valid_image(img: Optional[Image.Image]) -> bool:
#     """Validate image before processing"""
#     try:
#         if img is None:
#             return False
#         arr = np.array(img)
#         if len(arr.shape) != 3 or arr.shape[2] != 3:
#             return False
#         h, w, _ = arr.shape
#         if h < 64 or w < 64 or h > 4096 or w > 4096:
#             return False
#         if np.std(arr) < 2:
#             return False
#         if np.any(np.isnan(arr)):
#             return False
#         return True
#     except Exception:
#         return False


# def resize_image_for_paligemma(img: Image.Image, target_size: int = 448) -> Optional[Image.Image]:
#     """
#     Fixed image resizing for PaliGemma

#     PaliGemma2 supports: 224, 448, or 896
#     Default: 448 for paligemma2-3b-pt-448
#     """
#     if img is None:
#         return None
#     try:
#         if img.mode != "RGB":
#             img = img.convert("RGB")
#         img_resized = img.resize((target_size, target_size), Image.LANCZOS)
#         if not is_valid_image(img_resized):
#             print(f"[WARN] Invalid image after resize: {img_resized.size}")
#             return None
#         return img_resized
#     except Exception as e:
#         print(f"[ERROR] Image resize failed: {e}")
#         return None


# def download_image_fixed(url: str, max_size: int = 768, cache_dir: str = "image_cache") -> Optional[Image.Image]:
#     """
#     Download and resize for PaliGemma (always to 448x448).
#     """
#     if not url or str(url).strip().lower() in {"", "nan", "none"}:
#         return None

#     os.makedirs(cache_dir, exist_ok=True)
#     url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()
#     ext = os.path.splitext(url)[-1].lower()
#     if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
#         ext = ".jpg"

#     local_path = os.path.join(cache_dir, f"{url_hash}_pali448{ext}")

#     if os.path.exists(local_path):
#         try:
#             img = Image.open(local_path).convert("RGB")
#             return img
#         except Exception:
#             os.remove(local_path)

#     try:
#         r = requests.get(url, timeout=30)
#         r.raise_for_status()
#         img = Image.open(BytesIO(r.content)).convert("RGB")
#         img_resized = resize_image_for_paligemma(img, target_size=448)
#         if img_resized:
#             img_resized.save(local_path)
#             return img_resized
#         return None
#     except Exception as e:
#         print(f"[WARN] Image download failed: {url} ({e})")
#         return None


# def load_local_image_fixed(path: str, max_size: int = 768) -> Optional[Image.Image]:
#     """
#     Load local image and resize for PaliGemma (448x448).
#     """
#     if not path or not os.path.exists(path):
#         return None
#     try:
#         img = Image.open(path).convert("RGB")
#         img_resized = resize_image_for_paligemma(img, target_size=448)
#         return img_resized
#     except Exception as e:
#         print(f"[WARN] Image load failed: {path} ({e})")
#         return None


# def load_local_image(path: str, max_size: int = 768) -> Optional[Image.Image]:
#     """
#     Generic local image loader (for vLLM models like Qwen2VL / Idefics3).
#     """
#     if not path or not os.path.exists(path):
#         return None
#     try:
#         img = Image.open(path).convert("RGB")
#         img.thumbnail((max_size, max_size))
#         return img
#     except Exception as e:
#         print(f"[WARN] Generic image load failed: {path} ({e})")
#         return None


# _SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9""\(\[])')


# def split_into_sentences(text: str) -> List[str]:
#     """Split text into sentences"""
#     text = (text or "").strip()
#     if not text:
#         return []
#     chunks = _SENT_SPLIT_RE.split(text)
#     sents = [re.sub(r"\s+", " ", c).strip() for c in chunks]
#     return [s for s in sents if len(s) > 2]


# def truncate_text(processor, text: str, max_tokens: int) -> str:
#     """Truncate text to max tokens"""
#     if not text:
#         return text
#     tokenizer = getattr(processor, "tokenizer", processor)
#     tokens = tokenizer.encode(text, add_special_tokens=False)
#     if len(tokens) <= max_tokens:
#         return text
#     truncated = tokens[:max_tokens]
#     return tokenizer.decode(truncated, skip_special_tokens=True)


# _LABEL_RE = re.compile(r"(SUPPORT|REFUTE)", re.IGNORECASE)


# def parse_label(response: str) -> str:
#     """Extract SUPPORT/REFUTE from response"""
#     m = _LABEL_RE.search(response or "")
#     if not m:
#         return "REFUTE"
#     return m.group(1).upper()


# # ============================================================================
# # PROMPT GENERATION (MODALITY-AWARE)
# # ============================================================================

# class PromptBuilder:
#     """Build prompts based on available modalities"""

#     @staticmethod
#     def build_text_evidence_prompt(
#         claim: ClaimData,
#         evidence_sentence: str,
#         processor,
#         max_tokens: int = 512,  # reduced to avoid context overflow
#     ) -> str:
#         """Prompt for text evidence classification"""
#         claim_text = truncate_text(processor, claim.get_full_text(), max_tokens // 3)
#         evidence_sentence = truncate_text(processor, evidence_sentence, max_tokens // 3)

#         context_parts = []
#         if claim.has_image():
#             context_parts.append("An image of the claim is provided for visual context.")
#         if claim.has_text():
#             context_parts.append(f"CLAIM TEXT: {claim_text}")

#         context = "\n".join(context_parts) if context_parts else "CLAIM: [Visual only - see image]"

#         return (
#             "You are a fact-checking assistant that classifies textual evidence.\n\n"
#             "### Task:\n"
#             "Determine whether the evidence sentence SUPPORTS or REFUTES the given claim.\n\n"
#             "### Instructions:\n"
#             "- Analyze the textual content and its logical relationship to the claim.\n"
#             "- If a claim image is provided, consider visual information as well.\n"
#             "- Focus on factual consistency between evidence and claim.\n\n"
#             f"{context}\n\n"
#             f"**EVIDENCE SENTENCE:** {evidence_sentence}\n\n"
#             "### Response:\n"
#             "Reply with exactly one word: SUPPORT or REFUTE\n"
#         )

#     @staticmethod
#     def build_visual_evidence_prompt(
#         claim: ClaimData,
#         evidence: EvidenceData,
#         processor,
#         max_tokens: int = 512,  # reduced to avoid context overflow
#     ) -> str:
#         """Prompt for visual evidence classification"""
#         claim_text = truncate_text(processor, claim.get_full_text(), max_tokens // 3)
#         evidence_text = truncate_text(processor, evidence.get_full_text(), max_tokens // 3)

#         context_parts = []
#         if claim.has_text():
#             context_parts.append(f"CLAIM TEXT: {claim_text}")
#         else:
#             context_parts.append("CLAIM: [Visual only - see claim image]")

#         if evidence.has_text():
#             context_parts.append(f"EVIDENCE TEXT/OCR: {evidence_text}")

#         context = "\n".join(context_parts)

#         return (
#             "You are a multimodal fact-checking assistant that analyzes visual evidence.\n\n"
#             "### Task:\n"
#             "Determine whether the evidence image SUPPORTS or REFUTES the given claim.\n\n"
#             "### Instructions:\n"
#             "- Analyze visual content: objects, scenes, text, layout, style.\n"
#             "- Compare evidence image with the claim (text and/or image).\n"
#             "- If evidence text/OCR is provided, use it to aid analysis.\n"
#             "- Consider context and factual consistency.\n\n"
#             f"{context}\n\n"
#             "### Response:\n"
#             "Reply with exactly one word: SUPPORT or REFUTE\n"
#         )


# # ============================================================================
# # MODEL WRAPPERS
# # ============================================================================

# class HFPaliGemmaWrapper:
#     """Hugging Face PaliGemma wrapper with 4-bit quantization"""

#     def __init__(self, model_id: str, max_new_tokens: int = 256):
#         print(f"[INFO] Loading PaliGemma with HF Transformers (4-bit): {model_id}")

#         quantization_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_compute_dtype=torch.float16,
#             bnb_4bit_use_double_quant=True,
#             bnb_4bit_quant_type="nf4",
#         )

#         self.processor = PaliGemmaProcessor.from_pretrained(
#             model_id, trust_remote_code=True
#         )

#         self.model = PaliGemmaForConditionalGeneration.from_pretrained(
#             model_id,
#             quantization_config=quantization_config,
#             device_map="auto",
#             trust_remote_code=True,
#             torch_dtype=torch.float16,
#         )

#         # Set pad_token to eos_token to avoid generation warnings
#         self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
#         self.model.generation_config.pad_token_id = self.processor.tokenizer.eos_token_id

#         self.model.eval()
#         self.max_new_tokens = max_new_tokens
#         self.device = next(self.model.parameters()).device

#         print(f"[INFO] PaliGemma loaded on device: {self.device}")

#     def generate(self, prompts: List[str], images_lists: List[List[Image.Image]]) -> List[str]:
#         responses: List[str] = []

#         for prompt, images in tqdm(
#             list(zip(prompts, images_lists)),
#             total=len(prompts),
#             desc="PaliGemma (HF)",
#             leave=False,
#         ):
#             try:
#                 valid_images = [img for img in images if is_valid_image(img)]

#                 # PaliGemma requires images — for text-only we use a white placeholder
#                 if not valid_images:
#                     blank_img = Image.new("RGB", (448, 448), color=(255, 255, 255))
#                     valid_images = [blank_img]
#                     print("[INFO] Text-only input, using blank placeholder image")

#                 image_tokens = "<image>" * len(valid_images)
#                 full_prompt = image_tokens + prompt

#                 inputs = self.processor(
#                     text=full_prompt,
#                     images=valid_images,
#                     return_tensors="pt",
#                     padding=True,
#                 ).to(self.device)

#                 with torch.no_grad():
#                     outputs = self.model.generate(
#                         **inputs,
#                         max_new_tokens=self.max_new_tokens,
#                         do_sample=False,
#                         pad_token_id=self.processor.tokenizer.eos_token_id,
#                         repetition_penalty=1.1,
#                         eos_token_id=self.processor.tokenizer.eos_token_id,
#                     )

#                 generated_ids = outputs[0][inputs.input_ids.shape[1]:]
#                 response = self.processor.decode(
#                     generated_ids, skip_special_tokens=True
#                 ).strip()

#                 if not response or len(response) < 3:
#                     response = "REFUTE"

#                 responses.append(response)

#             except Exception as e:
#                 print(f"[ERROR] PaliGemma generation failed: {e}")
#                 import traceback
#                 traceback.print_exc()
#                 responses.append("REFUTE")

#         return responses


# class VLLMWrapper:
#     """vLLM wrapper for Qwen2VL and Idefics3"""

#     def __init__(self, model_key: str, config: dict, quantize: Optional[str] = None):
#         if not VLLM_AVAILABLE:
#             raise RuntimeError("vLLM not available but required for this model")

#         print(f"[INFO] Loading {model_key} with vLLM: {config['model_id']}")

#         self.model_key = model_key
#         self.config = config
#         self.processor = AutoProcessor.from_pretrained(
#             config["model_id"], trust_remote_code=True
#         )

#         llm_kwargs = {
#             "model": config["model_id"],
#             "max_model_len": config["max_model_len"],
#             "trust_remote_code": True,
#         }

#         if "quantization" in config:
#             llm_kwargs["quantization"] = config["quantization"]
#             llm_kwargs["dtype"] = config.get("dtype", "half")
#             llm_kwargs["gpu_memory_utilization"] = 0.7
#             print(f"[INFO] Using {config['quantization']} quantization")
#         elif quantize:
#             llm_kwargs["quantization"] = quantize
#             llm_kwargs["dtype"] = "half"
#             llm_kwargs["gpu_memory_utilization"] = 0.7
#             print(f"[INFO] Applying {quantize} quantization")
#         else:
#             llm_kwargs["dtype"] = "bfloat16"
#             llm_kwargs["gpu_memory_utilization"] = 0.9

#         llm_kwargs["limit_mm_per_prompt"] = {"image": 2}

#         try:
#             self.llm = LLM(**llm_kwargs)
#         except Exception as e:
#             print(f"[ERROR] Failed to load: {e}")
#             print("[INFO] Retrying with reduced memory...")
#             llm_kwargs["gpu_memory_utilization"] = 0.5
#             self.llm = LLM(**llm_kwargs)

#         self.sampling_params = SamplingParams(
#             temperature=0.0,
#             max_tokens=config["max_new_tokens"],
#             skip_special_tokens=True,
#         )

#     def _format_prompt(self, prompt: str, images: List[Image.Image]) -> str:
#         """Format prompt with chat template"""
#         valid_imgs = [img for img in images if is_valid_image(img)]

#         content = [{"type": "image"} for _ in valid_imgs]
#         content.append({"type": "text", "text": prompt})
#         messages = [{"role": "user", "content": content}]

#         return self.processor.apply_chat_template(
#             messages, add_generation_prompt=True, tokenize=False
#         )

#     def generate(self, prompts: List[str], images_lists: List[List[Image.Image]]) -> List[str]:
#         requests = []
#         for prompt, imgs in zip(prompts, images_lists):
#             valid_imgs = [img for img in imgs if is_valid_image(img)]
#             formatted_prompt = self._format_prompt(prompt, valid_imgs)

#             requests.append(
#                 {
#                     "prompt": formatted_prompt,
#                     "multi_modal_data": {"image": valid_imgs} if valid_imgs else {},
#                 }
#             )

#         try:
#             outputs = self.llm.generate(requests, self.sampling_params)
#             return [out.outputs[0].text.strip() for out in outputs]
#         except Exception as e:
#             print(f"[ERROR] vLLM generation failed: {e}")
#             return ["ERROR"] * len(prompts)


# class UnifiedVLM:
#     """Unified wrapper that routes to HF or vLLM based on model"""

#     def __init__(self, model_key: str, quantize: Optional[str] = None):
#         assert model_key in MODEL_CONFIG, f"Unknown model: {model_key}"

#         self.model_key = model_key
#         self.config = MODEL_CONFIG[model_key]
#         self.backend = self.config["backend"]

#         if self.backend == "hf":
#             self.model = HFPaliGemmaWrapper(
#                 self.config["model_id"], self.config["max_new_tokens"]
#             )
#             self.processor = self.model.processor
#         elif self.backend == "vllm":
#             self.model = VLLMWrapper(model_key, self.config, quantize)
#             self.processor = self.model.processor
#         else:
#             raise ValueError(f"Unknown backend: {self.backend}")

#     def classify_batch(
#         self,
#         prompts: List[str],
#         images_lists: List[List[Image.Image]],
#     ) -> List[str]:
#         return self.model.generate(prompts, images_lists)


# # ============================================================================
# # EVIDENCE CLASSIFIER
# # ============================================================================

# class EvidenceClassifier:
#     """Classify evidence based on available modalities"""

#     def __init__(self, vlm: UnifiedVLM):
#         self.vlm = vlm
#         self.prompt_builder = PromptBuilder()

#     def classify_text_evidence(
#         self,
#         claim: ClaimData,
#         sentences: List[str],
#         batch_size: int = 8,
#     ) -> List[Dict[str, Any]]:
#         if not sentences:
#             return []

#         predictions: List[Dict[str, Any]] = []

#         for i in range(0, len(sentences), batch_size):
#             batch_sents = sentences[i : i + batch_size]

#             prompts = [
#                 self.prompt_builder.build_text_evidence_prompt(
#                     claim, sent, self.vlm.processor
#                 )
#                 for sent in batch_sents
#             ]

#             images_lists = [
#                 [claim.image] if claim.has_image() else [] for _ in batch_sents
#             ]

#             responses = self.vlm.classify_batch(prompts, images_lists)

#             for sent, resp in zip(batch_sents, responses):
#                 predictions.append(
#                     {
#                         "sentence": sent,
#                         "prediction": parse_label(resp),
#                         "raw_response": resp,
#                     }
#                 )

#         return predictions

#     def classify_visual_evidence(
#         self,
#         claim: ClaimData,
#         evidence: EvidenceData,
#     ) -> Tuple[str, str]:
#         if not evidence.has_image():
#             return "NO_EVIDENCE", "No visual evidence available"

#         prompt = self.prompt_builder.build_visual_evidence_prompt(
#             claim, evidence, self.vlm.processor
#         )

#         images: List[Image.Image] = []
#         if claim.has_image():
#             images.append(claim.image)
#         images.append(evidence.image)

#         try:
#             responses = self.vlm.classify_batch([prompt], [images])
#             response = responses[0]
#             label = parse_label(response)
#             return label, response
#         except Exception as e:
#             return "ERROR", f"Classification failed: {e}"

#     def classify_combined(
#         self,
#         claim: ClaimData,
#         evidence: EvidenceData,
#         batch_size: int = 8,
#     ) -> Dict[str, Any]:
#         result: Dict[str, Any] = {
#             "claim_id": claim.claim_id,
#             "claim_modality": claim.modality.value,
#             "evidence_modality": evidence.modality.value,
#             "supporting_text": [],
#             "refuting_text": [],
#             "text_predictions": [],
#             "visual_label": None,
#             "visual_response": None,
#         }

#         if evidence.has_text():
#             if not evidence.sentences:
#                 evidence.sentences = split_into_sentences(evidence.get_full_text())

#             text_preds = self.classify_text_evidence(
#                 claim, evidence.sentences, batch_size
#             )

#             result["text_predictions"] = text_preds
#             result["supporting_text"] = [
#                 p["sentence"] for p in text_preds if p["prediction"] == "SUPPORT"
#             ]
#             result["refuting_text"] = [
#                 p["sentence"] for p in text_preds if p["prediction"] == "REFUTE"
#             ]

#         if evidence.has_image():
#             label, response = self.classify_visual_evidence(claim, evidence)
#             result["visual_label"] = label
#             result["visual_response"] = response

#         return result


# # ============================================================================
# # DATA LOADERS
# # ============================================================================

# def load_factify_data_fixed(csv_path: str, image_size: int, cache_dir: str):
#     """
#     Factify loader (kept for completeness).
#     Uses PaliGemma-style image preprocessing.
#     """
#     pairs: List[Tuple[ClaimData, EvidenceData]] = []

#     with open(csv_path, "r", encoding="utf-8") as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             claim = ClaimData(
#                 claim_id=str(row.get("Id", "")).strip(),
#                 text=row.get("claim", "").strip() or None,
#                 ocr=row.get("claim_ocr", "").strip() or None,
#                 image_url=row.get("claim_image", "").strip() or None,
#             )

#             claim.image = download_image_fixed(claim.image_url, cache_dir=cache_dir)
#             if claim.image and claim.image.size != (448, 448):
#                 claim.image = resize_image_for_paligemma(claim.image, 448)

#             if claim.has_text() and claim.has_image():
#                 claim.modality = ModalityType.TEXT_IMAGE
#             elif claim.has_text():
#                 claim.modality = ModalityType.TEXT_ONLY
#             elif claim.has_image():
#                 claim.modality = ModalityType.IMAGE_ONLY

#             evidence = EvidenceData(
#                 evidence_id=f"{claim.claim_id}_evidence",
#                 text=row.get("document", "").strip() or None,
#                 ocr=row.get("document_ocr", "").strip() or None,
#                 image_url=row.get("document_image", "").strip() or None,
#             )

#             evidence.image = download_image_fixed(
#                 evidence.image_url, cache_dir=cache_dir
#             )
#             if evidence.image and evidence.image.size != (448, 448):
#                 evidence.image = resize_image_for_paligemma(evidence.image, 448)

#             if evidence.has_text() and evidence.has_image():
#                 evidence.modality = ModalityType.TEXT_IMAGE
#             elif evidence.has_text():
#                 evidence.modality = ModalityType.TEXT_ONLY
#             elif evidence.has_image():
#                 evidence.modality = ModalityType.IMAGE_ONLY

#             pairs.append((claim, evidence))

#     return pairs


# def concatenate_images_horizontally_fixed(
#     img1: Image.Image,
#     img2: Image.Image,
#     target_size: int = 448,
# ) -> Optional[Image.Image]:
#     if img1 is None and img2 is None:
#         return None

#     if img1 is not None:
#         img1 = resize_image_for_paligemma(img1, target_size)
#     if img2 is not None:
#         img2 = resize_image_for_paligemma(img2, target_size)

#     if img1 is None:
#         return img2
#     if img2 is None:
#         return img1

#     try:
#         combined = Image.new("RGB", (target_size * 2, target_size))
#         combined.paste(img1, (0, 0))
#         combined.paste(img2, (target_size, 0))
#         combined_final = resize_image_for_paligemma(combined, target_size)
#         return combined_final
#     except Exception as e:
#         print(f"[ERROR] Image concatenation failed: {e}")
#         return img1


# def load_mocheg_data(
#     data_root: str,
#     split: str,
#     image_size: int,
#     cache_dir: str,
#     max_text_per_claim: int = 20,
# ) -> List[Tuple[ClaimData, EvidenceData]]:
#     """Load MOCHEG dataset format"""
#     from collections import defaultdict

#     split_dir = os.path.join(data_root, split)
#     corpus2_path = os.path.join(split_dir, "Corpus2.csv")
#     qrels_path = os.path.join(split_dir, "text_evidence_qrels_sentence_level.csv")
#     img_qrels_path = os.path.join(split_dir, "img_evidence_qrels.csv")
#     images_dir = os.path.join(split_dir, "images")
#     supp_path = os.path.join(data_root, "supplementary", "Corpus3_sentence_level.csv")

#     c2 = pd.read_csv(corpus2_path, dtype=str).fillna("")
#     claims_dict: Dict[str, str] = {}
#     for _, row in c2.iterrows():
#         cid = str(row.get("claim_id", "")).strip()
#         if cid:
#             claims_dict[cid] = str(row.get("Claim", "")).strip()

#     qrels = pd.read_csv(qrels_path, dtype=str).fillna("")
#     qrels = qrels[qrels["RELEVANCY"].astype(str) == "1"]
#     sents_df = pd.read_csv(supp_path, dtype=str).fillna("")
#     merged = qrels.merge(sents_df, left_on="DOCUMENT#", right_on="corpus_id", how="left")

#     text_evi: Dict[str, List[str]] = defaultdict(list)
#     for _, row in merged.iterrows():
#         cid = str(row.get("TOPIC", "")).strip()
#         sent = str(row.get("paragraph", "")).strip()
#         if cid and sent:
#             text_evi[cid].append(sent)

#     if max_text_per_claim:
#         for k in text_evi:
#             text_evi[k] = text_evi[k][:max_text_per_claim]

#     img_qrels = pd.read_csv(img_qrels_path, dtype=str).fillna("")
#     img_qrels = img_qrels[img_qrels["RELEVANCY"].astype(str) == "1"]

#     img_evi: Dict[str, List[str]] = defaultdict(list)
#     for _, row in img_qrels.iterrows():
#         cid = str(row.get("TOPIC", "")).strip()
#         fname = str(row.get("DOCUMENT#", "")).strip()
#         if cid and fname:
#             img_path = os.path.join(images_dir, fname)
#             if os.path.exists(img_path):
#                 img_evi[cid].append(img_path)

#     pairs: List[Tuple[ClaimData, EvidenceData]] = []
#     for cid in claims_dict:
#         claim = ClaimData(
#             claim_id=cid,
#             text=claims_dict[cid],
#             modality=ModalityType.TEXT_ONLY,
#         )

#         evidence = EvidenceData(
#             evidence_id=f"{cid}_evidence",
#             sentences=text_evi.get(cid, []),
#         )

#         if cid in img_evi and img_evi[cid]:
#             evidence.image = load_local_image(img_evi[cid][0], image_size)

#         if evidence.sentences and evidence.has_image():
#             evidence.modality = ModalityType.TEXT_IMAGE
#         elif evidence.sentences:
#             evidence.modality = ModalityType.TEXT_ONLY
#         elif evidence.has_image():
#             evidence.modality = ModalityType.IMAGE_ONLY

#         pairs.append((claim, evidence))

#     return pairs


# def load_mr2_data(
#     json_path: str, image_size: int, cache_dir: str
# ) -> List[Tuple[ClaimData, EvidenceData]]:
#     """
#     Load MR2 dataset (kept as in your version, but used via generic pipeline).
#     """
#     print(f"[INFO] Loading MR2 data from {json_path}")

#     with open(json_path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     json_dir = os.path.dirname(json_path)
#     mr2_dir = os.path.dirname(json_dir)
#     base_dir = os.path.abspath(os.getcwd())
#     print(f"[DEBUG] JSON dir: {json_dir}")
#     print(f"[DEBUG] MR2 dir: {mr2_dir}")
#     print(f"[DEBUG] Image base_dir: {base_dir}")

#     pairs: List[Tuple[ClaimData, EvidenceData]] = []
#     missing_claims = 0
#     missing_evidence = 0
#     total_claims = 0
#     total_evidence = 0
#     warn_logged = 0

#     example_checks = []

#     for idx, item in enumerate(data):
#         claim_text = item.get("claim_text", "").strip()
#         claim_id = str(item.get("id", idx))
#         claim_url = item.get("claim_image")

#         claim_image = None
#         full_claim_path = None
#         if claim_url:
#             if claim_url.startswith("mr2/"):
#                 full_claim_path = os.path.join(base_dir, claim_url)
#             else:
#                 full_claim_path = os.path.join(mr2_dir, claim_url)

#             total_claims += 1
#             if os.path.exists(full_claim_path):
#                 claim_image = load_local_image_fixed(full_claim_path, cache_dir)
#                 if claim_image is None and warn_logged < 10:
#                     print(
#                         f"[WARN] Claim image invalid after load: {claim_url} (full: {full_claim_path})"
#                     )
#                     warn_logged += 1
#             else:
#                 missing_claims += 1
#                 if warn_logged < 10:
#                     print(
#                         f"[WARN] Claim image missing: {claim_url} (full: {full_claim_path})"
#                     )
#                     warn_logged += 1

#         claim = ClaimData(
#             claim_id=claim_id,
#             text=claim_text,
#             image_url=claim_url,
#             image=claim_image,
#             modality=ModalityType.TEXT_IMAGE if claim_image else ModalityType.TEXT_ONLY,
#         )

#         evidence_texts = item.get("evidence_texts", [])
#         evidence_urls = item.get("evidence_images", [])

#         evidence_image = None
#         evidence_modality = ModalityType.TEXT_ONLY
#         full_ev_path = None
#         if evidence_urls:
#             for ev_url in evidence_urls:
#                 if ev_url.startswith("mr2/"):
#                     full_ev_path = os.path.join(base_dir, ev_url)
#                 else:
#                     full_ev_path = os.path.join(mr2_dir, ev_url)

#                 total_evidence += 1
#                 if os.path.exists(full_ev_path):
#                     temp_img = load_local_image_fixed(full_ev_path, cache_dir)
#                     if temp_img:
#                         evidence_image = temp_img
#                         evidence_modality = ModalityType.TEXT_IMAGE
#                         break
#                 else:
#                     missing_evidence += 1
#                     if warn_logged < 10:
#                         print(
#                             f"[WARN] Evidence image missing: {ev_url} (full: {full_ev_path})"
#                         )
#                         warn_logged += 1

#         evidence = EvidenceData(
#             evidence_id=claim_id,
#             sentences=evidence_texts,
#             image_url=evidence_urls[0] if evidence_urls else None,
#             image=evidence_image,
#             modality=evidence_modality,
#         )

#         pairs.append((claim, evidence))

#         if idx < 5:
#             checks = {
#                 "claim_id": claim_id,
#                 "claim_path_exists": bool(claim_image),
#                 "evidence_path_exists": bool(evidence_image),
#                 "claim_full_path": full_claim_path,
#                 "evidence_full_path": full_ev_path,
#             }
#             example_checks.append(checks)
#             if idx == 0:
#                 print(
#                     f"[DEBUG] Example full claim path: {checks.get('claim_full_path', 'N/A')}"
#                 )
#                 print(
#                     f"[DEBUG] Example full evidence path: {checks.get('evidence_full_path', 'N/A')}"
#                 )

#     if warn_logged >= 10:
#         print(f"[INFO] (Suppressed {warn_logged - 10} additional WARN messages)")

#     print(f"\n[MR2 DATASET STATS]")
#     print(f"Total items: {len(data)}")
#     print(f"Successfully loaded pairs: {len(pairs)}")
#     print(f"Valid claim images: {total_claims - missing_claims}")
#     print(f"Valid evidence images: {total_evidence - missing_evidence}")
#     if missing_claims > 0:
#         print(f"[WARN] Missing claim images: {missing_claims}/{total_claims}")
#     if missing_evidence > 0:
#         print(f"[WARN] Missing evidence images: {missing_evidence}/{total_evidence}")

#     print(f"\n[DEBUG] First 5 path checks:")
#     for check in example_checks:
#         print(
#             f"  ID {check['claim_id']}: Claim exists={check['claim_path_exists']}, Evidence exists={check['evidence_path_exists']}"
#         )

#     # quick size check
#     size_errors = []
#     for i, (claim, evidence) in enumerate(pairs):
#         if claim.image and claim.image.size != (image_size, image_size):
#             size_errors.append(f"Claim {claim.claim_id}: {claim.image.size}")
#         if evidence.image and evidence.image.size != (image_size, image_size):
#             size_errors.append(f"Evidence {evidence.evidence_id}: {evidence.image.size}")

#     if size_errors:
#         print(f"\n[INFO] Found {len(size_errors)} images with wrong size (fixed):")
#         for err in size_errors[:5]:
#             print(f"  - {err}")
#     else:
#         print(f"✅ All images are correctly sized ({image_size}x{image_size})")

#     return pairs


# def load_verite_data(
#     csv_path: str,
#     image_size: int,
#     cache_dir: str,
#     backend: str = "vllm",
# ) -> List[Tuple[ClaimData, EvidenceData]]:
#     """
#     Load VERITE dataset.

#     Mapping:
#       - caption column -> claim text
#       - image_path (or Image_path, image, etc.) -> evidence image (local path)

#     Each row becomes:
#       Claim: caption as text-only
#       Evidence: image-only (loaded from image_path)
#     """
#     print(f"[INFO] Loading VERITE data from {csv_path}")
#     df = pd.read_csv(csv_path)

#     col_map = {c.lower(): c for c in df.columns}
#     caption_col = col_map.get("caption")
#     if caption_col is None:
#         raise ValueError("VERITE CSV must contain a 'caption' column")

#     image_col = (
#         col_map.get("image_path")
#         or col_map.get("imagepath")
#         or col_map.get("image")
#         or col_map.get("img_path")
#     )
#     if image_col is None:
#         raise ValueError(
#             "VERITE CSV must contain an image path column "
#             "(e.g., 'image_path', 'image', 'img_path')"
#         )

#     label_col = col_map.get("label")  # optional: true / miscaptioned / out-of-context

#     base_dir = os.path.dirname(os.path.abspath(csv_path))

#     pairs: List[Tuple[ClaimData, EvidenceData]] = []

#     for idx, row in df.iterrows():
#         caption = str(row.get(caption_col, "") or "").strip()
#         if not caption:
#             continue

#         img_rel = str(row.get(image_col, "") or "").strip()
#         if not img_rel:
#             # Evidence image missing; we still create a claim but with no evidence
#             img_full = None
#         else:
#             img_full = img_rel
#             if not os.path.isabs(img_full):
#                 img_full = os.path.join(base_dir, img_rel)

#         # Load image depending on backend
#         evidence_img = None
#         if img_full and os.path.exists(img_full):
#             if backend == "hf":
#                 evidence_img = load_local_image_fixed(img_full, cache_dir)
#             else:
#                 evidence_img = load_local_image(img_full, image_size)
#         else:
#             if img_full:
#                 print(f"[WARN] VERITE image missing: {img_full}")

#         claim_id = str(row.get("id", idx))

#         claim = ClaimData(
#             claim_id=claim_id,
#             text=caption,
#             modality=ModalityType.TEXT_ONLY,
#         )

#         evidence = EvidenceData(
#             evidence_id=f"{claim_id}_evidence",
#             image=evidence_img,
#             image_url=img_rel if img_rel else None,
#         )

#         if evidence.has_image():
#             evidence.modality = ModalityType.IMAGE_ONLY
#         else:
#             evidence.modality = ModalityType.NONE

#         pairs.append((claim, evidence))

#     print(f"[INFO] Loaded {len(pairs)} VERITE claim-evidence pairs")
#     return pairs


# # ============================================================================
# # MAIN PIPELINE (GENERIC FOR ALL DATASETS, INCLUDING VERITE)
# # ============================================================================

# def run_classification(
#     data_path: str,
#     dataset_type: str,
#     model_key: str,
#     output_json: str,
#     cache_dir: str = "image_cache",
#     batch_size: int = 4,
#     quantize: Optional[str] = None,
#     checkpoint_interval: int = 100,
#     max_text_per_claim: int = 40,
# ):
#     """
#     Main classification pipeline for Factify / MOCHEG / MR2 / VERITE.
#     """
#     print(f"[INFO] Starting classification")
#     print(f"[INFO] Input: {data_path}")
#     print(f"[INFO] Dataset: {dataset_type}")
#     print(f"[INFO] Model: {model_key}")
#     print(f"[INFO] Output: {output_json}")

#     # -------------------------
#     # Load model
#     # -------------------------
#     print(f"\n[INFO] Initializing model...")
#     vlm = UnifiedVLM(model_key=model_key, quantize=quantize)
#     classifier = EvidenceClassifier(vlm)

#     backend = MODEL_CONFIG[model_key]["backend"]
#     if model_key == "paligemma":
#         image_size = 448
#     else:
#         image_size = MODEL_CONFIG[model_key].get("image_size", 768)

#     # -------------------------
#     # Load dataset
#     # -------------------------
#     print(f"\n[INFO] Loading dataset...")
#     if dataset_type == "factify":
#         pairs = load_factify_data_fixed(data_path, image_size, cache_dir)
#     elif dataset_type == "mocheg":
#         split = "val"
#         pairs = load_mocheg_data(
#             data_root=data_path,
#             split=split,
#             image_size=image_size,
#             cache_dir=cache_dir,
#             max_text_per_claim=max_text_per_claim,
#         )
#     elif dataset_type == "mr2":
#         pairs = load_mr2_data(data_path, image_size, cache_dir)
#     elif dataset_type == "verite":
#         pairs = load_verite_data(
#             data_path, image_size, cache_dir, backend=backend
#         )
#     else:
#         raise ValueError(f"Unknown dataset_type: {dataset_type}")

#     if not pairs:
#         print("[ERROR] No data loaded!")
#         return

#     print(f"[INFO] Loaded {len(pairs)} claim-evidence pairs")

#     # -------------------------
#     # Resume from checkpoint
#     # -------------------------
#     results: List[Dict[str, Any]] = []
#     start_idx = 0
#     if os.path.exists(output_json):
#         try:
#             with open(output_json, "r", encoding="utf-8") as f:
#                 results = json.load(f)
#             start_idx = len(results)
#             print(f"[INFO] Resuming from index {start_idx}")
#         except json.JSONDecodeError:
#             print("[WARN] Existing output_json is not valid JSON, starting fresh")
#             results = []
#             start_idx = 0

#     # -------------------------
#     # Process
#     # -------------------------
#     for i, (claim, evidence) in enumerate(
#         tqdm(pairs[start_idx:], desc="Classifying")
#     ):
#         idx = start_idx + i

#         try:
#             result = classifier.classify_combined(claim, evidence, batch_size)

#             result.update(
#                 {
#                     "claim_text": claim.get_full_text(),
#                     "claim_image_url": claim.image_url,
#                     "evidence_image_url": evidence.image_url,
#                     "dataset": dataset_type,
#                 }
#             )

#             results.append(result)

#             if (idx + 1) % checkpoint_interval == 0:
#                 os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
#                 with open(output_json, "w", encoding="utf-8") as f:
#                     json.dump(results, f, indent=2, ensure_ascii=False)
#                 print(f"[CHECKPOINT] Saved {idx + 1} results")

#         except Exception as e:
#             print(f"[ERROR] Failed on claim {claim.claim_id}: {e}")
#             import traceback

#             traceback.print_exc()
#             results.append(
#                 {
#                     "claim_id": claim.claim_id,
#                     "error": str(e),
#                     "skipped": True,
#                 }
#             )

#     # -------------------------
#     # Final save
#     # -------------------------
#     os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
#     with open(output_json, "w", encoding="utf-8") as f:
#         json.dump(results, f, indent=2, ensure_ascii=False)

#     print(f"\n[DONE] Saved {len(results)} results to {output_json}")

#     # -------------------------
#     # Stats
#     # -------------------------
#     print("\n[STATISTICS]")
#     successful = len([r for r in results if not r.get("skipped")])
#     print(f"Successful: {successful}/{len(results)}")

#     modality_stats: Dict[str, int] = {}
#     for result in results:
#         if not result.get("skipped"):
#             key = f"{result['claim_modality']}_{result['evidence_modality']}"
#             modality_stats[key] = modality_stats.get(key, 0) + 1

#     print("\nModality combinations:")
#     for combo, count in sorted(modality_stats.items()):
#         print(f"  {combo}: {count}")


# # ============================================================================
# # CLI
# # ============================================================================

# def main():
#     parser = argparse.ArgumentParser(
#         description="Unified Multimodal Evidence Classification (Hybrid: vLLM + HF)"
#     )
#     parser.add_argument(
#         "--data_path",
#         required=True,
#         help="Path to data (CSV for Factify/VERITE, root dir for MOCHEG, JSON for MR2)",
#     )
#     parser.add_argument(
#         "--dataset_type",
#         required=True,
#         choices=["factify", "mocheg", "mr2", "verite"],
#         help="Dataset format type",
#     )
#     parser.add_argument(
#         "--model_key",
#         required=True,
#         choices=list(MODEL_CONFIG.keys()),
#         help="Model to use (paligemma=HF, others=vLLM)",
#     )
#     parser.add_argument(
#         "--output_json",
#         required=True,
#         help="Output JSON file path",
#     )
#     parser.add_argument(
#         "--cache_dir",
#         default="image_cache",
#         help="Directory for caching downloaded images",
#     )
#     parser.add_argument(
#         "--batch_size",
#         type=int,
#         default=4,
#         help="Batch size for text evidence classification (lower for HF models)",
#     )
#     parser.add_argument(
#         "--quantize",
#         type=str,
#         default=None,
#         choices=["awq", "gptq"],
#         help="Manual quantization for vLLM models (ignored for PaliGemma)",
#     )
#     parser.add_argument(
#         "--checkpoint_interval",
#         type=int,
#         default=100,
#         help="Save checkpoint every N samples",
#     )
#     parser.add_argument(
#         "--max_text_per_claim",
#         type=int,
#         default=40,
#         help="Maximum text evidence sentences per claim (for MOCHEG)",
#     )

#     args = parser.parse_args()

#     if args.model_key == "paligemma" and args.quantize:
#         print("[WARN] --quantize ignored for PaliGemma (uses HF 4-bit)")

#     run_classification(
#         data_path=args.data_path,
#         dataset_type=args.dataset_type,
#         model_key=args.model_key,
#         output_json=args.output_json,
#         cache_dir=args.cache_dir,
#         batch_size=args.batch_size,
#         quantize=args.quantize,
#         checkpoint_interval=args.checkpoint_interval,
#         max_text_per_claim=args.max_text_per_claim,
#     )


# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Multimodal Evidence Classification System
Hybrid: vLLM for Qwen2VL/Idefics3, HF Transformers for PaliGemma

Supported Modality Combinations:
CLAIM: text + image, text only, image only
EVIDENCE: text + image, text only, image only, none

Supported Models:
- PaliGemma (HF, 4-bit)
- Qwen2VL (vLLM)
- Idefics3 (vLLM)

Supported Datasets:
- Factify
- MOCHEG
- MR2
- VERITE  <-- added
"""

import os
import re
import csv
import json
import argparse
import hashlib
import sys
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import requests
from PIL import Image
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd

from transformers import AutoProcessor
# vLLM imports
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("[WARN] vLLM not available")

# HF Transformers imports for PaliGemma
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    BitsAndBytesConfig,
)

csv.field_size_limit(sys.maxsize)


# ============================================================================
# CONFIGURATION & ENUMS
# ============================================================================

class ModalityType(Enum):
    """Supported modality combinations"""
    TEXT_IMAGE = "text_image"
    TEXT_ONLY = "text_only"
    IMAGE_ONLY = "image_only"
    NONE = "none"


@dataclass
class ClaimData:
    """Structured claim representation"""
    claim_id: str
    text: Optional[str] = None
    ocr: Optional[str] = None
    image_url: Optional[str] = None
    image: Optional[Image.Image] = None
    modality: ModalityType = ModalityType.NONE

    def get_full_text(self) -> str:
        """Combine text and OCR"""
        parts = []
        if self.text:
            parts.append(self.text.strip())
        if self.ocr:
            parts.append(self.ocr.strip())
        return " ".join(parts)

    def has_text(self) -> bool:
        return bool(self.text or self.ocr)

    def has_image(self) -> bool:
        return self.image is not None


@dataclass
class EvidenceData:
    """Structured evidence representation"""
    evidence_id: str
    text: Optional[str] = None
    ocr: Optional[str] = None
    image_url: Optional[str] = None
    image: Optional[Image.Image] = None
    sentences: Optional[List[str]] = None
    modality: ModalityType = ModalityType.NONE

    def get_full_text(self) -> str:
        """Combine text and OCR"""
        parts = []
        if self.text:
            parts.append(self.text.strip())
        if self.ocr:
            parts.append(self.ocr.strip())
        return " ".join(parts)

    def has_text(self) -> bool:
        return bool(self.text or self.ocr or self.sentences)

    def has_image(self) -> bool:
        return self.image is not None


MODEL_CONFIG = {
    # PaliGemma - Uses HF Transformers (4-bit quantized)
    "paligemma": {
        "model_id": "google/paligemma2-3b-pt-448",
        "backend": "hf",  # Use Hugging Face Transformers
        "max_new_tokens": 256,
        "image_size": 448,
        "low_res_size": 224,
        "max_model_len": 8192,
        "use_chat_template": True,
        "quantization": "4bit",  # 4-bit quantization
    },

    # Qwen2VL - Uses vLLM
    "qwen2vl": {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "backend": "vllm",
        "max_new_tokens": 256,
        "image_size": 768,
        "max_model_len": 10000,
        "visual_token_count": 256,
        "use_chat_template": True,
    },

    # Idefics3 - Uses vLLM
    "idefics3": {
        "model_id": "leon-se/Idefics3-8B-Llama3-bnb_nf4",  # or HuggingFaceM4/Idefics3-8B-Llama3
        "backend": "vllm",
        "max_new_tokens": 256,
        "image_size": 768,
        "max_model_len": 4096,  # increased from 2048 to avoid prompt length errors
        "visual_token_count": 256,
        "use_chat_template": True,
    },

    # AWQ Quantized models (vLLM)
    "qwen2vl-awq": {
        "model_id": "Qwen/Qwen2-VL-7B-Instruct-AWQ",
        "backend": "vllm",
        "max_new_tokens": 256,
        "image_size": 768,
        "max_model_len": 8192,
        "visual_token_count": 256,
        "use_chat_template": True,
        "quantization": "awq",
        "dtype": "half",
    },
    "idefics3-awq": {
        "model_id": "ronantakizawa/idefics3-8b-llama3-awq",
        "backend": "vllm",
        "max_new_tokens": 256,
        "image_size": 768,
        "max_model_len": 8192,
        "visual_token_count": 256,
        "use_chat_template": True,
        "quantization": "awq",
        "dtype": "half",
    },
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def is_valid_image(img: Optional[Image.Image]) -> bool:
    """Validate image before processing"""
    try:
        if img is None:
            return False
        arr = np.array(img)
        if len(arr.shape) != 3 or arr.shape[2] != 3:
            return False
        h, w, _ = arr.shape
        if h < 64 or w < 64 or h > 4096 or w > 4096:
            return False
        if np.std(arr) < 2:
            return False
        if np.any(np.isnan(arr)):
            return False
        return True
    except Exception:
        return False


def resize_image_for_paligemma(img: Image.Image, target_size: int = 448) -> Optional[Image.Image]:
    """
    Fixed image resizing for PaliGemma

    PaliGemma2 supports: 224, 448, or 896
    Default: 448 for paligemma2-3b-pt-448
    """
    if img is None:
        return None
    try:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_resized = img.resize((target_size, target_size), Image.LANCZOS)
        if not is_valid_image(img_resized):
            print(f"[WARN] Invalid image after resize: {img_resized.size}")
            return None
        return img_resized
    except Exception as e:
        print(f"[ERROR] Image resize failed: {e}")
        return None


def download_image_fixed(url: str, max_size: int = 768, cache_dir: str = "image_cache") -> Optional[Image.Image]:
    """
    Download and resize for PaliGemma (always to 448x448).
    """
    if not url or str(url).strip().lower() in {"", "nan", "none"}:
        return None

    os.makedirs(cache_dir, exist_ok=True)
    url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()
    ext = os.path.splitext(url)[-1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        ext = ".jpg"

    local_path = os.path.join(cache_dir, f"{url_hash}_pali448{ext}")

    if os.path.exists(local_path):
        try:
            img = Image.open(local_path).convert("RGB")
            return img
        except Exception:
            os.remove(local_path)

    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        img_resized = resize_image_for_paligemma(img, target_size=448)
        if img_resized:
            img_resized.save(local_path)
            return img_resized
        return None
    except Exception as e:
        print(f"[WARN] Image download failed: {url} ({e})")
        return None


def load_local_image_fixed(path: str, max_size: int = 768) -> Optional[Image.Image]:
    """
    Load local image and resize for PaliGemma (448x448).
    """
    if not path or not os.path.exists(path):
        return None
    try:
        img = Image.open(path).convert("RGB")
        img_resized = resize_image_for_paligemma(img, target_size=448)
        return img_resized
    except Exception as e:
        print(f"[WARN] Image load failed: {path} ({e})")
        return None


def load_local_image(path: str, max_size: int = 768) -> Optional[Image.Image]:
    """
    Generic local image loader (for vLLM models like Qwen2VL / Idefics3).
    """
    if not path or not os.path.exists(path):
        return None
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((max_size, max_size))
        return img
    except Exception as e:
        print(f"[WARN] Generic image load failed: {path} ({e})")
        return None


_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9""\(\[])')


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences"""
    text = (text or "").strip()
    if not text:
        return []
    chunks = _SENT_SPLIT_RE.split(text)
    sents = [re.sub(r"\s+", " ", c).strip() for c in chunks]
    return [s for s in sents if len(s) > 2]


def truncate_text(processor, text: str, max_tokens: int) -> str:
    """Truncate text to max tokens"""
    if not text:
        return text
    tokenizer = getattr(processor, "tokenizer", processor)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    truncated = tokens[:max_tokens]
    return tokenizer.decode(truncated, skip_special_tokens=True)


_LABEL_RE = re.compile(r"(SUPPORT|REFUTE)", re.IGNORECASE)


def parse_label(response: str) -> str:
    """Extract SUPPORT/REFUTE from response"""
    m = _LABEL_RE.search(response or "")
    if not m:
        return "REFUTE"
    return m.group(1).upper()


# ============================================================================
# PROMPT GENERATION (MODALITY-AWARE)
# ============================================================================

class PromptBuilder:
    """Build prompts based on available modalities"""

    @staticmethod
    def build_text_evidence_prompt(
        claim: ClaimData,
        evidence_sentence: str,
        processor,
        max_tokens: int = 512,  # reduced to avoid context overflow
    ) -> str:
        """Prompt for text evidence classification"""
        claim_text = truncate_text(processor, claim.get_full_text(), max_tokens // 3)
        evidence_sentence = truncate_text(processor, evidence_sentence, max_tokens // 3)

        context_parts = []
        if claim.has_image():
            context_parts.append("An image of the claim is provided for visual context.")
        if claim.has_text():
            context_parts.append(f"CLAIM TEXT: {claim_text}")

        context = "\n".join(context_parts) if context_parts else "CLAIM: [Visual only - see image]"

        return (
            "You are a fact-checking assistant that classifies textual evidence.\n\n"
            "### Task:\n"
            "Determine whether the evidence sentence SUPPORTS or REFUTES the given claim.\n\n"
            "### Instructions:\n"
            "- Analyze the textual content and its logical relationship to the claim.\n"
            "- If a claim image is provided, consider visual information as well.\n"
            "- Focus on factual consistency between evidence and claim.\n\n"
            f"{context}\n\n"
            f"**EVIDENCE SENTENCE:** {evidence_sentence}\n\n"
            "### Response:\n"
            "Reply with exactly one word: SUPPORT or REFUTE\n"
        )

    @staticmethod
    def build_visual_evidence_prompt(
        claim: ClaimData,
        evidence: EvidenceData,
        processor,
        max_tokens: int = 512,  # reduced to avoid context overflow
    ) -> str:
        """Prompt for visual evidence classification"""
        claim_text = truncate_text(processor, claim.get_full_text(), max_tokens // 3)
        evidence_text = truncate_text(processor, evidence.get_full_text(), max_tokens // 3)

        context_parts = []
        if claim.has_text():
            context_parts.append(f"CLAIM TEXT: {claim_text}")
        else:
            context_parts.append("CLAIM: [Visual only - see claim image]")

        if evidence.has_text():
            context_parts.append(f"EVIDENCE TEXT/OCR: {evidence_text}")

        context = "\n".join(context_parts)

        return (
            "You are a multimodal fact-checking assistant that analyzes visual evidence.\n\n"
            "### Task:\n"
            "Determine whether the evidence image SUPPORTS or REFUTES the given claim.\n\n"
            "### Instructions:\n"
            "- Analyze visual content: objects, scenes, text, layout, style.\n"
            "- Compare evidence image with the claim (text and/or image).\n"
            "- If evidence text/OCR is provided, use it to aid analysis.\n"
            "- Consider context and factual consistency.\n\n"
            f"{context}\n\n"
            "### Response:\n"
            "Reply with exactly one word: SUPPORT or REFUTE\n"
        )


# ============================================================================
# MODEL WRAPPERS
# ============================================================================

class HFPaliGemmaWrapper:
    """Hugging Face PaliGemma wrapper with 4-bit quantization"""

    def __init__(self, model_id: str, max_new_tokens: int = 256):
        print(f"[INFO] Loading PaliGemma with HF Transformers (4-bit): {model_id}")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        self.processor = PaliGemmaProcessor.from_pretrained(
            model_id, trust_remote_code=True
        )

        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        # Set pad_token to eos_token to avoid generation warnings
        self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.model.generation_config.pad_token_id = self.processor.tokenizer.eos_token_id

        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.device = next(self.model.parameters()).device

        print(f"[INFO] PaliGemma loaded on device: {self.device}")

    def generate(self, prompts: List[str], images_lists: List[List[Image.Image]]) -> List[str]:
        responses: List[str] = []

        for prompt, images in tqdm(
            list(zip(prompts, images_lists)),
            total=len(prompts),
            desc="PaliGemma (HF)",
            leave=False,
        ):
            try:
                valid_images = [img for img in images if is_valid_image(img)]

                # PaliGemma requires images — for text-only we use a white placeholder
                if not valid_images:
                    blank_img = Image.new("RGB", (448, 448), color=(255, 255, 255))
                    valid_images = [blank_img]
                    print("[INFO] Text-only input, using blank placeholder image")

                image_tokens = "<image>" * len(valid_images)
                full_prompt = image_tokens + prompt

                inputs = self.processor(
                    text=full_prompt,
                    images=valid_images,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.processor.tokenizer.eos_token_id,
                        repetition_penalty=1.1,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                    )

                generated_ids = outputs[0][inputs.input_ids.shape[1]:]
                response = self.processor.decode(
                    generated_ids, skip_special_tokens=True
                ).strip()

                if not response or len(response) < 3:
                    response = "REFUTE"

                responses.append(response)

            except Exception as e:
                print(f"[ERROR] PaliGemma generation failed: {e}")
                import traceback
                traceback.print_exc()
                responses.append("REFUTE")

        return responses


class VLLMWrapper:
    """vLLM wrapper for Qwen2VL and Idefics3"""

    def __init__(self, model_key: str, config: dict, quantize: Optional[str] = None):
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM not available but required for this model")

        print(f"[INFO] Loading {model_key} with vLLM: {config['model_id']}")

        self.model_key = model_key
        self.config = config
        self.processor = AutoProcessor.from_pretrained(
            config["model_id"], trust_remote_code=True
        )

        llm_kwargs = {
            "model": config["model_id"],
            "max_model_len": config["max_model_len"],
            "trust_remote_code": True,
        }

        if "quantization" in config:
            llm_kwargs["quantization"] = config["quantization"]
            llm_kwargs["dtype"] = config.get("dtype", "half")
            llm_kwargs["gpu_memory_utilization"] = 0.7
            print(f"[INFO] Using {config['quantization']} quantization")
        elif quantize:
            llm_kwargs["quantization"] = quantize
            llm_kwargs["dtype"] = "half"
            llm_kwargs["gpu_memory_utilization"] = 0.7
            print(f"[INFO] Applying {quantize} quantization")
        else:
            llm_kwargs["dtype"] = "bfloat16"
            llm_kwargs["gpu_memory_utilization"] = 0.9

        llm_kwargs["limit_mm_per_prompt"] = {"image": 2}

        try:
            self.llm = LLM(**llm_kwargs)
        except Exception as e:
            print(f"[ERROR] Failed to load: {e}")
            print("[INFO] Retrying with reduced memory...")
            llm_kwargs["gpu_memory_utilization"] = 0.5
            self.llm = LLM(**llm_kwargs)

        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=config["max_new_tokens"],
            skip_special_tokens=True,
        )

    def _format_prompt(self, prompt: str, images: List[Image.Image]) -> str:
        """Format prompt with chat template"""
        valid_imgs = [img for img in images if is_valid_image(img)]

        content = [{"type": "image"} for _ in valid_imgs]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        return self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

    def generate(self, prompts: List[str], images_lists: List[List[Image.Image]]) -> List[str]:
        requests = []
        for prompt, imgs in zip(prompts, images_lists):
            valid_imgs = [img for img in imgs if is_valid_image(img)]
            formatted_prompt = self._format_prompt(prompt, valid_imgs)

            requests.append(
                {
                    "prompt": formatted_prompt,
                    "multi_modal_data": {"image": valid_imgs} if valid_imgs else {},
                }
            )

        try:
            outputs = self.llm.generate(requests, self.sampling_params)
            return [out.outputs[0].text.strip() for out in outputs]
        except Exception as e:
            print(f"[ERROR] vLLM generation failed: {e}")
            return ["ERROR"] * len(prompts)


class UnifiedVLM:
    """Unified wrapper that routes to HF or vLLM based on model"""

    def __init__(self, model_key: str, quantize: Optional[str] = None):
        assert model_key in MODEL_CONFIG, f"Unknown model: {model_key}"

        self.model_key = model_key
        self.config = MODEL_CONFIG[model_key]
        self.backend = self.config["backend"]

        if self.backend == "hf":
            self.model = HFPaliGemmaWrapper(
                self.config["model_id"], self.config["max_new_tokens"]
            )
            self.processor = self.model.processor
        elif self.backend == "vllm":
            self.model = VLLMWrapper(model_key, self.config, quantize)
            self.processor = self.model.processor
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def classify_batch(
        self,
        prompts: List[str],
        images_lists: List[List[Image.Image]],
    ) -> List[str]:
        return self.model.generate(prompts, images_lists)


# ============================================================================
# EVIDENCE CLASSIFIER
# ============================================================================

class EvidenceClassifier:
    """Classify evidence based on available modalities"""

    def __init__(self, vlm: UnifiedVLM):
        self.vlm = vlm
        self.prompt_builder = PromptBuilder()

    def classify_text_evidence(
        self,
        claim: ClaimData,
        sentences: List[str],
        batch_size: int = 8,
    ) -> List[Dict[str, Any]]:
        if not sentences:
            return []

        predictions: List[Dict[str, Any]] = []

        for i in range(0, len(sentences), batch_size):
            batch_sents = sentences[i: i + batch_size]

            prompts = [
                self.prompt_builder.build_text_evidence_prompt(
                    claim, sent, self.vlm.processor
                )
                for sent in batch_sents
            ]

            images_lists = [
                [claim.image] if claim.has_image() else [] for _ in batch_sents
            ]

            responses = self.vlm.classify_batch(prompts, images_lists)

            for sent, resp in zip(batch_sents, responses):
                predictions.append(
                    {
                        "sentence": sent,
                        "prediction": parse_label(resp),
                        "raw_response": resp,
                    }
                )

        return predictions

    def classify_visual_evidence(
        self,
        claim: ClaimData,
        evidence: EvidenceData,
    ) -> Tuple[str, str]:
        if not evidence.has_image():
            return "NO_EVIDENCE", "No visual evidence available"

        prompt = self.prompt_builder.build_visual_evidence_prompt(
            claim, evidence, self.vlm.processor
        )

        images: List[Image.Image] = []
        if claim.has_image():
            images.append(claim.image)
        images.append(evidence.image)

        try:
            responses = self.vlm.classify_batch([prompt], [images])
            response = responses[0]
            label = parse_label(response)
            return label, response
        except Exception as e:
            return "ERROR", f"Classification failed: {e}"

    def classify_combined(
        self,
        claim: ClaimData,
        evidence: EvidenceData,
        batch_size: int = 8,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "claim_id": claim.claim_id,
            "claim_modality": claim.modality.value,
            "evidence_modality": evidence.modality.value,
            "supporting_text": [],
            "refuting_text": [],
            "text_predictions": [],
            "visual_label": None,
            "visual_response": None,
        }

        # ---- Text evidence ----
        if evidence.has_text():
            if not evidence.sentences:
                evidence.sentences = split_into_sentences(evidence.get_full_text())

            text_preds = self.classify_text_evidence(
                claim, evidence.sentences, batch_size
            )

            result["text_predictions"] = text_preds
            result["supporting_text"] = [
                p["sentence"] for p in text_preds if p["prediction"] == "SUPPORT"
            ]
            result["refuting_text"] = [
                p["sentence"] for p in text_preds if p["prediction"] == "REFUTE"
            ]

        # ---- Visual evidence ----
        if evidence.has_image():
            label, response = self.classify_visual_evidence(claim, evidence)
            result["visual_label"] = label

            # Normalize visual_response to just SUPPORT. / REFUTE.
            if label in ("SUPPORT", "REFUTE"):
                result["visual_response"] = f"{label}."
            else:
                # Fallback if parsing fails
                result["visual_response"] = "REFUTE."

            # Keep the raw model output in a separate field
            result["visual_raw_response"] = response

        return result


# ============================================================================
# DATA LOADERS
# ============================================================================

def load_factify_data_fixed(csv_path: str, image_size: int, cache_dir: str):
    """
    Factify loader (kept for completeness).
    Uses PaliGemma-style image preprocessing.
    """
    pairs: List[Tuple[ClaimData, EvidenceData]] = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            claim = ClaimData(
                claim_id=str(row.get("Id", "")).strip(),
                text=row.get("claim", "").strip() or None,
                ocr=row.get("claim_ocr", "").strip() or None,
                image_url=row.get("claim_image", "").strip() or None,
            )

            claim.image = download_image_fixed(claim.image_url, cache_dir=cache_dir)
            if claim.image and claim.image.size != (448, 448):
                claim.image = resize_image_for_paligemma(claim.image, 448)

            if claim.has_text() and claim.has_image():
                claim.modality = ModalityType.TEXT_IMAGE
            elif claim.has_text():
                claim.modality = ModalityType.TEXT_ONLY
            elif claim.has_image():
                claim.modality = ModalityType.IMAGE_ONLY

            evidence = EvidenceData(
                evidence_id=f"{claim.claim_id}_evidence",
                text=row.get("document", "").strip() or None,
                ocr=row.get("document_ocr", "").strip() or None,
                image_url=row.get("document_image", "").strip() or None,
            )

            evidence.image = download_image_fixed(
                evidence.image_url, cache_dir=cache_dir
            )
            if evidence.image and evidence.image.size != (448, 448):
                evidence.image = resize_image_for_paligemma(evidence.image, 448)

            if evidence.has_text() and evidence.has_image():
                evidence.modality = ModalityType.TEXT_IMAGE
            elif evidence.has_text():
                evidence.modality = ModalityType.TEXT_ONLY
            elif evidence.has_image():
                evidence.modality = ModalityType.IMAGE_ONLY

            pairs.append((claim, evidence))

    return pairs


def concatenate_images_horizontally_fixed(
    img1: Image.Image,
    img2: Image.Image,
    target_size: int = 448,
) -> Optional[Image.Image]:
    if img1 is None and img2 is None:
        return None

    if img1 is not None:
        img1 = resize_image_for_paligemma(img1, target_size)
    if img2 is not None:
        img2 = resize_image_for_paligemma(img2, target_size)

    if img1 is None:
        return img2
    if img2 is None:
        return img1

    try:
        combined = Image.new("RGB", (target_size * 2, target_size))
        combined.paste(img1, (0, 0))
        combined.paste(img2, (target_size, 0))
        combined_final = resize_image_for_paligemma(combined, target_size)
        return combined_final
    except Exception as e:
        print(f"[ERROR] Image concatenation failed: {e}")
        return img1


def load_mocheg_data(
    data_root: str,
    split: str,
    image_size: int,
    cache_dir: str,
    max_text_per_claim: int = 20,
) -> List[Tuple[ClaimData, EvidenceData]]:
    """Load MOCHEG dataset format"""
    from collections import defaultdict

    split_dir = os.path.join(data_root, split)
    corpus2_path = os.path.join(split_dir, "Corpus2.csv")
    qrels_path = os.path.join(split_dir, "text_evidence_qrels_sentence_level.csv")
    img_qrels_path = os.path.join(split_dir, "img_evidence_qrels.csv")
    images_dir = os.path.join(split_dir, "images")
    supp_path = os.path.join(data_root, "supplementary", "Corpus3_sentence_level.csv")

    c2 = pd.read_csv(corpus2_path, dtype=str).fillna("")
    claims_dict: Dict[str, str] = {}
    for _, row in c2.iterrows():
        cid = str(row.get("claim_id", "")).strip()
        if cid:
            claims_dict[cid] = str(row.get("Claim", "")).strip()

    qrels = pd.read_csv(qrels_path, dtype=str).fillna("")
    qrels = qrels[qrels["RELEVANCY"].astype(str) == "1"]
    sents_df = pd.read_csv(supp_path, dtype=str).fillna("")
    merged = qrels.merge(sents_df, left_on="DOCUMENT#", right_on="corpus_id", how="left")

    text_evi: Dict[str, List[str]] = defaultdict(list)
    for _, row in merged.iterrows():
        cid = str(row.get("TOPIC", "")).strip()
        sent = str(row.get("paragraph", "")).strip()
        if cid and sent:
            text_evi[cid].append(sent)

    if max_text_per_claim:
        for k in text_evi:
            text_evi[k] = text_evi[k][:max_text_per_claim]

    img_qrels = pd.read_csv(img_qrels_path, dtype=str).fillna("")
    img_qrels = img_qrels[img_qrels["RELEVANCY"].astype(str) == "1"]

    img_evi: Dict[str, List[str]] = defaultdict(list)
    for _, row in img_qrels.iterrows():
        cid = str(row.get("TOPIC", "")).strip()
        fname = str(row.get("DOCUMENT#", "")).strip()
        if cid and fname:
            img_path = os.path.join(images_dir, fname)
            if os.path.exists(img_path):
                img_evi[cid].append(img_path)

    pairs: List[Tuple[ClaimData, EvidenceData]] = []
    for cid in claims_dict:
        claim = ClaimData(
            claim_id=cid,
            text=claims_dict[cid],
            modality=ModalityType.TEXT_ONLY,
        )

        evidence = EvidenceData(
            evidence_id=f"{cid}_evidence",
            sentences=text_evi.get(cid, []),
        )

        if cid in img_evi and img_evi[cid]:
            evidence.image = load_local_image(img_evi[cid][0], image_size)

        if evidence.sentences and evidence.has_image():
            evidence.modality = ModalityType.TEXT_IMAGE
        elif evidence.sentences:
            evidence.modality = ModalityType.TEXT_ONLY
        elif evidence.has_image():
            evidence.modality = ModalityType.IMAGE_ONLY

        pairs.append((claim, evidence))

    return pairs


def load_mr2_data(
    json_path: str, image_size: int, cache_dir: str
) -> List[Tuple[ClaimData, EvidenceData]]:
    """
    Load MR2 dataset (kept as in your version, but used via generic pipeline).
    """
    print(f"[INFO] Loading MR2 data from {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    json_dir = os.path.dirname(json_path)
    mr2_dir = os.path.dirname(json_dir)
    base_dir = os.path.abspath(os.getcwd())
    print(f"[DEBUG] JSON dir: {json_dir}")
    print(f"[DEBUG] MR2 dir: {mr2_dir}")
    print(f"[DEBUG] Image base_dir: {base_dir}")

    pairs: List[Tuple[ClaimData, EvidenceData]] = []
    missing_claims = 0
    missing_evidence = 0
    total_claims = 0
    total_evidence = 0
    warn_logged = 0

    example_checks = []

    for idx, item in enumerate(data):
        claim_text = item.get("claim_text", "").strip()
        claim_id = str(item.get("id", idx))
        claim_url = item.get("claim_image")

        claim_image = None
        full_claim_path = None
        if claim_url:
            if claim_url.startswith("mr2/"):
                full_claim_path = os.path.join(base_dir, claim_url)
            else:
                full_claim_path = os.path.join(mr2_dir, claim_url)

            total_claims += 1
            if os.path.exists(full_claim_path):
                claim_image = load_local_image_fixed(full_claim_path, cache_dir)
                if claim_image is None and warn_logged < 10:
                    print(
                        f"[WARN] Claim image invalid after load: {claim_url} (full: {full_claim_path})"
                    )
                    warn_logged += 1
            else:
                missing_claims += 1
                if warn_logged < 10:
                    print(
                        f"[WARN] Claim image missing: {claim_url} (full: {full_claim_path})"
                    )
                    warn_logged += 1

        claim = ClaimData(
            claim_id=claim_id,
            text=claim_text,
            image_url=claim_url,
            image=claim_image,
            modality=ModalityType.TEXT_IMAGE if claim_image else ModalityType.TEXT_ONLY,
        )

        evidence_texts = item.get("evidence_texts", [])
        evidence_urls = item.get("evidence_images", [])

        evidence_image = None
        evidence_modality = ModalityType.TEXT_ONLY
        full_ev_path = None
        if evidence_urls:
            for ev_url in evidence_urls:
                if ev_url.startswith("mr2/"):
                    full_ev_path = os.path.join(base_dir, ev_url)
                else:
                    full_ev_path = os.path.join(mr2_dir, ev_url)

                total_evidence += 1
                if os.path.exists(full_ev_path):
                    temp_img = load_local_image_fixed(full_ev_path, cache_dir)
                    if temp_img:
                        evidence_image = temp_img
                        evidence_modality = ModalityType.TEXT_IMAGE
                        break
                else:
                    missing_evidence += 1
                    if warn_logged < 10:
                        print(
                            f"[WARN] Evidence image missing: {ev_url} (full: {full_ev_path})"
                        )
                        warn_logged += 1

        evidence = EvidenceData(
            evidence_id=claim_id,
            sentences=evidence_texts,
            image_url=evidence_urls[0] if evidence_urls else None,
            image=evidence_image,
            modality=evidence_modality,
        )

        pairs.append((claim, evidence))

        if idx < 5:
            checks = {
                "claim_id": claim_id,
                "claim_path_exists": bool(claim_image),
                "evidence_path_exists": bool(evidence_image),
                "claim_full_path": full_claim_path,
                "evidence_full_path": full_ev_path,
            }
            example_checks.append(checks)
            if idx == 0:
                print(
                    f"[DEBUG] Example full claim path: {checks.get('claim_full_path', 'N/A')}"
                )
                print(
                    f"[DEBUG] Example full evidence path: {checks.get('evidence_full_path', 'N/A')}"
                )

    if warn_logged >= 10:
        print(f"[INFO] (Suppressed {warn_logged - 10} additional WARN messages)")

    print(f"\n[MR2 DATASET STATS]")
    print(f"Total items: {len(data)}")
    print(f"Successfully loaded pairs: {len(pairs)}")
    print(f"Valid claim images: {total_claims - missing_claims}")
    print(f"Valid evidence images: {total_evidence - missing_evidence}")
    if missing_claims > 0:
        print(f"[WARN] Missing claim images: {missing_claims}/{total_claims}")
    if missing_evidence > 0:
        print(f"[WARN] Missing evidence images: {missing_evidence}/{total_evidence}")

    print(f"\n[DEBUG] First 5 path checks:")
    for check in example_checks:
        print(
            f"  ID {check['claim_id']}: Claim exists={check['claim_path_exists']}, Evidence exists={check['evidence_path_exists']}"
        )

    # quick size check
    size_errors = []
    for i, (claim, evidence) in enumerate(pairs):
        if claim.image and claim.image.size != (image_size, image_size):
            size_errors.append(f"Claim {claim.claim_id}: {claim.image.size}")
        if evidence.image and evidence.image.size != (image_size, image_size):
            size_errors.append(f"Evidence {evidence.evidence_id}: {evidence.image.size}")

    if size_errors:
        print(f"\n[INFO] Found {len(size_errors)} images with wrong size (fixed):")
        for err in size_errors[:5]:
            print(f"  - {err}")
    else:
        print(f"✅ All images are correctly sized ({image_size}x{image_size})")

    return pairs


def load_verite_data(
    csv_path: str,
    image_size: int,
    cache_dir: str,
    backend: str = "vllm",
) -> List[Tuple[ClaimData, EvidenceData, Dict[str, Any]]]:
    """
    Load VERITE dataset.

    Mapping:
      - caption column -> claim text
      - image_path (or Image_path, image, etc.) -> evidence image (local path)

    Each row becomes:
      Claim: caption as text-only
      Evidence: image-only (loaded from image_path)

    Returns triples: (claim, evidence, meta) where meta contains:
      - dataset_type
      - dataset_label
      - image_rel_path
      - image_full_path
      - row_index
    """
    print(f"[INFO] Loading VERITE data from {csv_path}")
    df = pd.read_csv(csv_path)

    col_map = {c.lower(): c for c in df.columns}
    caption_col = col_map.get("caption")
    if caption_col is None:
        raise ValueError("VERITE CSV must contain a 'caption' column")

    image_col = (
        col_map.get("image_path")
        or col_map.get("imagepath")
        or col_map.get("image")
        or col_map.get("img_path")
    )
    if image_col is None:
        raise ValueError(
            "VERITE CSV must contain an image path column "
            "(e.g., 'image_path', 'image', 'img_path')"
        )

    label_col = col_map.get("label")  # optional: true / miscaptioned / out-of-context

    base_dir = os.path.dirname(os.path.abspath(csv_path))

    triples: List[Tuple[ClaimData, EvidenceData, Dict[str, Any]]] = []

    for idx, row in df.iterrows():
        caption = str(row.get(caption_col, "") or "").strip()
        if not caption:
            continue

        img_rel = str(row.get(image_col, "") or "").strip()
        if not img_rel:
            img_full = None
        else:
            img_full = img_rel
            if not os.path.isabs(img_full):
                img_full = os.path.join(base_dir, img_rel)

        evidence_img = None
        if img_full and os.path.exists(img_full):
            if backend == "hf":
                evidence_img = load_local_image_fixed(img_full, cache_dir)
            else:
                evidence_img = load_local_image(img_full, image_size)
        else:
            if img_full:
                print(f"[WARN] VERITE image missing: {img_full}")

        claim_id = str(row.get("id", idx))

        claim = ClaimData(
            claim_id=claim_id,
            text=caption,
            modality=ModalityType.TEXT_ONLY,
        )

        evidence = EvidenceData(
            evidence_id=f"{claim_id}_evidence",
            image=evidence_img,
            image_url=img_rel if img_rel else None,
        )

        if evidence.has_image():
            evidence.modality = ModalityType.IMAGE_ONLY
        else:
            evidence.modality = ModalityType.NONE

        dataset_label = str(row.get(label_col, "")).strip() if label_col else None

        meta: Dict[str, Any] = {
            "dataset_type": "verite",
            "dataset_label": dataset_label,   # 'true', 'miscaptioned', 'out-of-context'
            "image_rel_path": img_rel,
            "image_full_path": img_full,
            "row_index": int(idx),
        }

        triples.append((claim, evidence, meta))

    print(f"[INFO] Loaded {len(triples)} VERITE claim-evidence pairs")
    return triples


# ============================================================================
# MAIN PIPELINE (GENERIC FOR ALL DATASETS, INCLUDING VERITE)
# ============================================================================

def run_classification(
    data_path: str,
    dataset_type: str,
    model_key: str,
    output_json: str,
    cache_dir: str = "image_cache",
    batch_size: int = 4,
    quantize: Optional[str] = None,
    checkpoint_interval: int = 100,
    max_text_per_claim: int = 40,
):
    """
    Main classification pipeline for Factify / MOCHEG / MR2 / VERITE.
    """
    print(f"[INFO] Starting classification")
    print(f"[INFO] Input: {data_path}")
    print(f"[INFO] Dataset: {dataset_type}")
    print(f"[INFO] Model: {model_key}")
    print(f"[INFO] Output: {output_json}")

    # -------------------------
    # Load model
    # -------------------------
    print(f"\n[INFO] Initializing model...")
    vlm = UnifiedVLM(model_key=model_key, quantize=quantize)
    classifier = EvidenceClassifier(vlm)

    backend = MODEL_CONFIG[model_key]["backend"]
    if model_key == "paligemma":
        image_size = 448
    else:
        image_size = MODEL_CONFIG[model_key].get("image_size", 768)

    # -------------------------
    # Load dataset
    # -------------------------
    print(f"\n[INFO] Loading dataset...")
    if dataset_type == "factify":
        pairs = load_factify_data_fixed(data_path, image_size, cache_dir)
    elif dataset_type == "mocheg":
        split = "val"
        pairs = load_mocheg_data(
            data_root=data_path,
            split=split,
            image_size=image_size,
            cache_dir=cache_dir,
            max_text_per_claim=max_text_per_claim,
        )
    elif dataset_type == "mr2":
        pairs = load_mr2_data(data_path, image_size, cache_dir)
    elif dataset_type == "verite":
        # returns (claim, evidence, meta) triples
        pairs = load_verite_data(
            data_path, image_size, cache_dir, backend=backend
        )
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    if not pairs:
        print("[ERROR] No data loaded!")
        return

    print(f"[INFO] Loaded {len(pairs)} claim-evidence pairs")

    # -------------------------
    # Resume from checkpoint
    # -------------------------
    results: List[Dict[str, Any]] = []
    start_idx = 0
    if os.path.exists(output_json):
        try:
            with open(output_json, "r", encoding="utf-8") as f:
                results = json.load(f)
            start_idx = len(results)
            print(f"[INFO] Resuming from index {start_idx}")
        except json.JSONDecodeError:
            print("[WARN] Existing output_json is not valid JSON, starting fresh")
            results = []
            start_idx = 0

    # -------------------------
    # Process
    # -------------------------
    for i, pair in enumerate(
        tqdm(pairs[start_idx:], desc="Classifying")
    ):
        idx = start_idx + i

        # Support both (claim, evidence) and (claim, evidence, meta)
        if isinstance(pair, (list, tuple)) and len(pair) == 3:
            claim, evidence, meta = pair
        else:
            claim, evidence = pair
            meta = {}

        try:
            result = classifier.classify_combined(claim, evidence, batch_size)

            # Generic fields
            result.update(
                {
                    "claim_text": claim.get_full_text(),
                    "claim_image_url": claim.image_url,
                    "evidence_image_url": evidence.image_url,
                    "dataset": dataset_type,        # keep for backward compatibility
                    "dataset_type": dataset_type,   # match reference file
                }
            )

            # Add dataset-specific metadata (e.g. VERITE labels/paths)
            if meta:
                result.update(meta)

            results.append(result)

            if (idx + 1) % checkpoint_interval == 0:
                os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
                with open(output_json, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"[CHECKPOINT] Saved {idx + 1} results")

        except Exception as e:
            print(f"[ERROR] Failed on claim {claim.claim_id}: {e}")
            import traceback
            traceback.print_exc()
            results.append(
                {
                    "claim_id": claim.claim_id,
                    "error": str(e),
                    "skipped": True,
                }
            )

    # -------------------------
    # Final save
    # -------------------------
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] Saved {len(results)} results to {output_json}")

    # -------------------------
    # Stats
    # -------------------------
    print("\n[STATISTICS]")
    successful = len([r for r in results if not r.get("skipped")])
    print(f"Successful: {successful}/{len(results)}")

    modality_stats: Dict[str, int] = {}
    for result in results:
        if not result.get("skipped"):
            key = f"{result['claim_modality']}_{result['evidence_modality']}"
            modality_stats[key] = modality_stats.get(key, 0) + 1

    print("\nModality combinations:")
    for combo, count in sorted(modality_stats.items()):
        print(f"  {combo}: {count}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified Multimodal Evidence Classification (Hybrid: vLLM + HF)"
    )
    parser.add_argument(
        "--data_path",
        required=True,
        help="Path to data (CSV for Factify/VERITE, root dir for MOCHEG, JSON for MR2)",
    )
    parser.add_argument(
        "--dataset_type",
        required=True,
        choices=["factify", "mocheg", "mr2", "verite"],
        help="Dataset format type",
    )
    parser.add_argument(
        "--model_key",
        required=True,
        choices=list(MODEL_CONFIG.keys()),
        help="Model to use (paligemma=HF, others=vLLM)",
    )
    parser.add_argument(
        "--output_json",
        required=True,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--cache_dir",
        default="image_cache",
        help="Directory for caching downloaded images",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for text evidence classification (lower for HF models)",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        default=None,
        choices=["awq", "gptq"],
        help="Manual quantization for vLLM models (ignored for PaliGemma)",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=100,
        help="Save checkpoint every N samples",
    )
    parser.add_argument(
        "--max_text_per_claim",
        type=int,
        default=40,
        help="Maximum text evidence sentences per claim (for MOCHEG)",
    )

    args = parser.parse_args()

    if args.model_key == "paligemma" and args.quantize:
        print("[WARN] --quantize ignored for PaliGemma (uses HF 4-bit)")

    run_classification(
        data_path=args.data_path,
        dataset_type=args.dataset_type,
        model_key=args.model_key,
        output_json=args.output_json,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        quantize=args.quantize,
        checkpoint_interval=args.checkpoint_interval,
        max_text_per_claim=args.max_text_per_claim,
    )


if __name__ == "__main__":
    main()


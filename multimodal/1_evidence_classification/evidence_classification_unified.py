#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Multimodal Evidence Classification System
Now also supports VERITE dataset, using:
- caption column as the CLAIM text
- image_path/Image_path column as the EVIDENCE image

Supported Modality Combinations:
CLAIM: text + image, text only, image only
EVIDENCE: text + image, text only, image only, none

Supported Models: PaliGemma, Qwen2VL, Idefics3, etc.
Supported Datasets: Factify, MOCHEG, MR2, VERITE
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
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
import numpy as np
import pandas as pd

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
    # Full precision models
    "paligemma": {
        "model_id": "mlx-community/paligemma2-3b-mix-448-4bit",  # google/paligemma2-10b-mix-448
        "max_new_tokens": 256,
        "image_size": 448,
        "low_res_size": 224,
        "max_model_len": 3500,
        "visual_token_count": 256,
        "use_chat_template": False,
    },
    "qwen2vl": {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "max_new_tokens": 256,
        "image_size": 768,
        "max_model_len": 10000,
        "visual_token_count": 256,
        "use_chat_template": True,
    },
    "idefics3": {
        "model_id": "leon-se/Idefics3-8B-Llama3-bnb_nf4",  # HuggingFaceM4/Idefics3-8B-Llama3
        "max_new_tokens": 256,
        "image_size": 768,
        "max_model_len": 4096,
        "visual_token_count": 256,
        "use_chat_template": True,
    },

    # AWQ Quantized models (4-bit)
    "paligemma-awq": {
        "model_id": "google/paligemma2-10b-mix-448",  # Use base if AWQ not available
        "max_new_tokens": 256,
        "image_size": 448,
        "low_res_size": 224,
        "max_model_len": 8192,
        "visual_token_count": 256,
        "use_chat_template": False,
        "quantization": "awq",
        "dtype": "half",
    },
    "qwen2vl-awq": {
        "model_id": "Qwen/Qwen2-VL-7B-Instruct-AWQ",
        "max_new_tokens": 256,
        "image_size": 768,
        "max_model_len": 8192,
        "visual_token_count": 256,
        "use_chat_template": True,
        "quantization": "awq",
        "dtype": "half",
    },
    "idefics3-awq": {
        "model_id": "ronantakizawa/idefics3-8b-llama3-awq",  # Use base if AWQ not available
        "max_new_tokens": 256,
        "image_size": 768,
        "max_model_len": 4096,
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


def download_image(url: str, max_size: int = 768, cache_dir: str = "image_cache") -> Optional[Image.Image]:
    """Download and cache images"""
    if not url or str(url).strip().lower() in {"", "nan", "none"}:
        return None

    os.makedirs(cache_dir, exist_ok=True)
    url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()
    ext = os.path.splitext(url)[-1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        ext = ".jpg"

    local_path = os.path.join(cache_dir, f"{url_hash}{ext}")

    # Check cache
    if os.path.exists(local_path):
        try:
            img = Image.open(local_path).convert("RGB")
            img.thumbnail((max_size, max_size))
            return img
        except Exception:
            os.remove(local_path)

    # Download
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        img.thumbnail((max_size, max_size))
        img.save(local_path)
        return img
    except Exception as e:
        print(f"[warn] Image download failed: {url} ({e})")
        return None


def load_local_image(path: str, max_size: int = 768) -> Optional[Image.Image]:
    """Load local image file"""
    if not path or not os.path.exists(path):
        return None
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((max_size, max_size))
        return img
    except Exception as e:
        print(f"[warn] Image load failed: {path} ({e})")
        return None


_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9""\(\[])')


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences"""
    text = (text or "").strip()
    if not text:
        return []
    chunks = _SENT_SPLIT_RE.split(text)
    sents = [re.sub(r'\s+', ' ', c).strip() for c in chunks]
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
        max_tokens: int = 2000
    ) -> str:
        """
        Prompt for text evidence classification.
        Adapts based on claim modality.
        """
        # Truncate inputs
        claim_text = truncate_text(processor, claim.get_full_text(), max_tokens // 3)
        evidence_sentence = truncate_text(processor, evidence_sentence, max_tokens // 3)

        # Build context based on claim modality
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
        max_tokens: int = 2000
    ) -> str:
        """
        Prompt for visual evidence classification.
        Adapts based on claim and evidence modalities.
        """
        # Truncate inputs
        claim_text = truncate_text(processor, claim.get_full_text(), max_tokens // 3)
        evidence_text = truncate_text(processor, evidence.get_full_text(), max_tokens // 3)

        # Build context
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
# MODEL WRAPPER
# ============================================================================

class UnifiedVLM:
    """Unified VLM wrapper with modality-aware inference and quantization support"""

    def __init__(self, model_key: str, quantize: Optional[str] = None):
        assert model_key in MODEL_CONFIG, f"Unknown model: {model_key}"
        self.model_key = model_key
        self.config = MODEL_CONFIG[model_key]

        print(f"[INFO] Loading {model_key}: {self.config['model_id']}")

        self.processor = AutoProcessor.from_pretrained(
            self.config["model_id"], trust_remote_code=True
        )

        # Base LLM kwargs
        llm_kwargs = {
            "model": self.config["model_id"],
            "max_model_len": self.config["max_model_len"],
            "trust_remote_code": True,
        }

        # Quantization handling (priority: config > CLI argument)
        if "quantization" in self.config:
            # Built-in quantized model
            llm_kwargs["quantization"] = self.config["quantization"]
            llm_kwargs["dtype"] = self.config.get("dtype", "half")
            llm_kwargs["gpu_memory_utilization"] = 0.7  # Lower for quantized
            print(f"[INFO] Using built-in {self.config['quantization']} quantization")
        elif quantize:
            # Manual quantization override
            llm_kwargs["quantization"] = quantize
            llm_kwargs["dtype"] = "half"
            llm_kwargs["gpu_memory_utilization"] = 0.7
            print(f"[INFO] Applying manual {quantize} quantization")
        else:
            # Full precision
            llm_kwargs["dtype"] = "bfloat16"
            llm_kwargs["gpu_memory_utilization"] = 0.9

        # Model-specific settings
        if "paligemma" in model_key:
            llm_kwargs["enforce_eager"] = True
            print("[INFO] PaliGemma: Enforcing eager mode")

        llm_kwargs["limit_mm_per_prompt"] = {"image": 2}  # Max 2 images

        # Initialize vLLM
        try:
            self.llm = LLM(**llm_kwargs)
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            print("[INFO] Retrying with reduced memory...")
            llm_kwargs["gpu_memory_utilization"] = 0.5
            self.llm = LLM(**llm_kwargs)

        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=self.config["max_new_tokens"],
            skip_special_tokens=True,
        )

    def _build_formatted_prompt(
        self,
        plain_prompt: str,
        images: List[Image.Image]
    ) -> str:
        """Build formatted prompt based on model type"""
        valid_imgs = [img for img in images if is_valid_image(img)]

        if not self.config["use_chat_template"]:
            # PaliGemma: Simple format
            if valid_imgs:
                image_tokens = "".join(["<image>\n"] * len(valid_imgs))
                return image_tokens + plain_prompt
            return plain_prompt
        else:
            # Qwen2VL/Idefics3: Chat template
            content = [{"type": "image"} for _ in valid_imgs]
            content.append({"type": "text", "text": plain_prompt})
            messages = [{"role": "user", "content": content}]
            return self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )

    def classify_batch(
        self,
        prompts: List[str],
        images_lists: List[List[Image.Image]]
    ) -> List[str]:
        """
        Batch classification with modality support.

        Args:
            prompts: List of plain text prompts
            images_lists: List of image lists (can be empty for text-only)

        Returns:
            List of model responses
        """
        assert len(prompts) == len(images_lists)

        requests = []
        for prompt, imgs in zip(prompts, images_lists):
            valid_imgs = [img for img in imgs if is_valid_image(img)]
            formatted_prompt = self._build_formatted_prompt(prompt, valid_imgs)

            requests.append({
                "prompt": formatted_prompt,
                "multi_modal_data": {"image": valid_imgs} if valid_imgs else {}
            })

        # Sequential processing for PaliGemma
        if "paligemma" in self.model_key:
            responses = []
            for req in tqdm(requests, desc="PaliGemma (sequential)", leave=False):
                try:
                    out = self.llm.generate([req], self.sampling_params)
                    responses.append(out[0].outputs[0].text.strip())
                except Exception as e:
                    print(f"[ERROR] Generation failed: {e}")
                    responses.append("ERROR")
            return responses

        # Batch processing for others
        try:
            outputs = self.llm.generate(requests, self.sampling_params)
            return [out.outputs[0].text.strip() for out in outputs]
        except Exception as e:
            print(f"[ERROR] Batch generation failed: {e}")
            return ["ERROR"] * len(prompts)


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
        batch_size: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Classify text evidence sentences.
        Handles: claim_text + claim_image, claim_text only, claim_image only
        """
        if not sentences:
            return []

        predictions = []

        for i in range(0, len(sentences), batch_size):
            batch_sents = sentences[i:i+batch_size]

            # Build prompts
            prompts = [
                self.prompt_builder.build_text_evidence_prompt(
                    claim, sent, self.vlm.processor
                )
                for sent in batch_sents
            ]

            # Prepare images (claim image if available)
            images_lists = [
                [claim.image] if claim.has_image() else []
                for _ in batch_sents
            ]

            # Classify
            responses = self.vlm.classify_batch(prompts, images_lists)

            # Parse results
            for sent, resp in zip(batch_sents, responses):
                predictions.append({
                    "sentence": sent,
                    "prediction": parse_label(resp),
                    "raw_response": resp
                })

        return predictions

    def classify_visual_evidence(
        self,
        claim: ClaimData,
        evidence: EvidenceData
    ) -> Tuple[str, str]:
        """
        Classify visual evidence.
        Handles all modality combinations.
        """
        if not evidence.has_image():
            return "NO_EVIDENCE", "No visual evidence available"

        # Build prompt
        prompt = self.prompt_builder.build_visual_evidence_prompt(
            claim, evidence, self.vlm.processor
        )

        # Prepare images
        images = []
        if claim.has_image():
            images.append(claim.image)
        images.append(evidence.image)

        # Classify
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
        batch_size: int = 8
    ) -> Dict[str, Any]:
        """
        Classify all available evidence (text + visual).

        Returns comprehensive results with:
        - Text evidence predictions (if available)
        - Visual evidence prediction (if available)
        - Supporting/refuting evidence lists
        """
        result = {
            "claim_id": claim.claim_id,
            "claim_modality": claim.modality.value,
            "evidence_modality": evidence.modality.value,
            "supporting_text": [],
            "refuting_text": [],
            "text_predictions": [],
            "visual_label": None,
            "visual_response": None,
        }

        # Classify text evidence
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

        # Classify visual evidence
        if evidence.has_image():
            label, response = self.classify_visual_evidence(claim, evidence)
            result["visual_label"] = label
            result["visual_response"] = response

        return result


# ============================================================================
# DATA LOADERS
# ============================================================================

def load_factify_data(csv_path: str, image_size: int, cache_dir: str) -> List[Tuple[ClaimData, EvidenceData]]:
    """Load Factify dataset format"""
    pairs: List[Tuple[ClaimData, EvidenceData]] = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Claim
            claim = ClaimData(
                claim_id=str(row.get("Id", "")).strip(),
                text=row.get("claim", "").strip() or None,
                ocr=row.get("claim_ocr", "").strip() or None,
                image_url=row.get("claim_image", "").strip() or None,
            )
            claim.image = download_image(claim.image_url, image_size, cache_dir)

            # Determine claim modality
            if claim.has_text() and claim.has_image():
                claim.modality = ModalityType.TEXT_IMAGE
            elif claim.has_text():
                claim.modality = ModalityType.TEXT_ONLY
            elif claim.has_image():
                claim.modality = ModalityType.IMAGE_ONLY

            # Evidence
            evidence = EvidenceData(
                evidence_id=f"{claim.claim_id}_evidence",
                text=row.get("document", "").strip() or None,
                ocr=row.get("document_ocr", "").strip() or None,
                image_url=row.get("document_image", "").strip() or None,
            )
            evidence.image = download_image(evidence.image_url, image_size, cache_dir)

            # Determine evidence modality
            if evidence.has_text() and evidence.has_image():
                evidence.modality = ModalityType.TEXT_IMAGE
            elif evidence.has_text():
                evidence.modality = ModalityType.TEXT_ONLY
            elif evidence.has_image():
                evidence.modality = ModalityType.IMAGE_ONLY

            pairs.append((claim, evidence))

    return pairs


def load_mocheg_data(
    data_root: str,
    split: str,
    image_size: int,
    cache_dir: str,
    max_text_per_claim: int = 20
) -> List[Tuple[ClaimData, EvidenceData]]:
    """Load MOCHEG dataset format"""
    from collections import defaultdict

    split_dir = os.path.join(data_root, split)
    corpus2_path = os.path.join(split_dir, "Corpus2.csv")
    qrels_path = os.path.join(split_dir, "text_evidence_qrels_sentence_level.csv")
    img_qrels_path = os.path.join(split_dir, "img_evidence_qrels.csv")
    images_dir = os.path.join(split_dir, "images")
    supp_path = os.path.join(data_root, "supplementary", "Corpus3_sentence_level.csv")

    # Load claims
    c2 = pd.read_csv(corpus2_path, dtype=str).fillna("")
    claims_dict: Dict[str, str] = {}
    for _, row in c2.iterrows():
        cid = str(row.get("claim_id", "")).strip()
        if cid:
            claims_dict[cid] = str(row.get("Claim", "")).strip()

    # Load text evidence
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

    # Cap sentences
    if max_text_per_claim:
        for k in text_evi:
            text_evi[k] = text_evi[k][:max_text_per_claim]

    # Load image evidence
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

    # Build pairs
    pairs: List[Tuple[ClaimData, EvidenceData]] = []
    for cid in claims_dict:
        claim = ClaimData(
            claim_id=cid,
            text=claims_dict[cid],
            modality=ModalityType.TEXT_ONLY
        )

        # Evidence
        evidence = EvidenceData(
            evidence_id=f"{cid}_evidence",
            sentences=text_evi.get(cid, [])
        )

        # Load first image if available
        if cid in img_evi and img_evi[cid]:
            evidence.image = load_local_image(img_evi[cid][0], image_size)

        # Determine evidence modality
        if evidence.sentences and evidence.has_image():
            evidence.modality = ModalityType.TEXT_IMAGE
        elif evidence.sentences:
            evidence.modality = ModalityType.TEXT_ONLY
        elif evidence.has_image():
            evidence.modality = ModalityType.IMAGE_ONLY

        pairs.append((claim, evidence))

    return pairs


def load_mr2_data(json_path: str, image_size: int, cache_dir: str) -> List[Tuple[ClaimData, EvidenceData]]:
    """Load MR2 dataset format"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    pairs: List[Tuple[ClaimData, EvidenceData]] = []
    base_dir = os.path.dirname(json_path)

    for item in data:
        # Claim
        claim = ClaimData(
            claim_id=str(item["id"]),
            text=item.get("claim_text", ""),
            modality=ModalityType.TEXT_IMAGE
        )

        # Load claim image
        claim_img_path = os.path.join(base_dir, "..", item["claim_image"])
        if os.path.exists(claim_img_path):
            claim.image = load_local_image(claim_img_path, image_size)

        # Evidence - combine all texts and images
        evidence = EvidenceData(
            evidence_id=f"{item['id']}_evidence",
            sentences=item.get("evidence_texts", [])
        )

        # Load ALL evidence images, use first
        evidence_images: List[Image.Image] = []
        for img_path in item.get("evidence_images", []):
            full_path = os.path.join(base_dir, "..", img_path)
            if os.path.exists(full_path):
                img = load_local_image(full_path, image_size)
                if img is not None:
                    evidence_images.append(img)

        if evidence_images:
            evidence.image = evidence_images[0]

        # Determine evidence modality
        if evidence.sentences and evidence.has_image():
            evidence.modality = ModalityType.TEXT_IMAGE
        elif evidence.sentences:
            evidence.modality = ModalityType.TEXT_ONLY
        elif evidence.has_image():
            evidence.modality = ModalityType.IMAGE_ONLY

        pairs.append((claim, evidence))

    return pairs


def load_verite_data(
    csv_path: str,
    image_size: int,
    cache_dir: str
) -> List[Tuple[ClaimData, EvidenceData, Dict[str, Any]]]:
    """
    Load VERITE dataset format.

    Mapping (as requested):
      - caption column -> claim text
      - image_path / Image_path column -> evidence image (local path)

    Each row becomes:
      Claim: text-only (caption)
      Evidence: image-only (loaded from image_path)
    """
    df = pd.read_csv(csv_path)
    # Normalize column names for robustness
    col_map = {c.lower(): c for c in df.columns}

    caption_col = col_map.get("caption")
    if caption_col is None:
        raise ValueError("VERITE CSV must contain a 'caption' column")

    image_col = (
        col_map.get("image_path")
        or col_map.get("imagepath")
        or col_map.get("image")
        or col_map.get("image_path".lower())
    )
    if image_col is None:
        raise ValueError("VERITE CSV must contain an 'image_path' (or similar) column")

    label_col = col_map.get("label")  # true / miscaptioned / out-of-context

    base_dir = os.path.dirname(os.path.abspath(csv_path))

    triples: List[Tuple[ClaimData, EvidenceData, Dict[str, Any]]] = []

    for idx, row in df.iterrows():
        caption = str(row.get(caption_col, "") or "").strip()
        img_rel = str(row.get(image_col, "") or "").strip()

        if not caption:
            # Skip rows without caption
            continue

        # Build full path to image
        img_full = img_rel
        if not os.path.isabs(img_full):
            img_full = os.path.join(base_dir, img_rel)

        evidence_img = load_local_image(img_full, image_size)

        # Claim: caption is the claim text
        claim_id = str(row.get("id", idx))
        claim = ClaimData(
            claim_id=claim_id,
            text=caption,
            modality=ModalityType.TEXT_ONLY,
        )

        # Evidence: image only
        evidence = EvidenceData(
            evidence_id=f"{claim_id}_evidence",
            image=evidence_img,
        )
        if evidence.has_image():
            evidence.modality = ModalityType.IMAGE_ONLY

        dataset_label = str(row.get(label_col, "")).strip() if label_col else None

        meta: Dict[str, Any] = {
            "dataset_type": "verite",
            "dataset_label": dataset_label,          # 'true', 'miscaptioned', 'out-of-context'
            "image_rel_path": img_rel,
            "image_full_path": img_full,
            "row_index": int(idx),
        }

        triples.append((claim, evidence, meta))

    return triples


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_classification(
    data_path: str,
    dataset_type: str,
    model_key: str,
    output_json: str,
    cache_dir: str = "image_cache",
    batch_size: int = 8,
    quantize: Optional[str] = None,
    checkpoint_interval: int = 100,
    max_text_per_claim: int = 40,
):
    """
    Main classification pipeline.

    Args:
        data_path: Path to dataset (CSV for Factify/VERITE, root dir for MOCHEG, JSON for MR2)
        dataset_type: "factify" or "mocheg" or "mr2" or "verite"
        model_key: Model identifier
        output_json: Output file path
        cache_dir: Image cache directory
        batch_size: Batch size for text evidence classification
        quantize: Quantization method
        checkpoint_interval: Save checkpoint every N samples
        max_text_per_claim: Max text evidence per claim (for MOCHEG)
    """
    # Load model
    vlm = UnifiedVLM(model_key, quantize)
    classifier = EvidenceClassifier(vlm)

    # Load data
    image_size = vlm.config.get("low_res_size", vlm.config["image_size"])

    print(f"[INFO] Loading {dataset_type} dataset from {data_path}")
    if dataset_type == "factify":
        pairs = load_factify_data(data_path, image_size, cache_dir)
    elif dataset_type == "mocheg":
        split = os.path.basename(data_path) if os.path.isfile(data_path) else "val"
        pairs = load_mocheg_data(data_root=data_path, split=split, image_size=image_size,
                                 cache_dir=cache_dir, max_text_per_claim=max_text_per_claim)
    elif dataset_type == "mr2":
        pairs = load_mr2_data(data_path, image_size, cache_dir)
    elif dataset_type == "verite":
        # returns (claim, evidence, meta) triples
        pairs = load_verite_data(data_path, image_size, cache_dir)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    print(f"[INFO] Loaded {len(pairs)} items")

    # Resume from checkpoint if exists
    results: List[Dict[str, Any]] = []
    start_idx = 0
    if os.path.exists(output_json):
        with open(output_json, 'r', encoding='utf-8') as f:
            try:
                results = json.load(f)
                start_idx = len(results)
                print(f"[INFO] Resuming from index {start_idx}")
            except json.JSONDecodeError:
                print("[WARN] Existing output_json is not valid JSON, starting fresh")
                results = []
                start_idx = 0

    # Process
    for i, pair in enumerate(tqdm(pairs[start_idx:], desc="Classifying")):
        idx = start_idx + i

        # Support both (claim, evidence) and (claim, evidence, meta)
        if isinstance(pair, (list, tuple)) and len(pair) == 3:
            claim, evidence, meta = pair
        else:
            claim, evidence = pair
            meta = {}

        try:
            result = classifier.classify_combined(claim, evidence, batch_size)

            # Add generic metadata
            result.update({
                "claim_text": claim.get_full_text(),
                "claim_image_url": claim.image_url,
                "evidence_image_url": evidence.image_url,
            })

            # Add dataset-specific metadata (e.g., VERITE labels/paths)
            if meta:
                result.update(meta)

            results.append(result)

            # Checkpoint
            if (idx + 1) % checkpoint_interval == 0:
                os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
                with open(output_json, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"[CHECKPOINT] Saved {idx + 1} results")

        except Exception as e:
            print(f"[ERROR] Failed on claim {claim.claim_id}: {e}")
            results.append({
                "claim_id": claim.claim_id,
                "error": str(e),
                "skipped": True
            })

    # Final save
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Saved {len(results)} results to {output_json}")

    # Print statistics
    modality_stats: Dict[str, int] = {}
    for result in results:
        if not result.get("skipped"):
            key = f"{result['claim_modality']}_{result['evidence_modality']}"
            modality_stats[key] = modality_stats.get(key, 0) + 1

    print("\n[STATS] Modality combinations processed:")
    for combo, count in sorted(modality_stats.items()):
        print(f"  {combo}: {count}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified Multimodal Evidence Classification (Factify, MOCHEG, MR2, VERITE)"
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
        help="Model to use",
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
        default=8,
        help="Batch size for text evidence classification",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        default=None,
        choices=["awq", "gptq", "bitsandbytes"],
        help="Manual quantization override (for non-quantized models)",
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
        help="Maximum text evidence sentences per claim (used for MOCHEG)",
    )

    args = parser.parse_args()

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

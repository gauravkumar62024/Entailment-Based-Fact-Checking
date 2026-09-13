#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 2: Multimodal Justification Generation using Quantized PaliGemma (HF Transformers)

Takes evidence classification output from Step 1 and generates:
- Support justification (why claim is true based on supporting evidence)
- Refute justification (why claim is false based on refuting evidence)

Uses 4-bit quantized PaliGemma with Hugging Face Transformers (matching Step 1 approach)
"""

import os
import json
import argparse
import hashlib
from io import BytesIO
from typing import List, Optional
from pathlib import Path

import requests
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig


# =============================
# Image Utilities
# =============================
def is_valid_image(img: Optional[Image.Image]) -> bool:
    """Validate image before processing"""
    if img is None:
        return False
    try:
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


def resize_image_for_paligemma(img: Image.Image, target_size: int = 448) -> Image.Image:
    """Resize image to exact square size for PaliGemma"""
    if img is None:
        return None
    try:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_resized = img.resize((target_size, target_size), Image.LANCZOS)
        if not is_valid_image(img_resized):
            print(f"[WARN] Invalid image after resize: {img_resized.size}")
            return None
        return img_resized
    except Exception as e:
        print(f"[ERROR] Image resize failed: {e}")
        return None


def download_image(url: str, target_size: int = 448, cache_dir: str = "image_cache") -> Optional[Image.Image]:
    """Download and cache images with proper resizing for PaliGemma"""
    if not url or str(url).strip().lower() in {"", "nan", "none", "null"}:
        return None
    
    os.makedirs(cache_dir, exist_ok=True)
    url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()
    ext = ".jpg"
    local_path = os.path.join(cache_dir, f"{url_hash}_pali{target_size}{ext}")
    
    # Check cache
    if os.path.exists(local_path):
        try:
            img = Image.open(local_path).convert("RGB")
            return img
        except Exception:
            os.remove(local_path)
    
    # Download and resize
    try:
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img_resized = resize_image_for_paligemma(img, target_size)
        if img_resized:
            img_resized.save(local_path, quality=95)
            return img_resized
        return None
    except Exception as e:
        print(f"[WARN] Image download failed: {url} | Error: {e}")
        return None


def load_local_image(path: str, target_size: int = 448) -> Optional[Image.Image]:
    """Load local image with proper resizing"""
    if not path or not os.path.exists(path):
        return None
    try:
        img = Image.open(path).convert("RGB")
        return resize_image_for_paligemma(img, target_size)
    except Exception as e:
        print(f"[WARN] Image load failed: {path} ({e})")
        return None


def concatenate_images_horizontally(
    img1: Image.Image, 
    img2: Image.Image, 
    target_size: int = 448
) -> Image.Image:
    """
    Concatenate two images side-by-side then resize to target_size.
    Workaround for PaliGemma's single-image limitation.
    """
    if img1 is None and img2 is None:
        return None
    
    # Resize inputs first
    if img1 is not None:
        img1 = resize_image_for_paligemma(img1, target_size)
    if img2 is not None:
        img2 = resize_image_for_paligemma(img2, target_size)
    
    # Handle missing images
    if img1 is None:
        return img2
    if img2 is None:
        return img1
    
    try:
        # Create side-by-side canvas (896x448)
        combined = Image.new('RGB', (target_size * 2, target_size))
        combined.paste(img1, (0, 0))
        combined.paste(img2, (target_size, 0))
        
        # Resize back to 448x448
        combined_final = resize_image_for_paligemma(combined, target_size)
        return combined_final
    except Exception as e:
        print(f"[ERROR] Image concatenation failed: {e}")
        return img1


def truncate_text(text: str, max_chars: int = 800) -> str:
    """Simple character-based truncation"""
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


# =============================
# Prompt Builders (OCR Removed)
# =============================
def build_support_prompt(
    claim: str,
    supporting_text: List[str],
    visual_evidence_label: Optional[str] = None,
    visual_raw_response: Optional[str] = None
) -> str:
    """
    Build prompt for SUPPORT justification.
    Modified: Removed OCR sections, kept core structure.
    """
    claim = truncate_text(claim, 600)
    
    # Format text evidence
    if supporting_text:
        truncated_sents = [truncate_text(sent, 200) for sent in supporting_text[:5]]
        text_evidence = "\n".join(f"- {sent}" for sent in truncated_sents)
    else:
        text_evidence = "No supporting text evidence provided."
    
    # Format visual evidence
    visual_info = ""
    if visual_evidence_label == "SUPPORT":
        visual_info = "**Visual Evidence Classification:** SUPPORT\n"
        if visual_raw_response:
            visual_info += f"**Visual Analysis:** {truncate_text(visual_raw_response, 200)}\n"
    elif visual_evidence_label == "REFUTE":
        visual_info = "**Visual Evidence Classification:** REFUTE (contradictory to text evidence)\n"
    else:
        visual_info = "**Visual Evidence:** Not available or inconclusive.\n"
    
    prompt = f"""You are a fact-checking assistant specially designed to perform abductive reasoning for the multimodal fact-checking task.

### Task:
Given the following claim and a veracity label **True**, analyze the statement and provide a concise, well-reasoned explanation (150 words or fewer) based on the provided textual and visual evidence. 
Do not mention the label in your response. 
Focus only on key supporting factors and logical reasoning, and consider both textual and visual information when forming your explanation.

### Claim:
{claim}

### Supporting Text Evidence:
{text_evidence}

{visual_info}

### Instructions:
- Do NOT explicitly mention the label "True" in your response
- Synthesize information from text and visual analysis
- If images contain relevant details (objects, text, context), reference them naturally
- Focus only on key supporting factors and logical reasoning
- Use clear, objective language
- Keep the explanation within 150 words

### Response:
"""
    return prompt


def build_refute_prompt(
    claim: str,
    refuting_text: List[str],
    visual_evidence_label: Optional[str] = None,
    visual_raw_response: Optional[str] = None
) -> str:
    """
    Build prompt for REFUTE justification.
    Modified: Removed OCR sections, kept core structure.
    """
    claim = truncate_text(claim, 600)
    
    # Format text evidence
    if refuting_text:
        truncated_sents = [truncate_text(sent, 200) for sent in refuting_text[:5]]
        text_evidence = "\n".join(f"- {sent}" for sent in truncated_sents)
    else:
        text_evidence = "No refuting text evidence provided."
    
    # Format visual evidence
    visual_info = ""
    if visual_evidence_label == "REFUTE":
        visual_info = "**Visual Evidence Classification:** REFUTE\n"
        if visual_raw_response:
            visual_info += f"**Visual Analysis:** {truncate_text(visual_raw_response, 200)}\n"
    elif visual_evidence_label == "SUPPORT":
        visual_info = "**Visual Evidence Classification:** SUPPORT (contradictory to text evidence)\n"
    else:
        visual_info = "**Visual Evidence:** Not available or inconclusive.\n"
    
    prompt = f"""You are a fact-checking assistant specially designed to perform abductive reasoning for the multimodal fact-checking task.

### Task:
Given the following claim and a veracity label **False**, analyze the statement and provide a concise, well-reasoned explanation (150 words or fewer) based on the provided textual and visual evidence. 
Do not mention the label in your response. 
Focus only on key contradictory factors and logical reasoning, and consider both textual and visual information when forming your explanation.

### Claim:
{claim}

### Refuting Text Evidence:
{text_evidence}

{visual_info}

### Instructions:
- Do NOT explicitly mention the label "False" in your response
- Synthesize information from text and visual analysis
- If images contain contradictory details (objects, text, context), reference them naturally
- Focus only on key contradictory factors and logical reasoning
- Use clear, objective language
- Keep the explanation within 150 words

### Response:
"""
    return prompt


# =============================
# PaliGemma Model Wrapper
# =============================
class PaliGemmaJustificationGenerator:
    """Hugging Face PaliGemma wrapper with 4-bit quantization for justification generation"""
    
    def __init__(self, model_id: str = "google/paligemma2-3b-pt-448"):
        print(f"[INFO] Loading PaliGemma with 4-bit quantization: {model_id}")
        
        # 4-bit quantization config
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        self.processor = PaliGemmaProcessor.from_pretrained(model_id, trust_remote_code=True)
        
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        
        # Fix pad token
        self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.model.generation_config.pad_token_id = self.processor.tokenizer.eos_token_id
        
        self.model.eval()
        self.device = next(self.model.parameters()).device
        
        print(f"[INFO] PaliGemma loaded on device: {self.device}")
    
    def generate(self, prompt: str, image: Image.Image, max_new_tokens: int = 512) -> str:
        """Generate justification for a single prompt + image"""
        try:
            # Handle missing or invalid image
            if not is_valid_image(image):
                blank_img = Image.new('RGB', (448, 448), color=(255, 255, 255))
                image = blank_img
                print("[INFO] Using blank placeholder image")
            
            # PaliGemma format: <image> token + prompt
            full_prompt = "<image>" + prompt
            
            inputs = self.processor(
                text=full_prompt,
                images=[image],
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    temperature=0.1,
                    top_p=0.9
                )
            
            # Decode only new tokens
            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            response = self.processor.decode(generated_ids, skip_special_tokens=True).strip()
            
            if not response or len(response) < 10:
                response = "Unable to generate justification based on available evidence."
            
            return response
            
        except Exception as e:
            print(f"[ERROR] Generation failed: {e}")
            import traceback
            traceback.print_exc()
            return "ERROR: Generation failed"


# =============================
# Main Processing Pipeline
# =============================
def process_justification_generation(
    input_json: str,
    output_json: str,
    model_id: str = "google/paligemma2-10b-pt-448",
    cache_dir: str = "image_cache",
    max_new_tokens: int = 512,
    checkpoint_interval: int = 50
):
    """
    Main pipeline: Generate support and refute justifications
    
    Input: JSON from Step 1 (evidence classification)
    Output: JSON with support_justification and refute_justification
    """
    
    # Load model
    generator = PaliGemmaJustificationGenerator(model_id)
    target_size = 448
    
    # Load input data from Step 1
    print(f"[INFO] Loading Step 1 output from {input_json}")
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"[INFO] Processing {len(data)} samples")
    
    # Resume from checkpoint if exists
    results = []
    start_idx = 0
    if os.path.exists(output_json):
        print(f"[INFO] Found existing output, resuming...")
        with open(output_json, "r", encoding="utf-8") as f:
            results = json.load(f)
        start_idx = len(results)
        print(f"[INFO] Resuming from index {start_idx}")
    
    # Process each sample
    for idx in tqdm(range(start_idx, len(data)), desc="Generating justifications"):
        entry = data[idx]
        
        try:
            # Extract fields from Step 1 output
            claim_id = entry.get("claim_id", "")
            claim_text = entry.get("claim_text", "")
            
            # Evidence from Step 1
            supporting_text = entry.get("supporting_text", [])
            refuting_text = entry.get("refuting_text", [])
            visual_label = entry.get("visual_label")
            visual_response = entry.get("visual_response")
            
            # Image URLs
            claim_img_url = entry.get("claim_image_url", "")
            evidence_img_url = entry.get("evidence_image_url", "")
            
            # Download/load images
            claim_img = None
            evidence_img = None
            
            # Try to load images
            if claim_img_url:
                # Check if it's a URL or local path
                if claim_img_url.startswith('http'):
                    claim_img = download_image(claim_img_url, target_size, cache_dir)
                else:
                    claim_img = load_local_image(claim_img_url, target_size)
            
            if evidence_img_url:
                if evidence_img_url.startswith('http'):
                    evidence_img = download_image(evidence_img_url, target_size, cache_dir)
                else:
                    evidence_img = load_local_image(evidence_img_url, target_size)
            
            # Concatenate images for PaliGemma's single-image constraint
            combined_img = concatenate_images_horizontally(claim_img, evidence_img, target_size)
            
            # Build prompts
            support_prompt = build_support_prompt(
                claim=claim_text,
                supporting_text=supporting_text,
                visual_evidence_label=visual_label,
                visual_raw_response=visual_response
            )
            
            refute_prompt = build_refute_prompt(
                claim=claim_text,
                refuting_text=refuting_text,
                visual_evidence_label=visual_label,
                visual_raw_response=visual_response
            )
            
            # Generate justifications
            print(f"[INFO] Generating justifications for claim {claim_id} ({idx+1}/{len(data)})")
            
            support_justification = generator.generate(
                prompt=support_prompt,
                image=combined_img,
                max_new_tokens=max_new_tokens
            )
            
            refute_justification = generator.generate(
                prompt=refute_prompt,
                image=combined_img,
                max_new_tokens=max_new_tokens
            )
            
            # Compile result
            result = {
                "claim_id": claim_id,
                "claim_text": claim_text,
                "support_justification": support_justification,
                "refute_justification": refute_justification,
                "supporting_evidence_count": len(supporting_text),
                "refuting_evidence_count": len(refuting_text),
                "has_visual_evidence": visual_label is not None
            }
            
            results.append(result)
            
            # Checkpoint
            if (idx + 1) % checkpoint_interval == 0:
                Path(output_json).parent.mkdir(parents=True, exist_ok=True)
                with open(output_json, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"[CHECKPOINT] Saved {len(results)} results")
        
        except Exception as e:
            print(f"[ERROR] Failed on claim {entry.get('claim_id', 'unknown')}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "claim_id": entry.get("claim_id", ""),
                "claim_text": entry.get("claim_text", ""),
                "support_justification": "ERROR: Generation failed",
                "refute_justification": "ERROR: Generation failed",
                "error": str(e)
            })
    
    # Final save
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[DONE] Saved {len(results)} results to {output_json}")
    
    # Print statistics
    print("\n[STATISTICS]")
    successful = len([r for r in results if "error" not in r])
    print(f"Successful: {successful}/{len(results)}")
    
    avg_support_len = np.mean([len(r.get("support_justification", "")) for r in results if "error" not in r])
    avg_refute_len = np.mean([len(r.get("refute_justification", "")) for r in results if "error" not in r])
    print(f"Average support justification length: {avg_support_len:.1f} chars")
    print(f"Average refute justification length: {avg_refute_len:.1f} chars")


# =============================
# CLI
# =============================
def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Generate Support and Refute Justifications using Quantized PaliGemma",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usage:
  python justification_generation_quantized.py \\
    --input_json outputs/mr2_evidence_classification.json \\
    --output_json outputs/mr2_justifications.json \\
    --model_id google/paligemma2-10b-pt-448 \\
    --checkpoint_interval 50 \\
    --max_new_tokens 512
        """
    )
    
    parser.add_argument(
        "--input_json", 
        type=str, 
        required=True,
        help="Path to Step 1 output JSON (evidence classification results)"
    )
    parser.add_argument(
        "--output_json", 
        type=str, 
        required=True,
        help="Path to save justification generation results"
    )
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="google/paligemma2-10b-pt-448",
        help="PaliGemma model ID"
    )
    parser.add_argument(
        "--cache_dir", 
        type=str, 
        default="image_cache",
        help="Directory for image caching"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=512,
        help="Maximum tokens to generate per justification"
    )
    parser.add_argument(
        "--checkpoint_interval", 
        type=int, 
        default=50,
        help="Save checkpoint every N samples"
    )
    
    args = parser.parse_args()
    
    process_justification_generation(
        input_json=args.input_json,
        output_json=args.output_json,
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        max_new_tokens=args.max_new_tokens,
        checkpoint_interval=args.checkpoint_interval
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 2: Multimodal Justification Generation using vLLM (Qwen2VL/Idefics3)

Takes evidence classification output from Step 1 and generates:
- Support justification (why claim is true based on supporting evidence)
- Refute justification (why claim is false based on refuting evidence)

Uses vLLM with AWQ quantization for efficient inference on MR2 dataset.
Supports: Qwen2VL-AWQ, Idefics3-AWQ
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
from vllm import LLM, SamplingParams
from transformers import AutoProcessor


# =============================
# Model Configuration
# =============================
MODEL_CONFIG = {
    "qwen2vl-awq": {
        "model_id": "Qwen/Qwen2-VL-7B-Instruct-AWQ",
        "max_new_tokens": 512,
        "image_size": 768,
        "max_model_len": 8192,
        "use_chat_template": True,
        "quantization": "awq",
        "dtype": "half",
    },
    "idefics3-awq": {
        "model_id": "ronantakizawa/idefics3-8b-llama3-awq",
        "max_new_tokens": 512,
        "image_size": 768,
        "max_model_len": 8192,
        "use_chat_template": True,
        "quantization": "awq",
        "dtype": "half",
    },
    # Non-quantized versions (if needed)
    "qwen2vl": {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "max_new_tokens": 512,
        "image_size": 768,
        "max_model_len": 10000,
        "use_chat_template": True,
    },
    "idefics3": {
        "model_id": "leon-se/Idefics3-8B-Llama3-bnb_nf4",  #HuggingFaceM4/Idefics3-8B-Llama3
        "max_new_tokens": 256,
        "image_size": 768,
        "max_model_len": 4096,
        "visual_token_count": 256,
        "use_chat_template": True,
    },
}


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


def download_image(url: str, max_size: int = 768, cache_dir: str = "image_cache") -> Optional[Image.Image]:
    """Download and cache images"""
    if not url or str(url).strip().lower() in {"", "nan", "none", "null"}:
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
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img.thumbnail((max_size, max_size))
        img.save(local_path)
        return img
    except Exception as e:
        print(f"[WARN] Image download failed: {url} | Error: {e}")
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
        print(f"[WARN] Image load failed: {path} ({e})")
        return None


def truncate_text(text: str, max_chars: int = 800) -> str:
    """Simple character-based truncation"""
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


# =============================
# Prompt Builders (NO OCR)
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
- Do NOT explicitly mention the label in your response
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
- Do NOT explicitly mention the label in your response
- If images contain contradictory details (objects, text, context), reference them naturally
- Focus only on key contradictory factors and logical reasoning
- Use clear, objective language
- Keep the explanation within 150 words

### Response:
"""
    return prompt


# =============================
# vLLM Model Wrapper
# =============================
class VLMJustificationGenerator:
    """vLLM wrapper for Qwen2VL and Idefics3 with quantization support"""
    
    def __init__(self, model_key: str):
        assert model_key in MODEL_CONFIG, f"Unknown model: {model_key}"
        
        self.model_key = model_key
        self.config = MODEL_CONFIG[model_key]
        
        print(f"[INFO] Loading {model_key}: {self.config['model_id']}")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.config["model_id"], 
            trust_remote_code=True
        )
        
        # Build vLLM kwargs
        llm_kwargs = {
            "model": self.config["model_id"],
            "max_model_len": self.config["max_model_len"],
            "trust_remote_code": True,
        }
        
        # Add quantization if specified
        if "quantization" in self.config:
            llm_kwargs["quantization"] = self.config["quantization"]
            llm_kwargs["dtype"] = self.config.get("dtype", "half")
            llm_kwargs["gpu_memory_utilization"] = 0.4
            print(f"[INFO] Using {self.config['quantization']} quantization")
        else:
            llm_kwargs["dtype"] = "bfloat16"
            llm_kwargs["gpu_memory_utilization"] = 0.4
        
        # Support multi-image input (claim + evidence)
        llm_kwargs["limit_mm_per_prompt"] = {"image": 2}
        
        # Initialize vLLM
        try:
            self.llm = LLM(**llm_kwargs)
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            print("[INFO] Retrying with reduced memory...")
            llm_kwargs["gpu_memory_utilization"] = 0.4
            self.llm = LLM(**llm_kwargs)
        
        # Sampling parameters for justification generation
        self.sampling_params = SamplingParams(
            temperature=0.1,  # Low but not zero for more natural text
            top_p=0.9,
            max_tokens=self.config["max_new_tokens"],
            skip_special_tokens=True,
        )
        
        print(f"[INFO] Model loaded successfully")
    
    def _format_prompt_with_chat_template(
        self, 
        prompt: str, 
        images: List[Image.Image]
    ) -> str:
        """Format prompt using model's chat template"""
        valid_imgs = [img for img in images if is_valid_image(img)]
        
        # Build message content
        content = []
        for _ in valid_imgs:
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})
        
        messages = [{"role": "user", "content": content}]
        
        return self.processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=False
        )
    
    def generate(
        self, 
        prompt: str, 
        images: List[Image.Image]
    ) -> str:
        """
        Generate justification for a single prompt + images.
        
        Args:
            prompt: Plain text prompt
            images: List of images (claim + evidence)
        
        Returns:
            Generated justification text
        """
        try:
            # Validate images
            valid_imgs = [img for img in images if is_valid_image(img)]
            
            # Format with chat template
            formatted_prompt = self._format_prompt_with_chat_template(prompt, valid_imgs)
            
            # Prepare vLLM request
            request = {
                "prompt": formatted_prompt,
                "multi_modal_data": {"image": valid_imgs} if valid_imgs else {}
            }
            
            # Generate
            outputs = self.llm.generate([request], self.sampling_params)
            response = outputs[0].outputs[0].text.strip()
            
            if not response or len(response) < 10:
                return "Unable to generate justification based on available evidence."
            
            return response
        
        except Exception as e:
            print(f"[ERROR] Generation failed: {e}")
            import traceback
            traceback.print_exc()
            return "ERROR: Generation failed"
    
    def generate_batch(
        self,
        prompts: List[str],
        images_lists: List[List[Image.Image]]
    ) -> List[str]:
        """
        Batch generation for efficiency.
        
        Args:
            prompts: List of plain text prompts
            images_lists: List of image lists (one per prompt)
        
        Returns:
            List of generated justifications
        """
        assert len(prompts) == len(images_lists)
        
        # Prepare requests
        requests = []
        for prompt, imgs in zip(prompts, images_lists):
            valid_imgs = [img for img in imgs if is_valid_image(img)]
            formatted_prompt = self._format_prompt_with_chat_template(prompt, valid_imgs)
            
            requests.append({
                "prompt": formatted_prompt,
                "multi_modal_data": {"image": valid_imgs} if valid_imgs else {}
            })
        
        try:
            # Batch generate
            outputs = self.llm.generate(requests, self.sampling_params)
            responses = [out.outputs[0].text.strip() for out in outputs]
            
            # Validate responses
            return [
                resp if resp and len(resp) >= 10 
                else "Unable to generate justification based on available evidence."
                for resp in responses
            ]
        
        except Exception as e:
            print(f"[ERROR] Batch generation failed: {e}")
            import traceback
            traceback.print_exc()
            return ["ERROR: Generation failed"] * len(prompts)


# =============================
# Main Processing Pipeline
# =============================
def process_justification_generation_mr2(
    input_json: str,
    output_json: str,
    model_key: str = "qwen2vl-awq",
    cache_dir: str = "image_cache",
    checkpoint_interval: int = 50,
    batch_size: int = 4
):
    """
    Main pipeline: Generate support and refute justifications for MR2 dataset.
    
    Args:
        input_json: Path to Step 1 output (evidence classification JSON)
        output_json: Path to save justification results
        model_key: Model to use (qwen2vl-awq, idefics3-awq, etc.)
        cache_dir: Image cache directory
        checkpoint_interval: Save every N samples
        batch_size: Number of samples to process together
    """
    
    # Load model
    print(f"\n[INFO] Initializing model: {model_key}")
    generator = VLMJustificationGenerator(model_key)
    image_size = generator.config["image_size"]
    
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
    
    # Determine base directory for MR2 images
    json_dir = os.path.dirname(input_json)
    mr2_base_dir = os.path.abspath(os.path.join(json_dir, ".."))
    
    # Process in batches
    for batch_start in tqdm(range(start_idx, len(data), batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, len(data))
        batch_data = data[batch_start:batch_end]
        
        batch_support_prompts = []
        batch_refute_prompts = []
        batch_images = []
        batch_metadata = []
        
        # Prepare batch
        for entry in batch_data:
            try:
                # Extract fields from Step 1 output
                claim_id = entry.get("claim_id", "")
                claim_text = entry.get("claim_text", "")
                
                # Evidence from Step 1
                supporting_text = entry.get("supporting_text", [])
                refuting_text = entry.get("refuting_text", [])
                visual_label = entry.get("visual_label")
                visual_response = entry.get("visual_response")
                
                # Image URLs (MR2 format: relative paths)
                claim_img_url = entry.get("claim_image_url", "")
                evidence_img_url = entry.get("evidence_image_url", "")
                
                # Load images
                claim_img = None
                evidence_img = None
                
                # MR2 paths are relative (e.g., "mr2/val/img/1303.jpg")
                if claim_img_url:
                    if claim_img_url.startswith('http'):
                        claim_img = download_image(claim_img_url, image_size, cache_dir)
                    else:
                        # Handle MR2 relative paths
                        if claim_img_url.startswith('mr2/'):
                            full_path = os.path.join(mr2_base_dir, claim_img_url)
                        else:
                            full_path = os.path.join(mr2_base_dir, claim_img_url)
                        claim_img = load_local_image(full_path, image_size)
                
                if evidence_img_url:
                    if evidence_img_url.startswith('http'):
                        evidence_img = download_image(evidence_img_url, image_size, cache_dir)
                    else:
                        if evidence_img_url.startswith('mr2/'):
                            full_path = os.path.join(mr2_base_dir, evidence_img_url)
                        else:
                            full_path = os.path.join(mr2_base_dir, evidence_img_url)
                        evidence_img = load_local_image(full_path, image_size)
                
                # Prepare images list (both claim and evidence)
                images = []
                if claim_img and is_valid_image(claim_img):
                    images.append(claim_img)
                if evidence_img and is_valid_image(evidence_img):
                    images.append(evidence_img)
                
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
                
                batch_support_prompts.append(support_prompt)
                batch_refute_prompts.append(refute_prompt)
                batch_images.append(images)
                batch_metadata.append({
                    "claim_id": claim_id,
                    "claim_text": claim_text
                })
            
            except Exception as e:
                print(f"[ERROR] Failed to prepare entry: {e}")
                batch_support_prompts.append("")
                batch_refute_prompts.append("")
                batch_images.append([])
                batch_metadata.append({
                    "claim_id": entry.get("claim_id", ""),
                    "claim_text": entry.get("claim_text", ""),
                    "error": str(e)
                })
        
        # Generate support justifications (batch)
        support_justifications = generator.generate_batch(
            batch_support_prompts,
            batch_images
        )
        
        # Generate refute justifications (batch)
        refute_justifications = generator.generate_batch(
            batch_refute_prompts,
            batch_images
        )
        
        # Compile results
        for i, meta in enumerate(batch_metadata):
            support_just = support_justifications[i] if i < len(support_justifications) else "ERROR"
            refute_just = refute_justifications[i] if i < len(refute_justifications) else "ERROR"
            
            result = {
                "claim_id": meta["claim_id"],
                "claim_text": meta["claim_text"],
                "support_justification": support_just,
                "refute_justification": refute_just,
            }
            
            if "error" in meta:
                result["error"] = meta["error"]
            
            results.append(result)
        
        # Checkpoint
        if (batch_start + batch_size) % checkpoint_interval == 0 or batch_end >= len(data):
            Path(output_json).parent.mkdir(parents=True, exist_ok=True)
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"[CHECKPOINT] Saved {len(results)} samples to {output_json}")
    
    # Final save
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[DONE] Saved {len(results)} results to {output_json}")
    
    # Print statistics
    print("\n[STATISTICS]")
    successful = len([r for r in results if "error" not in r])
    print(f"Successful: {successful}/{len(results)}")
    
    avg_support_len = np.mean([
        len(r.get("support_justification", "")) 
        for r in results if "error" not in r
    ])
    avg_refute_len = np.mean([
        len(r.get("refute_justification", "")) 
        for r in results if "error" not in r
    ])
    print(f"Average support justification length: {avg_support_len:.1f} chars")
    print(f"Average refute justification length: {avg_refute_len:.1f} chars")
    
    # Cleanup
    del generator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================
# CLI
# =============================
def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Generate Support/Refute Justifications using vLLM (Qwen2VL/Idefics3) for MR2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usage:

For Qwen2VL-AWQ:
  python justification_generation_vllm_mr2.py \\
    --input_json outputs/mr2_val_evidence_qwen2vl.json \\
    --output_json outputs/mr2_val_justifications_qwen2vl.json \\
    --model_key qwen2vl-awq \\
    --batch_size 4 \\
    --checkpoint_interval 50

For Idefics3-AWQ:
  python justification_generation_vllm_mr2.py \\
    --input_json outputs/mr2_val_evidence_idefics3.json \\
    --output_json outputs/mr2_val_justifications_idefics3.json \\
    --model_key idefics3-awq \\
    --batch_size 4 \\
    --checkpoint_interval 50
        """
    )
    
    parser.add_argument(
        "--input_json", 
        type=str, 
        required=True,
        help="Path to Step 1 output JSON (evidence classification results from MR2)"
    )
    parser.add_argument(
        "--output_json", 
        type=str, 
        required=True,
        help="Path to save justification generation results"
    )
    parser.add_argument(
        "--model_key", 
        type=str, 
        default="qwen2vl-awq",
        choices=list(MODEL_CONFIG.keys()),
        help="Model to use for generation"
    )
    parser.add_argument(
        "--cache_dir", 
        type=str, 
        default="image_cache",
        help="Directory for image caching"
    )
    parser.add_argument(
        "--checkpoint_interval", 
        type=int, 
        default=50,
        help="Save checkpoint every N samples"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=4,
        help="Batch size for generation (higher = faster but more memory)"
    )
    
    args = parser.parse_args()
    
    process_justification_generation_mr2(
        input_json=args.input_json,
        output_json=args.output_json,
        model_key=args.model_key,
        cache_dir=args.cache_dir,
        checkpoint_interval=args.checkpoint_interval,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()

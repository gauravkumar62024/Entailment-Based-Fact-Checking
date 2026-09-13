#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 2: Multimodal Justification Generation for MOCHEG Dataset

Generates support and refute justifications using evidence from Step 1.
Adapted for MOCHEG which has NO claim images - only document images.

Input:  JSON from Step 1 with evidence classification
Output: JSON with support_justification and refute_justification for each claim
"""

import json
import os
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoProcessor
from vllm import LLM, SamplingParams
from PIL import Image
import argparse
import numpy as np

# =============================
# Model Configuration
# =============================
MODELS_CONFIG = {
    "qwen2vl": {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "tokenizer_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "max_model_len": 10000,
        "max_new_tokens": 512,
        "image_size": 768,
        "use_chat_template": True
    },
    "idefics3": {
        "model_id": "HuggingFaceM4/Idefics3-8B-Llama3",
        "tokenizer_id": "HuggingFaceM4/Idefics3-8B-Llama3",
        "max_model_len": 10000,
        "max_new_tokens": 512,
        "image_size": 768,
        "use_chat_template": True
    },
    "paligemma": {
        "model_id": "google/paligemma2-3b-mix-448",
        "tokenizer_id": "google/paligemma2-3b-mix-448",
        "max_model_len": 8192,
        "max_new_tokens": 512,
        "image_size": 448,
        "use_chat_template": False  # PaliGemma: NO chat template
    }
}

# =============================
# Utility Functions
# =============================

def load_image(path: str, max_size: int):
    """Load image from local path (no URL)."""
    if not path or str(path).strip() == "" or not os.path.exists(path):
        return None
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((max_size, max_size))
        return img
    except Exception as e:
        print(f"[warn] Failed to open local image: {path} ({e})")
        return None


def is_valid_image(img):
    """Validate image before processing."""
    if img is None:
        return False
    try:
        arr = np.array(img)
        if len(arr.shape) != 3 or arr.shape[2] != 3:
            return False
        h, w, _ = arr.shape
        if h < 64 or w < 64 or h > 4096 or w > 4096:
            return False
        if np.std(arr) < 2 or np.any(np.isnan(arr)):
            return False
        return True
    except Exception:
        return False


def truncate_text(processor, text: str, max_length: int):
    """Truncate text to approximately max_length tokens."""
    if not text:
        return text
    tokenizer = getattr(processor, "tokenizer", processor)
    try:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= max_length:
            return text
        truncated_tokens = tokens[:max_length]
        return tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    except Exception:
        return text[:max_length * 4]  # Fallback: 4 chars per token estimate


def load_data(json_path: str):
    """Load JSON from Step 1."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# =============================
# VLM Setup
# =============================
def setup_vlm(model_key: str):
    """Setup Vision-Language Model with vLLM."""
    assert model_key in MODELS_CONFIG, f"Model {model_key} not in config"
    spec = MODELS_CONFIG[model_key]
    model_id = spec["model_id"]
    
    print(f"[INFO] Loading {model_key}: {model_id}")
    
    llm_kwargs = {
        "model": model_id,
        "dtype": "bfloat16",
        "max_model_len": spec["max_model_len"],
        "gpu_memory_utilization": 0.5,
        "trust_remote_code": True,
        "limit_mm_per_prompt": {"image": 1}  # MOCHEG: Only 1 image (document)
    }
    
    llm = LLM(**llm_kwargs)
    
    sampling_params = SamplingParams(
        temperature=0.01,
        top_p=0.9,
        repetition_penalty=1.05,
        max_tokens=spec["max_new_tokens"],
        skip_special_tokens=True
    )
    
    # Load processor
    try:
        processor = AutoProcessor.from_pretrained(
            spec["tokenizer_id"], 
            trust_remote_code=True
        )
    except Exception:
        processor = AutoTokenizer.from_pretrained(
            spec["tokenizer_id"],
            trust_remote_code=True
        )
    
    return llm, sampling_params, processor, spec


# =============================
# Prompt Building (MOCHEG-specific)
# =============================

# (KEEPING YOUR PROMPTS EXACTLY AS THEY WERE)

def build_support_prompt(
    claim_text: str,
    supporting_text: str,
    visual_evidence_label: str,
    visual_raw_response: str,
    processor,
    max_tokens: int = 2000
) -> str:
    """Build prompt for SUPPORT justification (label=True)."""
    claim_text = truncate_text(processor, claim_text, max_tokens // 3)
    supporting_text = truncate_text(processor, supporting_text, max_tokens // 2)
    
    visual_info = ""
    if visual_evidence_label == "SUPPORT":
        visual_info = f"**Visual Evidence Classification:** SUPPORT\n"
        if visual_raw_response:
            visual_info += f"**Visual Analysis:** {truncate_text(processor, visual_raw_response or '', 200)}\n"
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
{claim_text}

### Supporting Text Evidence:
{supporting_text}

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
    claim_text: str,
    refuting_text: str,
    visual_evidence_label: str,
    visual_raw_response: str,
    processor,
    max_tokens: int = 2000
) -> str:
    """Build prompt for REFUTE justification (label=False)."""
    claim_text = truncate_text(processor, claim_text, max_tokens // 3)
    refuting_text = truncate_text(processor, refuting_text, max_tokens // 2)
    
    visual_info = ""
    if visual_evidence_label == "REFUTE":
        visual_info = f"**Visual Evidence Classification:** REFUTE\n"
        if visual_raw_response:
            visual_info += f"**Visual Analysis:** {truncate_text(processor, visual_raw_response or '', 200)}\n"
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
{claim_text}

### Refuting Text Evidence:
{refuting_text}

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
# Generation Function
# =============================
def generate_justification(
    llm,
    sampling_params,
    processor,
    prompt: str,
    doc_img: Image.Image,
    model_key: str,
    spec: dict
) -> str:
    """Generate justification with single document image."""
    if not is_valid_image(doc_img):
        print(f"[warn] Invalid or missing document image, using text-only mode")
        doc_img = None
    
    if spec["use_chat_template"]:
        if doc_img:
            content = [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        else:
            content = [{"type": "text", "text": prompt}]
        
        messages = [{"role": "user", "content": content}]
        
        try:
            formatted_prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            print(f"[warn] Chat template failed: {e}, using direct prompt")
            formatted_prompt = prompt
    else:
        formatted_prompt = prompt
    
    if doc_img:
        request_dict = {"prompt": formatted_prompt, "multi_modal_data": {"image": doc_img}}
    else:
        request_dict = {"prompt": formatted_prompt}
    
    try:
        outputs = llm.generate([request_dict], sampling_params)
        response = outputs[0].outputs[0].text.strip()
        if not response:
            return "ERROR: No output generated"
        return response
    except Exception as e:
        print(f"[error] Generation failed: {e}")
        return f"ERROR: {e}"


# =============================
# Main Processing Pipeline
# =============================
def process_entries(
    data: list,
    llm,
    sampling_params,
    processor,
    model_key: str,
    spec: dict
):
    """Process each entry to generate support and refute justifications."""
    results = []
    
    for entry in tqdm(data, desc="Generating justifications"):
        try:
            ex_id = entry.get("Id", "")
            claim_text = entry.get("claim_text", "")
            
            # Use image path instead of URL
            doc_img_path = entry.get("document_image_path", "")
            
            # Evidence from Step 1
            supporting_text_list = entry.get("supporting_text_evidence", [])
            refuting_text_list = entry.get("refuting_text_evidence", [])
            
            # Visual evidence classification
            visual_evidence_label = entry.get("visual_evidence_label", "NO_EVIDENCE")
            visual_raw_response = entry.get("visual_evidence_raw_response", "")
            
            # Format text evidence
            supporting_text = " | ".join(supporting_text_list) if supporting_text_list else "No supporting evidence found."
            refuting_text = " | ".join(refuting_text_list) if refuting_text_list else "No refuting evidence found."
            
            # Load local image if available
            doc_img = load_image(doc_img_path, spec["image_size"]) if doc_img_path else None
            
            # Generate SUPPORT justification
            support_prompt = build_support_prompt(
                claim_text=claim_text,
                supporting_text=supporting_text,
                visual_evidence_label=visual_evidence_label,
                visual_raw_response=visual_raw_response,
                processor=processor
            )
            
            support_just = generate_justification(
                llm=llm,
                sampling_params=sampling_params,
                processor=processor,
                prompt=support_prompt,
                doc_img=doc_img,
                model_key=model_key,
                spec=spec
            )
            
            if "ERROR" in support_just:
                print(f"[warn] Support generation failed for {ex_id}")
            
            # Generate REFUTE justification
            refute_prompt = build_refute_prompt(
                claim_text=claim_text,
                refuting_text=refuting_text,
                visual_evidence_label=visual_evidence_label,
                visual_raw_response=visual_raw_response,
                processor=processor
            )
            
            refute_just = generate_justification(
                llm=llm,
                sampling_params=sampling_params,
                processor=processor,
                prompt=refute_prompt,
                doc_img=doc_img,
                model_key=model_key,
                spec=spec
            )
            
            if "ERROR" in refute_just:
                print(f"[warn] Refute generation failed for {ex_id}")
            
            # Compile result
            result = {
                "Id": ex_id,
                "claim_text": claim_text,
                "document_image_path": doc_img_path or "",
                "support_justification": support_just,
                "refute_justification": refute_just,
                "visual_evidence_label": visual_evidence_label
            }
            
            results.append(result)
        
        except Exception as e:
            print(f"[error] Failed processing entry {entry.get('Id', 'unknown')}: {e}")
            results.append({
                "Id": entry.get("Id", ""),
                "claim_text": entry.get("claim_text", ""),
                "support_justification": f"ERROR: {e}",
                "refute_justification": f"ERROR: {e}",
                "error": str(e)
            })
    
    return results


def process_split(
    split: str,
    input_json: str,
    output_json: str,
    model_key: str
):
    """Process one split (train/val/test)."""
    if not Path(input_json).exists():
        print(f"[skip] Input not found: {input_json}")
        return
    
    print(f"\n{'='*60}")
    print(f"Processing {split} split with {model_key}")
    print(f"{'='*60}")
    
    # Setup VLM
    llm, sampling_params, processor, spec = setup_vlm(model_key)
    
    try:
        # Load data
        data = load_data(input_json)
        print(f"[INFO] Loaded {len(data)} entries from {input_json}")
        
        # Process entries
        results = process_entries(
            data=data,
            llm=llm,
            sampling_params=sampling_params,
            processor=processor,
            model_key=model_key,
            spec=spec
        )
        
        # Save results
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"[DONE] Saved {len(results)} results to {output_json}")
    
    finally:
        # Cleanup
        del llm, sampling_params
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =============================
# CLI
# =============================
def main():
    parser = argparse.ArgumentParser(
        description="MOCHEG Justification Generation (Step 2, Local Image Path)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usage:
  python justification_generation_mocheg.py \\
    --input_json outputs/val_predictions_with_paths.json \\
    --output_json outputs/val_justifications_qwen2vl.json \\
    --model_key qwen2vl
        """
    )
    
    parser.add_argument("--input_json", type=str, required=True,
                        help="Path to Step 1 output JSON (evidence classification)")
    parser.add_argument("--output_json", type=str, required=True,
                        help="Path to save justifications JSON")
    parser.add_argument("--model_key", type=str, required=True,
                        choices=list(MODELS_CONFIG.keys()),
                        help="Model to use")
    
    args = parser.parse_args()
    
    # Extract split name from input path
    input_name = Path(args.input_json).stem
    split = input_name.split("_")[0] if "_" in input_name else "unknown"
    
    process_split(
        split=split,
        input_json=args.input_json,
        output_json=args.output_json,
        model_key=args.model_key
    )


if __name__ == "__main__":
    main()


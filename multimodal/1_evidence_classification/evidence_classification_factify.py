import os
import re
import csv
import json
import argparse
from io import BytesIO
from typing import List, Dict, Any, Tuple
import hashlib
import sys

import requests
from PIL import Image
from tqdm import tqdm
import torch
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

# Allow very large CSV fields
csv.field_size_limit(sys.maxsize)

# -----------------------------
# Supported VLMs
# -----------------------------
MODEL_MAP = {
    "idefics2": {
        "model_id": "HuggingFaceM4/Idefics3-8B-Llama3",
        "loader": "vision2seq",
        "max_new_tokens": 256,
        "image_size": 768,
        "max_model_len": 10000,
        "truncate_to": 9990
    },
    "paligemma": {
        "model_id": "google/paligemma2-10b-mix-448",
        "loader": "paligemma",
        "max_new_tokens": 256,
        "image_size": 768,
        "max_model_len": 8192,
        "truncate_to": 8190
    },
    "qwen2vl": {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "loader": "vision2seq",
        "max_new_tokens": 256,
        "image_size": 768,
        "max_model_len": 10000,
        "truncate_to": 9999
    }
}

# -----------------------------
# Utils
# -----------------------------

def download_image(url: str, max_size=768, cache_dir="image_cache"):
    """Download image with local caching for faster reuse."""
    if not url or str(url).strip().lower() in {"", "nan", "none"}:
        return None
    os.makedirs(cache_dir, exist_ok=True)
    url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()
    ext = os.path.splitext(url)[-1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        ext = ".jpg"
    local_path = os.path.join(cache_dir, f"{url_hash}{ext}")
    if os.path.exists(local_path):
        try:
            img = Image.open(local_path).convert("RGB")
            img.thumbnail((max_size, max_size))
            return img
        except Exception:
            os.remove(local_path)
    try:
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        img.thumbnail((max_size, max_size))
        img.save(local_path)
        return img
    except Exception as e:
        print(f"[warn] image fetch failed: {url} ({e})")
        return None


_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9""\(\[])')

def simple_sent_split(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    chunks = _SENT_SPLIT_RE.split(text)
    sents = [re.sub(r'\s+', ' ', c).strip() for c in chunks]
    return [s for s in sents if len(s) > 2]


def truncate_prompt(processor_or_tokenizer, text: str, max_length: int) -> str:
    """Truncate text to max_length tokens. Handles processor or direct tokenizer."""
    if not text:
        return text
    # Get the actual tokenizer (fallback if input is already tokenizer)
    tokenizer = getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_length:
        return text
    truncated_tokens = tokens[:max_length]
    truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    return truncated_text


# -----------------------------
# Classification helpers
# -----------------------------

_CLF_RE = re.compile(r"(SUPPORT|REFUTE)", re.IGNORECASE)

def parse_label(resp: str) -> str:
    """Extract SUPPORT or REFUTE from model response."""
    m = _CLF_RE.search(resp or "")
    if not m:
        return "REFUTE"
    return m.group(1).upper()


def build_text_evidence_prompt(
    claim_text: str,
    claim_ocr: str,
    evidence_sentence: str,
    processor,
    has_claim_img: bool = False,
    max_prompt_tokens: int = 2000,
) -> str:
    """
    Prompt for classifying a TEXT sentence from the document.
    Only uses claim image if available - NO document image here.
    """
    # Truncate using passed processor
    claim_text = truncate_prompt(processor, claim_text, max_prompt_tokens // 3)
    claim_ocr = truncate_prompt(processor, claim_ocr, max_prompt_tokens // 3)
    evidence_sentence = truncate_prompt(processor, evidence_sentence, max_prompt_tokens // 3)
    
    context = ""
    if has_claim_img:
        context = "An image of the claim is provided for additional context.\n"
    
    return (
        "You are a fact-checking assistant that classifies textual evidence.\n\n"
        "### Task:\n"
        "Determine whether the evidence sentence SUPPORTS or REFUTES the given claim.\n\n"
        "### Instructions:\n"
        "- Focus on the textual content and its logical relationship to the claim.\n"
        "- Consider any OCR text extracted from the claim image if provided.\n\n"
        f"{context}"
        f"**CLAIM:** {claim_text}\n"
        f"**CLAIM OCR:** {claim_ocr}\n\n"
        f"**EVIDENCE SENTENCE:** {evidence_sentence}\n\n"
        "### Response:\n"
        "Reply with exactly one word: SUPPORT or REFUTE\n"
    )


def build_visual_evidence_prompt(
    claim_text: str,
    claim_ocr: str,
    doc_ocr: str,
    processor,
    max_prompt_tokens: int = 2000,
) -> str:
    """
    Prompt for classifying the DOCUMENT IMAGE as visual evidence.
    Uses both claim image and document image.
    """
    # Truncate using passed processor
    claim_text = truncate_prompt(processor, claim_text, max_prompt_tokens // 3)
    claim_ocr = truncate_prompt(processor, claim_ocr, max_prompt_tokens // 3)
    doc_ocr = truncate_prompt(processor, doc_ocr, max_prompt_tokens // 3)
    
    return (
        "You are a multimodal fact-checking assistant that analyzes visual evidence.\n\n"
        "### Task:\n"
        "Determine whether the document image (provided below) SUPPORTS or REFUTES the claim.\n\n"
        "### Instructions:\n"
        "- Analyze the visual content of the document image.\n"
        "- Compare it with the claim and claim image (if provided).\n"
        "- Use OCR text extracted from both images to aid your analysis.\n"
        "- Focus on visual elements: photos, graphics, text style, layout, context.\n\n"
        f"**CLAIM:** {claim_text}\n"
        f"**CLAIM OCR:** {claim_ocr}\n"
        f"**DOCUMENT IMAGE OCR:** {doc_ocr}\n\n"
        "### Response:\n"
        "Reply with exactly one word: SUPPORT or REFUTE\n"
    )


def load_vlm(model_key: str, device: str = None, quantize: str = None):
    """Load Vision-Language Model with vLLM."""
    assert model_key in MODEL_MAP
    spec = MODEL_MAP[model_key]
    model_id = spec["model_id"]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[info] loading {model_key}: {model_id} with vLLM")

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    llm_kwargs = {
        "model": model_id,
        "dtype": "bfloat16" if device == "cuda" else "float32",
        "max_model_len": spec["max_model_len"],
        "gpu_memory_utilization": 0.6,
        "trust_remote_code": True,
    }
    if quantize:
        llm_kwargs["quantization"] = quantize
    if device == "cuda":
        llm_kwargs["limit_mm_per_prompt"] = {"image": 2}

    llm = LLM(**llm_kwargs)
    return llm, spec["max_new_tokens"], processor


def classify_with_vlm(
    llm,
    processor,
    max_new_tokens: int,
    plain_prompts: List[str],
    images_lists: List[List[Image.Image]],
    model_key: str,
    max_model_len: int,
) -> List[str]:
    """
    Batched VLM inference with vLLM.
    plain_prompts: List of plain prompts
    images_lists: List of list of PIL Images per prompt
    """
    assert len(plain_prompts) == len(images_lists)
    
    model_name = model_key.lower()  # e.g., 'qwen2vl'

    request_dicts = []
    for plain_prompt, imgs in zip(plain_prompts, images_lists):
        valid_imgs = [img for img in imgs if img is not None]
        
        # Build messages based on model
        if "qwen" in model_name or "idefics" in model_name:
            content = [{"type": "image"} for _ in valid_imgs] + [{"type": "text", "text": plain_prompt}]
            messages = [{"role": "user", "content": content}]
        else:  # paligemma, simple
            # For PaliGemma, adjust if needed; here use similar
            content = [{"type": "image"} for _ in valid_imgs] + [{"type": "text", "text": plain_prompt}]
            messages = [{"role": "user", "content": content}]

        formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

# ✨ Truncate formatted prompt if exceeds limit (model-agnostic)
        max_prompt_tokens = max_model_len - max_new_tokens - 100
        if max_prompt_tokens < 0:
            max_prompt_tokens = max_model_len // 2

        try:
            # Some processors (like Qwen2VL) are themselves the tokenizer
            tokenizer = getattr(processor, "tokenizer", processor)
            tokenized = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False)
            num_tokens = tokenized["input_ids"].shape[-1]
            if num_tokens > max_prompt_tokens:
                print(f"[warn] Prompt too long ({num_tokens} tokens). Truncating to {max_prompt_tokens}.")
                truncated_ids = tokenized["input_ids"][0][:max_prompt_tokens]
                formatted_prompt = tokenizer.decode(truncated_ids, skip_special_tokens=True)
        except Exception as e:
            print(f"[warn] Token length check failed: {e}")

        
        request_dicts.append({
            "prompt": formatted_prompt,
            "multi_modal_data": {"image": valid_imgs}
        })

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
        skip_special_tokens=True,
    )

    outputs = llm.generate(request_dicts, sampling_params)
    responses = [out.outputs[0].text.strip() for out in outputs]
    return responses


# -----------------------------
# Core processing per split
# -----------------------------
def process_split(
    csv_path: str,
    out_json: str,
    model_key: str,
    cache_dir: str,
    max_sents: int = 40,
    quantize: str = None,
    checkpoint_interval: int = 500,
):
    """
    Main processing function that classifies:
    1. Text evidence (document sentences)
    2. Visual evidence (document image)
    """
    llm, max_new_tokens, processor = load_vlm(model_key, quantize=quantize)
    spec = MODEL_MAP[model_key]
    image_size = spec["image_size"]
    max_model_len = spec["max_model_len"]

    rows: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Load existing results for resume
    results = []
    start_idx = 0
    if os.path.exists(out_json):
        print(f"[info] Loading existing results from {out_json}")
        with open(out_json, "r", encoding="utf-8") as f:
            results = json.load(f)
        start_idx = len(results)
        print(f"[info] Resuming from index {start_idx}")

    skipped = []
    for i, row in enumerate(tqdm(rows[start_idx:], desc=f"Processing {os.path.basename(csv_path)} from {start_idx}")):
        global_idx = start_idx + i
        try:
            ex_id = str(row.get("Id", "")).strip()
            claim_text = (row.get("claim") or "").strip()
            claim_ocr = (row.get("claim_ocr") or "").strip()
            doc_text = (row.get("document") or "").strip()
            doc_ocr = (row.get("document_ocr") or "").strip()
            claim_img_url = (row.get("claim_image") or "").strip()
            doc_img_url = (row.get("document_image") or "").strip()

            # Download images with model-specific size
            claim_img = download_image(claim_img_url, max_size=image_size, cache_dir=cache_dir)
            doc_img = download_image(doc_img_url, max_size=image_size, cache_dir=cache_dir)

            # NEW: Early check for overly long claim/doc (skip if >10k tokens)
            skip_reason = None
            tokenizer = getattr(processor, "tokenizer", processor)
            claim_tokens = len(tokenizer.encode(claim_text + claim_ocr, add_special_tokens=False))
            doc_tokens = len(tokenizer.encode(doc_text + doc_ocr, add_special_tokens=False))
            if claim_tokens > 10000 or doc_tokens > 10000:
                skip_reason = f"Long prompt (claim: {claim_tokens}t, doc: {doc_tokens}t)"
                print(f"[skip] {ex_id}: {skip_reason}")
                results.append({
                    "Id": ex_id,
                    "skipped": True,
                    "skip_reason": skip_reason,
                    "claim_text": claim_text,
                    "claim_ocr": claim_ocr,
                    "claim_image_url": claim_img_url,
                    "document_image_url": doc_img_url,
                    "supporting_text_evidence": [],
                    "refuting_text_evidence": [],
                    "text_evidence_predictions": [],
                    "visual_evidence_label": None,
                    "visual_evidence_raw_response": None,
                })
                continue

            # Split document text into sentences
            evidence_blob = "\n".join([doc_text, doc_ocr]).strip()
            sentences = simple_sent_split(evidence_blob)[:max_sents]

            # ============================================
            # PART 1: Classify TEXT evidence (sentences)
            # ============================================
            supporting_text = []
            refuting_text = []
            text_predictions = []

            if sentences:
                plain_prompts = [
                    build_text_evidence_prompt(
                        claim_text=claim_text,
                        claim_ocr=claim_ocr,
                        evidence_sentence=sent,
                        processor=processor,
                        has_claim_img=claim_img is not None,
                    )
                    for sent in sentences
                ]
                images_lists = [[claim_img]] * len(plain_prompts) if claim_img else [[] for _ in plain_prompts]

                batched_responses = classify_with_vlm(
                    llm=llm,
                    processor=processor,
                    max_new_tokens=max_new_tokens,
                    plain_prompts=plain_prompts,
                    images_lists=images_lists,
                    model_key=model_key,
                    max_model_len=max_model_len,
                )

                for sent, response in zip(sentences, batched_responses):
                    label = parse_label(response)
                    
                    if label == "SUPPORT":
                        supporting_text.append(sent)
                    else:
                        refuting_text.append(sent)

                    text_predictions.append({
                        "sentence": sent,
                        "prediction": label,
                        "raw_response": response,
                    })
            else:
                supporting_text = []
                refuting_text = []
                text_predictions = []

            # ============================================
            # PART 2: Classify VISUAL evidence (doc image)
            # ============================================
            visual_evidence_label = None
            visual_evidence_raw = None

            if doc_img is not None:
                plain_prompt = build_visual_evidence_prompt(
                    claim_text=claim_text,
                    claim_ocr=claim_ocr,
                    doc_ocr=doc_ocr,
                    processor=processor,
                )

                images_for_visual = [doc_img]
                if model_key != "idefics2" and claim_img:
                    images_for_visual.insert(0, claim_img)

                visual_responses = classify_with_vlm(
                    llm=llm,
                    processor=processor,
                    max_new_tokens=max_new_tokens,
                    plain_prompts=[plain_prompt],
                    images_lists=[images_for_visual],
                    model_key=model_key,
                    max_model_len=max_model_len,
                )

                visual_evidence_raw = visual_responses[0]
                visual_evidence_label = parse_label(visual_evidence_raw)

            # ============================================
            # Compile results
            # ============================================
            results.append({
                "Id": ex_id,
                "claim_text": claim_text,
                "claim_ocr": claim_ocr,
                "claim_image_url": claim_img_url,
                "document_image_url": doc_img_url,
                
                # Textual evidence
                "supporting_text_evidence": supporting_text,
                "refuting_text_evidence": refuting_text,
                "text_evidence_predictions": text_predictions,
                
                # Visual evidence
                "visual_evidence_label": visual_evidence_label,
                "visual_evidence_raw_response": visual_evidence_raw,
            })

            # Checkpoint every interval
            if (global_idx + 1) % checkpoint_interval == 0 or (global_idx + 1) == len(rows):
                if os.path.dirname(out_json):
                    os.makedirs(os.path.dirname(out_json), exist_ok=True)
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"[checkpoint] Saved {global_idx + 1} samples to {out_json}")

        except Exception as e:
            ex_id = str(row.get("Id", "unknown")).strip()
            print(f"[error] Skipping {ex_id}: {e}")
            skipped.append({
                "Id": ex_id,
                "error": str(e),
                "row": row
            })
            # Append skipped entry to results for consistency
            results.append({
                "Id": ex_id,
                "skipped": True,
                "error": str(e),
                "claim_text": "",
                "claim_ocr": "",
                "claim_image_url": "",
                "document_image_url": "",
                "supporting_text_evidence": [],
                "refuting_text_evidence": [],
                "text_evidence_predictions": [],
                "visual_evidence_label": None,
                "visual_evidence_raw_response": None,
            })
            continue

    # Save final results
    if os.path.dirname(out_json):
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[done] wrote {out_json} ({len(results)} items)")

    # Save skipped
    if skipped:
        skipped_json = out_json.replace(".json", "_skipped.json")
        with open(skipped_json, "w", encoding="utf-8") as f:
            json.dump(skipped, f, indent=2, ensure_ascii=False)
        print(f"[done] wrote skipped to {skipped_json} ({len(skipped)} items)")


# -----------------------------
# CLI
# -----------------------------
def main():
    p = argparse.ArgumentParser(
        description="Multimodal Evidence Classification: Separate text and visual evidence"
    )
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--model_key", type=str, default="qwen2vl", choices=list(MODEL_MAP.keys()))
    p.add_argument("--max_sents", type=int, default=40)
    p.add_argument("--cache_dir", type=str, default="image_cache")
    p.add_argument("--quantize", type=str, default=None, help="Quantization method, e.g., 'awq'")
    p.add_argument("--checkpoint_interval", type=int, default=500, help="Save checkpoint every N samples")
    args = p.parse_args()

    splits = {
        #"train": os.path.join(args.data_dir, "train.csv"),
        #"val":   os.path.join(args.data_dir, "val.csv"),
        "test":  os.path.join(args.data_dir, "test.csv"),
    }

    for split, csv_path in splits.items():
        if not os.path.exists(csv_path):
            print(f"[skip] {csv_path} not found")
            continue
        out_json = os.path.join(args.out_dir, f"{split}_evidence_{args.model_key}.json")
        process_split(
            csv_path,
            out_json,
            model_key=args.model_key,
            cache_dir=args.cache_dir,
            max_sents=args.max_sents,
            quantize=args.quantize,
            checkpoint_interval=args.checkpoint_interval,
        )

if __name__ == "__main__":
    main()

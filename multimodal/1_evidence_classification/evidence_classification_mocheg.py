#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MOCHEG unified loader + classifier (text + image) with proper vLLM/HF support.

Usage:
  python evidence_classification.py \
      --data_root /path/to/MOCHEG \
      --split val \
      --model_family paligemma \
      --hf_model_id google/paligemma2-3b-mix-448 \
      --output_json /path/to/out.json \
      --max_text_per_claim 20 \
      --batch_size 4

Model families supported:
  - paligemma  (no chat template, 1 image per prompt)
  - qwen2vl    (chat template, 1 image per prompt in MOCHEG)
  - idefics3   (chat template, 1 image per prompt in MOCHEG)
"""

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Any, Tuple

import pandas as pd
from PIL import Image

# -------- Try vLLM first; fallback to HF --------
_USE_VLLM = False
try:
    from vllm import LLM, SamplingParams
    _USE_VLLM = True
except Exception:
    _USE_VLLM = False

import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM


# ----------------------- Prompts -----------------------

TEXT_PROMPT = (
    "You are a fact-checking assistant that classifies textual evidence.\n\n"
    "Task:\n"
    "Determine whether the evidence sentence SUPPORTS or REFUTES the given claim.\n\n"
    "Instructions:\n"
    "- Focus on the textual content and its logical relationship to the claim.\n"
    "- Consider only the factual meaning of the evidence and claim.\n"
    "- Ignore irrelevant or stylistic differences between them.\n\n"
    "CLAIM: {claim}\n\n"
    "EVIDENCE SENTENCE: {sentence}\n\n"
    "Response:\n"
    "Reply with exactly one word: SUPPORT or REFUTE\n\n"
    "Answer:"
)

IMAGE_PROMPT = (
    "You are a multimodal fact-checking assistant that analyzes visual evidence.\n\n"
    "Task:\n"
    "Determine whether the image SUPPORTS or REFUTES the given claim.\n\n"
    "Instructions:\n"
    "- Analyze the visual content of the provided image.\n"
    "- Compare it carefully with the claim to assess factual consistency.\n"
    "- Focus on visual elements such as objects, scenes, or text appearing in the image.\n"
    "- If the image is unrelated or contradicts the claim, classify it as REFUTE.\n\n"
    "CLAIM: {claim}\n\n"
    "Response:\n"
    "Reply with exactly one word: SUPPORT or REFUTE\n\n"
    "Answer:"
)

LABEL_RE = re.compile(r"\b(SUPPORT|REFUTE)\b", flags=re.IGNORECASE)


def parse_label(s: str) -> str:
    if not s:
        return "REFUTE"
    m = LABEL_RE.search(s)
    if not m:
        return "REFUTE"
    lab = m.group(1).upper()
    return "SUPPORT" if lab == "SUPPORT" else "REFUTE"


# ----------------------- Data Loading -----------------------

def load_split(
    data_root: str,
    split: str,
    supplementary_sentence_path: str = None,
    max_text_per_claim: int = None
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Returns:
      claims: {claim_id: {"claim_text": str}}
      text_evidence_by_claim: {claim_id: [sentences...]}
      image_paths_by_claim: {claim_id: [image_filepaths...]}
    """
    split_dir = os.path.join(data_root, split)
    corpus2_path = os.path.join(split_dir, "Corpus2.csv")
    qrels_sent_path = os.path.join(split_dir, "text_evidence_qrels_sentence_level.csv")
    img_qrels_path = os.path.join(split_dir, "img_evidence_qrels.csv")
    images_dir = os.path.join(split_dir, "images")

    if supplementary_sentence_path is None:
        supplementary_sentence_path = os.path.join(
            data_root, "supplementary", "Corpus3_sentence_level.csv"
        )

    assert os.path.exists(corpus2_path), f"Missing {corpus2_path}"
    assert os.path.exists(qrels_sent_path), f"Missing {qrels_sent_path}"
    assert os.path.exists(img_qrels_path), f"Missing {img_qrels_path}"
    assert os.path.exists(supplementary_sentence_path), f"Missing {supplementary_sentence_path}"

    # 1) Claims from Corpus2
    c2 = pd.read_csv(corpus2_path, dtype=str).fillna("")
    claims: Dict[str, Dict[str, Any]] = {}
    for _, row in c2.iterrows():
        cid = str(row.get("claim_id", "")).strip()
        if not cid:
            continue
        if cid not in claims:
            claims[cid] = {"claim_text": str(row.get("Claim", "")).strip()}

    # 2) Sentence-level qrels -> pick relevant sentences
    qrels = pd.read_csv(qrels_sent_path, dtype=str).fillna("")
    qrels = qrels[qrels["RELEVANCY"].astype(str) == "1"]
    sents_df = pd.read_csv(supplementary_sentence_path, dtype=str).fillna("")

    merged = qrels.merge(sents_df, left_on="DOCUMENT#", right_on="corpus_id", how="left")
    text_evidence_by_claim: Dict[str, List[str]] = defaultdict(list)
    for _, row in merged.iterrows():
        cid = str(row.get("TOPIC", "")).strip()
        sent = str(row.get("paragraph", "")).strip()
        if cid and sent:
            text_evidence_by_claim[cid].append(sent)

    # Optional cap per claim
    if max_text_per_claim is not None and max_text_per_claim > 0:
        for k in list(text_evidence_by_claim.keys()):
            text_evidence_by_claim[k] = text_evidence_by_claim[k][:max_text_per_claim]

    # 3) Image evidence
    img_qrels = pd.read_csv(img_qrels_path, dtype=str).fillna("")
    img_qrels = img_qrels[img_qrels["RELEVANCY"].astype(str) == "1"]

    image_paths_by_claim: Dict[str, List[str]] = defaultdict(list)
    for _, row in img_qrels.iterrows():
        cid = str(row.get("TOPIC", "")).strip()
        fname = str(row.get("DOCUMENT#", "")).strip()
        if not cid or not fname:
            continue
        cand_paths = [
            os.path.join(images_dir, fname),
            os.path.join(data_root, "images", fname),
        ]
        chosen = None
        for p in cand_paths:
            if os.path.exists(p):
                chosen = p
                break
        if chosen:
            image_paths_by_claim[cid].append(chosen)

    # Make sure every claim has entries
    for cid in list(claims.keys()):
        text_evidence_by_claim.setdefault(cid, [])
        image_paths_by_claim.setdefault(cid, [])

    return claims, text_evidence_by_claim, image_paths_by_claim


# ----------------------- Model Wrapper -----------------------

class VLMWrapper:
    """
    Wrapper for vLLM/HF inference with proper multimodal support.
    Handles PaliGemma (no chat template) vs Qwen2VL/Idefics3 (chat template).
    """

    def __init__(self, family: str, hf_model_id: str, use_vllm: bool, max_new_tokens=64):
        self.family = family.lower()
        self.hf_model_id = hf_model_id
        self.use_vllm = use_vllm
        self.max_new_tokens = max_new_tokens

        # Load processor AND tokenizer (needed for both vLLM and HF)
        self.processor = AutoProcessor.from_pretrained(
            hf_model_id, trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_model_id, trust_remote_code=True
        )

        if use_vllm:
            print(f"[INFO] Initializing vLLM for {family}")
            self.sampler = SamplingParams(
                max_tokens=max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                stop=["<|endoftext|>", "</s>", "<|im_end|>"]  # Common stop tokens
            )
            
            # Model-specific configs
            if family == "paligemma":
                max_model_len = 8192
            elif family == "qwen2vl":
                max_model_len = 10000
            else:  # idefics3
                max_model_len = 10000
            
            self.llm = LLM(
                model=hf_model_id,
                dtype="bfloat16",  # More stable than float16
                max_model_len=max_model_len,
                gpu_memory_utilization=0.7,
                trust_remote_code=True,
                limit_mm_per_prompt={"image": 1}  # MOCHEG: single image only
            )
            self.model = None
        else:
            print(f"[INFO] Initializing HF transformers for {family}")
            self.model = AutoModelForCausalLM.from_pretrained(
                hf_model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            self.model.eval()
            self.device = next(self.model.parameters()).device

    def _format_text_prompt(self, prompt: str) -> str:
        """
        Format text-only prompt based on model family.
        This is critical for proper text generation.
        """
        if self.family == "paligemma":
            # PaliGemma: Direct prompt, no special formatting
            return prompt
        
        elif self.family == "qwen2vl":
            # Qwen2VL: Use chat template for text-only
            messages = [{
                "role": "user",
                "content": prompt
            }]
            return self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        
        elif self.family == "idefics3":
            # Idefics3: Use chat template
            messages = [{
                "role": "user", 
                "content": [{"type": "text", "text": prompt}]
            }]
            return self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        
        else:
            return prompt

    def generate_text_only(self, prompts: List[str]) -> List[str]:
        """Generate from text-only prompts (for text evidence classification)."""
        
        # Format prompts based on model family
        formatted_prompts = [self._format_text_prompt(p) for p in prompts]
        
        if self.use_vllm:
            try:
                outs = self.llm.generate(formatted_prompts, self.sampler)
                results = []
                for out in outs:
                    if out.outputs:
                        text = out.outputs[0].text.strip()
                        results.append(text)
                    else:
                        results.append("")
                return results
            except Exception as e:
                print(f"[ERROR] vLLM text generation failed: {e}")
                return [""] * len(prompts)
        else:
            results = []
            for formatted_p, original_p in zip(formatted_prompts, prompts):
                try:
                    inputs = self.tokenizer(
                        formatted_p, 
                        return_tensors="pt",
                        truncation=True,
                        max_length=2048
                    ).to(self.device)
                    
                    with torch.no_grad():
                        out = self.model.generate(
                            **inputs,
                            max_new_tokens=self.max_new_tokens,
                            do_sample=False,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                    
                    # Decode only the new tokens (skip input)
                    generated_ids = out[0][inputs.input_ids.shape[1]:]
                    text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                    
                    results.append(text)
                except Exception as e:
                    print(f"[ERROR] HF text generation failed: {e}")
                    results.append("")
            
            return results

    def generate_with_image(self, image: Image.Image, prompt: str) -> str:
        """
        Generate from single image + text prompt.
        Handles PaliGemma (direct prompt) vs Qwen2VL/Idefics3 (chat template).
        """
        if not image:
            return "ERROR: No image provided"

        if self.use_vllm:
            # Build prompt based on model family
            if self.family == "paligemma":
                # PaliGemma: NO chat template, direct prompt
                formatted_prompt = prompt
            else:
                # Qwen2VL/Idefics3: Use chat template
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }]
                formatted_prompt = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

            # vLLM multimodal request
            request = {
                "prompt": formatted_prompt,
                "multi_modal_data": {"image": image}
            }
            
            try:
                outs = self.llm.generate([request], self.sampler)
                return outs[0].outputs[0].text.strip()
            except Exception as e:
                print(f"[ERROR] vLLM generation failed: {e}")
                return f"ERROR: {e}"
        
        else:
            # HF transformers fallback
            try:
                inputs = self.processor(
                    text=prompt,
                    images=image,
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False
                    )
                
                result = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
                return result.strip()
            except Exception as e:
                print(f"[ERROR] HF generation failed: {e}")
                return f"ERROR: {e}"


# ----------------------- Classification Pipeline -----------------------

def classify_text_evidence(
    vlm: VLMWrapper,
    claim_text: str,
    sentences: List[str],
    batch_size: int = 4
) -> List[Dict[str, str]]:
    """Classify text evidence sentences in batches."""
    preds = []
    buf = []
    idxs = []
    
    for i, sent in enumerate(sentences):
        p = TEXT_PROMPT.format(claim=claim_text, sentence=sent)
        buf.append(p)
        idxs.append(i)
        
        if len(buf) >= batch_size:
            outs = vlm.generate_text_only(buf)
            for j, out in zip(idxs, outs):
                preds.append({
                    "sentence": sentences[j],
                    "prediction": parse_label(out),
                    "raw_response": out
                })
            buf, idxs = [], []
    
    # Process remaining
    if buf:
        outs = vlm.generate_text_only(buf)
        for j, out in zip(idxs, outs):
            preds.append({
                "sentence": sentences[j],
                "prediction": parse_label(out),
                "raw_response": out
            })
    
    return preds


def classify_image_evidence(
    vlm: VLMWrapper,
    claim_text: str,
    image_paths: List[str]
) -> Tuple[str, str]:
    """
    Classify visual evidence (support/refute) given one or more images.
    Returns: (label, raw_response)
    """
    if not image_paths:
        return "NO_EVIDENCE", "No image evidence available"

    results = []
    prompt = IMAGE_PROMPT.format(claim=claim_text)
    
    for path in image_paths:
        try:
            # Load image as PIL
            image = Image.open(path).convert("RGB")
            response = vlm.generate_with_image(image, prompt)
            results.append(response)
        except Exception as e:
            print(f"[ERROR] Failed on image {path}: {e}")
            continue

    if not results:
        return "NO_EVIDENCE", "No valid responses"

    # Majority vote or first valid decision
    joined = " ".join(results).lower()
    
    if "support" in joined and "refute" not in joined:
        label = "SUPPORT"
    elif "refute" in joined:
        label = "REFUTE"
    else:
        label = "REFUTE"  # Default to refute if ambiguous
    
    return label, results[0] if results else "No response"


# ----------------------- Main Pipeline -----------------------

def run_pipeline(
    data_root: str,
    split: str,
    hf_model_id: str,
    model_family: str,
    output_json: str,
    batch_size: int = 4,
    max_text_per_claim: int = 20,
    use_vllm: bool = None,
    max_new_tokens: int = 64
):
    if use_vllm is None:
        use_vllm = _USE_VLLM

    print(f"[INFO] Loading split={split} from {data_root}")
    claims, text_evi, image_evi = load_split(
        data_root=data_root,
        split=split,
        max_text_per_claim=max_text_per_claim
    )
    
    claim_ids = list(claims.keys())
    
    print(f"[INFO] Found {len(claim_ids)} claims in {split}")

    print(f"[INFO] Initializing model: family={model_family}, id={hf_model_id}, "
          f"backend={'vLLM' if use_vllm else 'HF'}")
    vlm = VLMWrapper(
        family=model_family,
        hf_model_id=hf_model_id,
        use_vllm=use_vllm,
        max_new_tokens=max_new_tokens
    )

    outputs = []

    for idx, cid in enumerate(claim_ids):
        if (idx + 1) % 10 == 0:
            print(f"[{idx+1}/{len(claim_ids)}] Processing claim {cid}")
        
        ctext = claims[cid]["claim_text"]
        text_sents = text_evi.get(cid, [])
        img_paths = image_evi.get(cid, [])

        # Text classification
        text_preds = classify_text_evidence(vlm, ctext, text_sents, batch_size=batch_size)
        supporting_texts = [t["sentence"] for t in text_preds if t["prediction"] == "SUPPORT"]
        refuting_texts = [t["sentence"] for t in text_preds if t["prediction"] == "REFUTE"]

        # Image classification
        v_label, v_raw = classify_image_evidence(vlm, ctext, img_paths)

        item = {
            "Id": str(cid),
            "claim_text": ctext,
            "supporting_text_evidence": supporting_texts,
            "refuting_text_evidence": refuting_texts,
            "text_evidence_predictions": text_preds,
            "visual_evidence_label": v_label,
            "visual_evidence_raw_response": v_raw
        }
        outputs.append(item)

    # Save results
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Wrote {len(outputs)} records to {output_json}")


# ----------------------- CLI -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Path to MOCHEG root")
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--model_family", required=True, choices=["paligemma", "qwen2vl", "idefics3"])
    ap.add_argument("--hf_model_id", required=True, help="HF model id")
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_text_per_claim", type=int, default=20)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--force_hf", action="store_true", help="Force HF backend")
    args = ap.parse_args()

    use_vllm = False if args.force_hf else _USE_VLLM

    run_pipeline(
        data_root=args.data_root,
        split=args.split,
        hf_model_id=args.hf_model_id,
        model_family=args.model_family,
        output_json=args.output_json,
        batch_size=args.batch_size,
        max_text_per_claim=args.max_text_per_claim,
        use_vllm=use_vllm,
        max_new_tokens=args.max_new_tokens
    )


if __name__ == "__main__":
    main()

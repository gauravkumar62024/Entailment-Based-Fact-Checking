# test_fixed.py
"""A cleaned-up evaluation script for X-FACT multilingual fact-checking.

Key fixes
---------
1. Canonical label handling
2. Consistent label list
3. Utility functions cleaned
4. Full dataset evaluation

Usage
-----
python test_fixed.py \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --adapter saves/llama3-8b_llama_webreteived_evidences/lora/sft \
  --data test_xfact.jsonl \
  --out_dir outputs/xfact_predictions/llama3_lora
"""
from __future__ import annotations
import unicodedata
import argparse
import json
import os
import re
from difflib import get_close_matches
from pathlib import Path
from typing import Dict, List, Optional
import csv
import torch
from peft import PeftModel
from sklearn.metrics import classification_report
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

###############################################################################
# Argument parsing
###############################################################################

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Finetuned model with LoRA adapter")
    p.add_argument("--base_model", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--max_examples", type=int, default=None)
    p.add_argument("--max_length", type=int, default=10000)
    return p.parse_args()

args = parse_args()

def extract_dataset_info_simple(file_path: str):
    """
    Simpler version assuming consistent naming conventions.
    """
    filename = os.path.basename(file_path)
    filename = filename.replace('_justifications.json', '')
    
    # Initialize result
    result = {
        'filename': filename + '_justifications.json',
        'dataset_name': None,
        'model_name': None,
        'dataset_type': None,
        'technique': None
    }
    
    # Extract from path
    parts = os.path.normpath(file_path).split(os.sep)
    
    # Get dataset name (xfact or ru22fact)
    for part in parts:
        if part in ['xfact', 'ru22fact', 'lair_raw', 'rawfc']:
            result['dataset_name'] = part
            break
    
    # Get model name
    for part in parts:
        if part in ['gemma', 'llama', 'mistral', 'qwen']:
            result['model_name'] = part
            break
    
    # Split filename by underscores
    name_parts = filename.split('_')
    
    # Dataset type is usually the first part
    if name_parts:
        result['dataset_type'] = name_parts[0]
    
    # Technique is usually the last part(s)
    # Look for chunking techniques
    if len(name_parts) > 1:
        technique_candidate = '_'.join(name_parts[1:])
        
        # Check for known techniques
        if 'semantic_chunking' in technique_candidate:
            result['technique'] = 'semantic_chunking'
        elif 'sentence_chunking' in technique_candidate:
            result['technique'] = 'sentence_chunking'
        elif 'langchain_chunking' in technique_candidate:
            result['technique'] = 'langchain_chunking'
        elif 'custom_chunking' in technique_candidate:
            result['technique'] = 'custom_chunking'
    
    return result

def extract_random_seed_simple(adapter_path: str) -> str:
    """
    Simple version to extract random seed from adapter path.
    
    Rules:
    1. If last directory is 'sft' -> 'default'
    2. If last directory is 'sft_XX' -> 'XX'
    3. Otherwise look for any '_XX' pattern in last directory
    """
    # Get the last directory name
    last_dir = os.path.basename(os.path.normpath(adapter_path))
    
    # Check if it's 'sft' or 'sft_XX'
    if last_dir == 'sft':
        return 'default'
    elif last_dir.startswith('sft_'):
        # Extract number after 'sft_'
        seed_str = last_dir[4:]  # Remove 'sft_'
        return seed_str if seed_str.isdigit() else 'default'
    else:
        # Look for any number at the end after underscore
        if '_' in last_dir:
            parts = last_dir.split('_')
            if parts[-1].isdigit():
                return parts[-1]
    
    return 'default'

###############################################################################
# Canonical labels
###############################################################################

LABEL_SETS = {
    "xfact": {
        "order": [
            "true", "mostly true", "partly true/misleading", "false",
            "mostly false", "complicated/hard-to-categorise", "other",
        ],
        "canonical": {
            "true": "true",
            "mostly true": "mostly true",
            "mostly-true": "mostly true",
            "partly true/misleading": "partly true/misleading",
            "partly-true/misleading": "partly true/misleading",
            "false": "false",
            "mostly false": "mostly false",
            "mostly-false": "mostly false",
            "complicated/hard-to-categorise": "complicated/hard-to-categorise",
            "complicated/hard to categorise": "complicated/hard-to-categorise",
            "complicated/hard to categorize": "complicated/hard-to-categorise",
            "other": "other",
        },
    },
    "ru22fact": {
        "order": ["supported", "refuted", "nei"],
        "canonical": {
            "supported": "supported",
            "support": "supported",
            "refuted": "refuted",
            "refute": "refuted",
            "nei": "nei",
            "not enough info": "nei",
        },
    },
}
LABEL_SETS.update({
    "lair_raw": {
        "order": [
            "pants-fire", "false", "barely-true",
            "half-true", "mostly-true", "true"
        ],
        "canonical": {
            "pants-fire": "pants-fire",
            "pants on fire": "pants-fire",
            "pants-on-fire": "pants-fire",

            "false": "false",

            "barely-true": "barely-true",
            "barely true": "barely-true",

            "half-true": "half-true",
            "half true": "half-true",

            "mostly-true": "mostly-true",
            "mostly true": "mostly-true",

            "true": "true",
        },
    },

    "rawfc": {
        "order": ["true", "false", "half"],
        "canonical": {
            "true": "true",
            "false": "false",
            "half": "half",
            "half-true": "half",
            "partially true": "half",
        },
    },
})

# dataset_name, _ = extract_dataset_and_testset(Path(args.data).stem)
info = extract_dataset_info_simple(args.data)
dataset_name = info["dataset_name"]
label_config = LABEL_SETS[dataset_name]
LABEL_ORDER = label_config["order"]
LABEL_CANONICAL = label_config["canonical"]

LABEL_SET_LOWER = set(LABEL_CANONICAL.keys())
_DASH_CHARS = re.compile(r"[\u2010-\u2015\u2212\uFE58\uFE63\uFF0D\u002D]")

def _standardise_dashes(text: str) -> str:
    return _DASH_CHARS.sub("-", text)

def normalise_label(text: str) -> str:
    if not text:
        return "other"
    cleaned = _standardise_dashes(unicodedata.normalize("NFKC", text))
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    if cleaned in LABEL_CANONICAL:
        return LABEL_CANONICAL[cleaned]
    match = get_close_matches(cleaned, LABEL_SET_LOWER, n=1, cutoff=0.7)
    return LABEL_CANONICAL[match[0]] if match else "other"



###############################################################################
# Prompt construction
###############################################################################

# INSTRUCTIONS = {
#     "ru22fact": (
#         "Classify the given {language} claim into one of three categories "
#         "(SUPPORTED, REFUTED, NEI) based on the provided true and false justifications:\n\n"
#         "1. SUPPORTED: Fully supported by the context.\n"
#         "2. REFUTED: Contradicted or unsupported by the context.\n"
#         "3. NEI: Not enough information to verify.\n\n"
#         "Provide exactly one label."
#     ),

#     "xfact": (
#         "Classify the given {language} claim into one of the seven categories "
#         "(TRUE, MOSTLY-TRUE, PARTLY-TRUE/MISLEADING, FALSE, MOSTLY-FALSE, "
#         "COMPLICATED/HARD-TO-CATEGORIZE, OTHER) based on the provided true and false justifications.\n\n"
#         "1. TRUE: The claim is fully supported by the support (true) justification, and the refute (false) justification does not meaningfully contradict it.\n"
#         "2. MOSTLY-TRUE: The claim is largely supported by the support justification, but the refute justification highlights minor inaccuracies, limitations, or missing details.\n"
#         "3. PARTLY-TRUE/MISLEADING: The support justification confirms some aspects of the claim, but the refute justification reveals significant omissions, distortions, or potentially misleading elements.\n"
#         "4. FALSE: The refute (false) justification clearly contradicts or disproves the claim, and the support justification does not provide credible backing.\n"
#         "5. MOSTLY-FALSE: The refute justification shows the claim is largely incorrect, though the support justification may contain a small element of truth. \n"
#         "6. COMPLICATED/HARD-TO-CATEGORIZE: Too complex or nuanced to assign a straightforward label.\n"
#         "7. OTHER: Does not fit into any of the above categories.\n\n"
#         "Provide exactly one label."
#     )
# }
# SYSTEM_PROMPT = {
#     "xfact":
#         "You are a multilingual fact-checking expert. You are given a news claim along with raw evidence. "
#         "Your task is to classify the veracity strictly based on the provided true and false justifications.\n\n"
#         "Output format:\nClaim Veracity: [label]\n\n"
#         "Labels: TRUE, MOSTLY-TRUE, PARTLY-TRUE/MISLEADING, FALSE, MOSTLY-FALSE, "
#         "COMPLICATED/HARD-TO-CATEGORISE, OTHER.",
#     "ru22fact":
#         "You are a multilingual fact-checking expert. Classify the claim into: SUPPORTED, REFUTED, NEI.\n\n"
#         "Output format:\nClaim Veracity: [label]"
# }  

# def build_prompt(claim, support_justification, refute_justification,dataset,language):
#     instruction =INSTRUCTIONS[dataset].format(language=language)
#     return (
#         f"##Instruction: {instruction}\n\n"
#         f"##input: Claim: {claim}\n\n"
#         f"True Justification:\n{support_justification}\n\n"
#         f"False Justification:\n{refute_justification}\n"
#         "##output: "
#     )


# def build_chat_prompt(tokenizer, claim, support_justification, refute_justification,language, dataset):
#     user_prompt = build_prompt(claim, support_justification,refute_justification, dataset,language)
#     system_prompt = SYSTEM_PROMPT[dataset]
#     if not hasattr(tokenizer, "apply_chat_template"):
#         return system_prompt + "\n\n" + user_prompt

#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": user_prompt},
#     ]
#     try:
#         return tokenizer.apply_chat_template(
#             messages, tokenize=False, add_generation_prompt=True
#         )
#     except:
#         messages = [{"role": "user", "content": system_prompt + "\n\n" + user_prompt}]
#         return tokenizer.apply_chat_template(
#             messages, tokenize=False, add_generation_prompt=True
#         )
INSTRUCTIONS = {
    "ru22fact": (
        "Classify the given {language} claim into one of three categories "
        "(SUPPORTED, REFUTED, NEI) based on the provided true and false justifications:\n\n"
        "1. SUPPORTED: Fully supported by the context.\n"
        "2. REFUTED: Contradicted or unsupported by the context.\n"
        "3. NEI: Not enough information to verify.\n\n"
        "Provide exactly one label."
    ),

    "xfact": (
        "Classify the given {language} claim into one of the seven categories "
        "(TRUE, MOSTLY-TRUE, PARTLY-TRUE/MISLEADING, FALSE, MOSTLY-FALSE, "
        "COMPLICATED/HARD-TO-CATEGORIZE, OTHER) based on the provided true and false justifications.\n\n"
        "1. TRUE: The claim is fully supported by the support (true) justification, and the refute (false) justification does not meaningfully contradict it.\n"
        "2. MOSTLY-TRUE: The claim is largely supported by the support justification, but the refute justification highlights minor inaccuracies, limitations, or missing details.\n"
        "3. PARTLY-TRUE/MISLEADING: The support justification confirms some aspects of the claim, but the refute justification reveals significant omissions, distortions, or potentially misleading elements.\n"
        "4. FALSE: The refute (false) justification clearly contradicts or disproves the claim, and the support justification does not provide credible backing.\n"
        "5. MOSTLY-FALSE: The refute justification shows the claim is largely incorrect, though the support justification may contain a small element of truth. \n"
        "6. COMPLICATED/HARD-TO-CATEGORIZE: Too complex or nuanced to assign a straightforward label.\n"
        "7. OTHER: Does not fit into any of the above categories.\n\n"
        "Provide exactly one label."
    )
}
INSTRUCTIONS.update({
    "lair_raw": (
        "Classify the given {language} claim into one of six categories:\n\n"
        "TRUE, MOSTLY-TRUE, HALF-TRUE, BARELY-TRUE, FALSE, PANTS-FIRE.\n\n"
        "Use the true and false justifications provided.\n"
        "Provide exactly one label."
    ),
    "rawfc": (
        "Classify the given {language} claim into one of three categories:\n\n"
        "TRUE, FALSE, HALF.\n\n"
        "Use the provided justifications.\n"
        "Provide exactly one label."
    ),
})


def build_prompt(claim, support_justification, refute_justification):
    return (
        f"Claim: {claim}\n\n"
        f"True Justification:\n{support_justification}\n\n"
        f"False Justification:\n{refute_justification}\n"
    )


def build_chat_prompt(tokenizer, claim, support_justification, refute_justification,language, dataset):
    user_prompt = build_prompt(claim, support_justification,refute_justification)
    system_prompt = INSTRUCTIONS[dataset].format(language=language)

    if not hasattr(tokenizer, "apply_chat_template"):
        return system_prompt + "\n\n" + user_prompt

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except:
        messages = [{"role": "user", "content": system_prompt + "\n\n" + user_prompt}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

###############################################################################
# Response cleaning
###############################################################################

STRUCTURED_RE = re.compile(
    r"(claim veracity|verdict|label|prediction)\s*:\s*([^\n]+)", re.I
)

LABEL_PATTERN_RE = re.compile(
    r"\b("
    + "|".join(map(re.escape, [_standardise_dashes(l) for l in LABEL_SET_LOWER]))
    + r")\b",
    re.I,
)

def extract_label(response: str) -> str:
    if not response:
        return "other"
    resp_std = _standardise_dashes(response)
    m = STRUCTURED_RE.search(resp_std)
    if m:
        return normalise_label(m.group(2))
    m2 = LABEL_PATTERN_RE.search(resp_std)
    if m2:
        return normalise_label(m2.group(0))
    return normalise_label(resp_std)

###############################################################################
# Data loading
###############################################################################

import os
import json
import csv

def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, dict):
            data = [data]
    return data



###############################################################################
# Evaluation logic (UPDATED)
###############################################################################

def evaluate(
    base_model_path: str,
    adapter_path: str,
    test_data_path: str,
    output_dir: str,
    max_examples: Optional[int] = None,
    max_length: int = 10000,
):

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model …")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    print("Loading LoRA adapter …")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    print("Reading examples …")
    data = load_data(test_data_path)
    print(f"Loaded {len(data)} examples")

    if max_examples is not None:
        data = data[:max_examples]

    print(f"→ Evaluating on {len(data)} examples\n")
    info = extract_dataset_info_simple(test_data_path)
    model_shortname = info["model_name"]
    dataset = info['dataset_name']
    testset = info["dataset_type"]
    technique = info["technique"]
    random_seed = extract_random_seed_simple(adapter_path)

    print("[Mode] Normal evaluation ")

    predictions, gold, results_dump = [], [], []
    for ex in tqdm(data):

        try:
            evidences = []
            claim = ex["claim"]
            gold_label = normalise_label(ex["label"])
            language = ex["language"]
            support_justification = ex["support_justification"]
            refute_justification = ex["refute_justification"]
        
            prompt = build_chat_prompt(tokenizer, claim, support_justification,refute_justification, language, dataset)

            enc = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(model.device)

            out = model.generate(
                **enc,
                max_new_tokens=10,
                temperature=0.0,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
            )

            decoded = tokenizer.decode(
                out[0][enc.input_ids.shape[1]:],
                skip_special_tokens=True
            )

            pred_label = extract_label(decoded)

            predictions.append(pred_label)
            gold.append(gold_label)
            results_dump.append({
                 "input": prompt,
                 "true_label": gold_label,
                 "predicted_label": pred_label,
                 "model_response": decoded,
                 "language": language,
            })

        except Exception as exc:
            print(f"[!] Error → {exc}")
            predictions.append("other")
            gold.append("other")

        # Normal directory (no language folder)
    out_dir_full = Path(output_dir) / f"{dataset}/{testset}/{model_shortname}/{technique}"
    out_dir_full.mkdir(parents=True, exist_ok=True)

        # Save predictions
    with open(out_dir_full / f"predictions_{random_seed}.json", "w", encoding="utf-8") as f:
        json.dump(results_dump, f, indent=2, ensure_ascii=False)

        # Save metrics
    from collections import Counter
    gold_counts = Counter(gold)
    filtered_labels = [l for l in LABEL_ORDER if gold_counts[l] > 0]

    report = classification_report(
        gold, predictions,
        labels=filtered_labels,
        zero_division=0,
        digits=4
    )

    with open(out_dir_full / f"metrics_{random_seed}.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nSaved → {out_dir_full}")
    return  # END MODE 1

   
###############################################################################
# CLI entry-point
###############################################################################

if __name__ == "__main__":
    evaluate(
        base_model_path=args.base_model,
        adapter_path=args.adapter,
        test_data_path=args.data,
        output_dir=args.out_dir,
        max_examples=args.max_examples,
        max_length=args.max_length,
    )


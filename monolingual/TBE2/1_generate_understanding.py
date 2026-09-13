"""
TBE-2 / IBE-2 / IBE-3, step 1: generate the GLM's overall claim-evidence "understanding".

This is the shared implementation for both monolingual datasets (LIAR-RAW and
RAW-FC); the two differ only in the file names of their step-0 inputs, which are
passed on the command line.

Input : transformed_data_{split}.json  -- records with
        {"event_id", "claim", "label", "tokenized_sentences": [...]}
Output: understanding_{split}_{model}.json -- records with an extra
        "understanding" field, consumed by
          * TBE-2 step 2 (ELM training: 2_roberta_train.py / 2_xlnet_train.py)
          * TBE-2 step 3 (llamafactory_format/transform_*.py -> LoRA / LoRA+)
          * IBE-2 and IBE-3 (prompted veracity prediction)

Example
-------
    # RAW-FC
    python 1_generate_understanding.py \
        --data_dir ../TBE3/rawfc/data_rawfc_step1 \
        --train transformed_data_train_rawfc.json \
        --val   transformed_data_val_rawfc.json \
        --test  transformed_data_test_rawfc.json \
        --models llama qwen2.5 gemma mistral falcon \
        --out_dir outputs/rawfc

    # LIAR-RAW
    python 1_generate_understanding.py \
        --data_dir ../TBE3/lair_raw/first_step_data_lair_raw \
        --train lair_raw_transformed_data_train.json \
        --val   transformed_data_val.json \
        --test  lair_raw_transformed_data_test.json \
        --models llama --out_dir outputs/lair_raw

    # smoke test: 5 claims, one model
    python 1_generate_understanding.py --data_dir ... --test ... \
        --models llama --limit 5 --out_dir /tmp/smoke
"""

import argparse
import json
import os

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Model ids as reported in Table 14 of the paper.
MODELS_CONFIG = {
    "llama":   "meta-llama/Llama-3.1-8B-Instruct",
    "qwen":    "Qwen/Qwen2-7B-Instruct",
    "qwen2.5": "Qwen/Qwen2.5-7B-Instruct-1M",
    "gemma":   "google/gemma-7b-it",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "falcon":  "tiiuae/Falcon3-7B-Instruct",
}


def setup_model(model_id, gpu_memory_utilization=0.7):
    model = LLM(
        model=model_id,
        tensor_parallel_size=1,
        max_model_len=5000,
        enable_chunked_prefill=True,
        max_num_batched_tokens=131072,
        enforce_eager=True,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    sampling_params = SamplingParams(
        temperature=0.01,
        top_p=0.8,
        repetition_penalty=1.05,
        max_tokens=512,
    )
    return model, sampling_params


def generate_understanding_prompt(claim, evidences):
    return (
        f"Here is a claim: \"{claim}\"\n"
        f"What is your understanding of this claim based on its context and meaning?\n\n"
        f"Here is the evidence: \"{evidences}\"\n"
        f"What is your understanding of this evidence in relation to the claim?\n\n"
        "Now, based on your understanding of both, summarize how the evidence relates to the claim. "
        "Provide a brief explanation (not more then 100 words) that captures the key reasoning between them."
    )


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_understanding(model, tokenizer, sampling_params, prompt):
    try:
        messages = [
            {
                "role": "system",
                "content": "You are a fact-checking assistant generating understanding of claims based on evidence.",
            },
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        print("Warning: Chat template not supported, using raw prompt instead.")
        text = prompt

    outputs = model.generate([text], sampling_params)
    return outputs[0].outputs[0].text.strip() if outputs else ""


def process_split(split, file_path, model_name, model, tokenizer, sampling_params, out_dir, limit=0):
    data = load_json(file_path)
    if limit:
        data = data[:limit]

    results = []
    for entry in tqdm(data, desc=f"{model_name} - {split}"):
        claim_id = entry.get("event_id", str(len(results)))
        claim = entry["claim"]
        label = entry["label"]
        tokenized_sentences = entry.get("tokenized_sentences", [])

        if not tokenized_sentences:
            print(f"Skipping claim {claim_id} due to missing evidence.")
            continue

        evidence_text = " ".join(tokenized_sentences)
        prompt = generate_understanding_prompt(claim, evidence_text)
        understanding = generate_understanding(model, tokenizer, sampling_params, prompt)

        results.append(
            {
                "claim_id": claim_id,
                "claim": claim,
                "label": label,
                "evidence": evidence_text,
                "understanding": understanding,
            }
        )

    os.makedirs(out_dir, exist_ok=True)
    output_file = os.path.join(out_dir, f"understanding_{split}_{model_name}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Saved {len(results)} records to {output_file}")
    return output_file


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True, help="Directory holding the step-0 transformed_data_*.json files")
    p.add_argument("--train", default=None, help="Train split filename (relative to --data_dir)")
    p.add_argument("--val", default=None, help="Validation split filename")
    p.add_argument("--test", default=None, help="Test split filename")
    p.add_argument("--models", nargs="+", default=["llama"], choices=list(MODELS_CONFIG),
                   help="Which GLMs to run (default: llama)")
    p.add_argument("--out_dir", default="outputs", help="Where to write understanding_{split}_{model}.json")
    p.add_argument("--limit", type=int, default=0,
                   help="Process only the first N claims per split (0 = all). Use a small value for smoke tests.")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    args = p.parse_args()

    splits = {k: v for k, v in (("train", args.train), ("val", args.val), ("test", args.test)) if v}
    if not splits:
        p.error("Provide at least one of --train / --val / --test")

    for model_key in args.models:
        model_id = MODELS_CONFIG[model_key]
        print(f"\n=== Running understanding generation with {model_key} ({model_id}) ===")
        model, sampling_params = setup_model(model_id, args.gpu_memory_utilization)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        try:
            for split, fname in splits.items():
                path = os.path.join(args.data_dir, fname)
                if not os.path.exists(path):
                    print(f"Missing input, skipping: {path}")
                    continue
                process_split(split, path, model_key, model, tokenizer,
                              sampling_params, args.out_dir, args.limit)
        finally:
            del model, sampling_params
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass


if __name__ == "__main__":
    main()

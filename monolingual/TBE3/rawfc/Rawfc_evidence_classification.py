import os
import re
import json
import argparse
from tqdm import tqdm
from pathlib import Path
from vllm import LLM, SamplingParams
import torch

# ----------------------- Argument Parsing ----------------------- #
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, required=True, help="Base output directory (e.g., ./outputs)")
parser.add_argument('--data_dir', type=str, default='.', help="Directory holding data_rawfc_step1/")
parser.add_argument('--limit', type=int, default=0,
                    help="Process only the first N claims (0 = full dataset). Use a small value for smoke tests.")
parser.add_argument('--models', nargs='+', default=None,
                    help="Short names of the GLMs to run (llama qwen gemma mistral falcon). Default: all five.")
parser.add_argument('--splits', nargs='+', default=None, choices=['train', 'val', 'test'],
                    help="Which splits to process. Default: all three.")
args = parser.parse_args()

# Define models and their short names
MODELS = {
    "meta-llama/Llama-3.1-8B-Instruct": "llama",
    "Qwen/Qwen2-7B-Instruct": "qwen",
    "google/gemma-7b-it": "gemma",
    "mistralai/Mistral-7B-Instruct-v0.3": "mistral",
    "tiiuae/Falcon3-7B-Instruct": "falcon"
}

DATASETS = {
    "train": "data_rawfc_step1/transformed_data_train_rawfc.json",
    "test": "data_rawfc_step1/transformed_data_test_rawfc.json",
    "val": "data_rawfc_step1/transformed_data_val_rawfc.json"
}

# --------------------- Model Setup --------------------- #
def setup_model(model_path):
    model = LLM(
        model=model_path,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=6000,
        dtype="float16",
        gpu_memory_utilization=0.6
    )
    sampling_params = SamplingParams(
        temperature=0.01,
        max_tokens=1000,
        top_p=0.95,
        frequency_penalty=0.1
    )
    return model, sampling_params

# ------------------ Prompt Generation ------------------ #
def generate_prompt(claim, label, sentence):
    return f"""<|im_start|>system
You are an evidence classifier that determines whether a sentence supports or refutes a given claim.
You must also assign a confidence score between 0 and 1 based on how strongly the sentence supports or refutes the claim.
<|im_end|>
<|im_start|>user
Task: Analyze the sentence below and classify its relationship to the given claim.

**Classification Rules:**
1. **SUPPORT** - The sentence provides evidence that supports the claim.
2. **REFUTE** - The sentence provides evidence that contradicts or refutes the claim.

**Scoring Rules (0 to 1):**
- **0.0 - 0.3** - Weak support/refutation.
- **0.4 - 0.6** - Moderate support/refutation.
- **0.7 - 1.0** - Strong support/refutation.

**Claim:** {claim}
**Label for Claim:** {label}
**Sentence to classify:** {sentence}

Respond in the format: `SUPPORT: 0.75` or `REFUTE: 0.60`
<|im_end|>
<|im_start|>assistant
This sentence is: """

# ---------------- Classification & Processing ---------------- #
def process_dataset(dataset, model, sampling_params, dataset_name, model_name):
    results = []

    for item in tqdm(dataset, desc=f"Processing {dataset_name} Claims"):
        claim = item['claim']
        label = item['label']
        event_id = item.get('event_id', 'unknown')
        sentences = item.get('tokenized_sentences', [])
        if not sentences:
            print(f"Skipping claim {event_id}: No tokenized sentences")
            continue

        prompts = [generate_prompt(claim, label, s) for s in sentences]
        outputs = model.generate(prompts, sampling_params)

        supporting = []
        refuting = []

        for sent, output in zip(sentences, outputs):
            raw = output.outputs[0].text.strip()
            match = re.search(r'(SUPPORT|REFUTE):\s*([0-9.]+)', raw, re.IGNORECASE)
            if match:
                cls = match.group(1).upper()
                if cls == "SUPPORT":
                    supporting.append(sent)
                elif cls == "REFUTE":
                    refuting.append(sent)
            else:
                print(f"Invalid response for sentence '{sent}' in claim {event_id}: {raw}")

        results.append({
            "claim": claim,
            "label": label,
            "supporting_evidence": supporting,
            "refuting_evidence": refuting
        })

    return results

# -------------------- Main Execution -------------------- #
def main():
    output_base = Path(args.output) / "outputs"
    output_base.mkdir(parents=True, exist_ok=True)

    models = MODELS
    if args.models:
        models = {p: n for p, n in MODELS.items() if n in args.models}
        if not models:
            raise SystemExit(f"--models {args.models} matched none of {sorted(MODELS.values())}")
    datasets = {k: v for k, v in DATASETS.items() if not args.splits or k in args.splits}

    for model_path, model_short_name in models.items():
        # Create model-specific output directory
        model_output_dir = output_base / model_short_name
        model_output_dir.mkdir(exist_ok=True)

        # Setup Model
        print(f"Setting up model: {model_path} ({model_short_name})")
        model, sampling_params = setup_model(model_path)

        for dataset_name, input_path in datasets.items():
            # Load input
            print(f"Loading dataset: {os.path.join(args.data_dir, input_path)}")
            with open(os.path.join(args.data_dir, input_path), 'r', encoding='utf-8') as f:
                data = json.load(f)
            if args.limit:
                data = data[:args.limit]
            # Process Dataset
            print(f"Processing {dataset_name} dataset with {model_short_name}...")
            final_data = process_dataset(data, model, sampling_params, dataset_name, model_short_name)

            # Save Output
            output_file = model_output_dir / f"entailment_evidence_rawfc_{dataset_name}_{model_short_name}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(final_data, f, indent=4, ensure_ascii=False)

            print(f"Output saved to: {output_file}")

        # Cleanup
        del model, sampling_params
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

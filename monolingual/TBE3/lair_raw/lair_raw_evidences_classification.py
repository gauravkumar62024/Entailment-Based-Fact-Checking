from vllm import LLM, SamplingParams
import json
import os
import re
from tqdm import tqdm
from typing import List, Dict, Any
from pathlib import Path
import argparse
import torch

# Define models and datasets
# The five GLMs of Table 14. Select a subset at run time with --models.
MODELS = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama":   "meta-llama/Llama-3.1-8B-Instruct",
    "gemma":   "google/gemma-7b-it",
    "falcon":  "tiiuae/Falcon3-7B-Instruct",
    "qwen":    "Qwen/Qwen2-7B-Instruct",
}

DATASETS = {
    "train": "lair_raw_transformed_data_train.json",
    "test": "lair_raw_transformed_data_test.json",
    "val": "transformed_data_val.json"  # Updated for consistency
}

def extract_label_and_score(raw_response: str) -> tuple:
    """Extract classification label and score from raw_response."""
    match = re.search(r'(SUPPORT|REFUTE):\s*([0-9.]+)', raw_response, re.IGNORECASE)
    if match:
        return match.group(1).upper(), float(match.group(2))
    return "REFUTE", 0.0  # Default to REFUTE for consistency

def create_output_dir(model_name: str) -> Path:
    """Create output directory for a given model."""
    model_dir = model_name.replace("/", "_").replace("-", "_")
    output_dir = Path(f"outputs/{model_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def read_json(file_path: str) -> List[Dict[str, Any]]:
    """Load JSON dataset."""
    with open(file_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    return dataset

def setup_model(model_path: str):
    """Initialize model with VLLM."""
    try:
        model = LLM(
            model=model_path,
            tensor_parallel_size=1,
            trust_remote_code=True,
            max_model_len=6800,
            dtype="float16",
            gpu_memory_utilization=0.6
        )
        sampling_params = SamplingParams(
            temperature=0.001,
            max_tokens=1000,
            top_p=0.95,
            frequency_penalty=0.1
        )
        return model, sampling_params
    except Exception as e:
        print(f"Error setting up model {model_path}: {str(e)}")
        raise

def generate_prompt_with_score(claim: str, label: str, sent: str) -> str:
    """Generate a prompt for classification."""
    return f"""<|im_start|>system
You are an evidence classifier that determines whether a sentence supports or refutes a given claim.
You must also assign a confidence score between 0 and 1 based on how strongly the sentence supports or refutes the claim.
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
**Sentence to classify:** {sent}

Carefully analyze the sentence and respond with:
1. A classification (**SUPPORT or REFUTE**).
2. A confidence score (a float value between 0 and 1).

Respond in the format: `CLASSIFICATION: SCORE`
Example Response: `SUPPORT: 0.75` or `REFUTE: 0.60`
<|im_end|>assistant
This sentence is: """

def classify_evidence_batch(
    model: LLM,
    sampling_params: SamplingParams,
    claim: str,
    label: str,
    sentences: List[str]
) -> Dict[str, List[str]]:
    """Classify claim-sentence pairs using the model and return sentence lists."""
    if not sentences:
        return {"supporting_evidence": [], "refuting_evidence": []}
    
    prompts = [generate_prompt_with_score(claim, label, sent) for sent in sentences]
    outputs = model.generate(prompts, sampling_params)
    
    supporting_evidence, refuting_evidence = [], []
    
    for i, output in enumerate(outputs):
        raw_response = output.outputs[0].text.strip()
        try:
            classification, score = extract_label_and_score(raw_response)
        except ValueError:
            classification = "REFUTE"
            score = 0.0
        
        score = max(0.0, min(score, 1.0))
        
        if classification == "SUPPORT":
            supporting_evidence.append(sentences[i])
        else:
            refuting_evidence.append(sentences[i])
    
    return {"supporting_evidence": supporting_evidence, "refuting_evidence": refuting_evidence}

def process_dataset(
    model_name: str,
    dataset_path: str,
    dataset_type: str,
    model,
    sampling_params,
    limit: int = 0
) -> List[Dict[str, Any]]:
    """Process dataset and classify evidence."""
    print(f"\nProcessing {dataset_type} dataset with model: {model_name}")
    output_dir = create_output_dir(model_name)
    
    print("Loading dataset...")
    data = read_json(dataset_path)
    if limit:
        data = data[:limit]
    print(f"Loaded {len(data)} examples.")
    
    results = []
    
    for item in tqdm(data, desc=f"Processing {dataset_type} claims"):
        claim_id = item.get('event_id', str(len(results)))
        claim = item['claim']
        label = item['label']
        tokenized_sentences = item.get('tokenized_sentences', [])
        
        evidence = classify_evidence_batch(
            model=model,
            sampling_params=sampling_params,
            claim=claim,
            label=label,
            sentences=tokenized_sentences
        )
        
        transformed_item = {
            "claim": claim,
            "label": label,
            "supporting_evidence": evidence["supporting_evidence"],
            "refuting_evidence": evidence["refuting_evidence"]
        }
        
        results.append(transformed_item)
    
    # Save results to a single JSON file
    model_dir = model_name.replace("/", "_").replace("-", "_")
    output_file = output_dir / f"{dataset_type}_lair_raw_{model_dir.split('_')[-1]}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    
    print(f"Completed processing {dataset_type} dataset with {model_name}. Saved to {output_file}")
    return results

def main():
    parser = argparse.ArgumentParser(description="Classify evidence for LIAR-RAW dataset.")
    parser.add_argument('--data_dir', type=str, default='.', help="Directory containing dataset files")
    parser.add_argument('--limit', type=int, default=0,
                        help="Process only the first N claims (0 = full dataset). Use a small value for smoke tests.")
    parser.add_argument('--models', nargs='+', default=None, choices=list(MODELS),
                        help="Short names of the GLMs to run. Default: all five.")
    parser.add_argument('--splits', nargs='+', default=None, choices=['train', 'val', 'test'],
                        help="Which splits to process. Default: all three.")
    args = parser.parse_args()
    
    selected = [MODELS[k] for k in (args.models or MODELS)]
    datasets = {k: v for k, v in DATASETS.items() if not args.splits or k in args.splits}

    for model_name in selected:
        print(f"\nStarting processing with model: {model_name}")
        
        # Setup model once per model
        try:
            print("Setting up model...")
            model, sampling_params = setup_model(model_name)
            print("Model setup complete.")
        except Exception as e:
            print(f"Failed to setup model {model_name}: {str(e)}. Skipping this model.")
            continue
        
        try:
            for dataset_type, dataset_file in datasets.items():
                dataset_path = os.path.join(args.data_dir, dataset_file)
                if not os.path.exists(dataset_path):
                    print(f"Dataset file not found: {dataset_path}. Skipping {dataset_type} for {model_name}")
                    continue
                
                try:
                    process_dataset(
                        model_name=model_name,
                        dataset_path=dataset_path,
                        dataset_type=dataset_type,
                        model=model,
                        sampling_params=sampling_params,
                        limit=args.limit
                    )
                except Exception as e:
                    print(f"Error processing {dataset_type} with {model_name}: {str(e)}")
                    continue
        finally:
            # Ensure cleanup happens even if an error occurs
            print(f"Clearing GPU cache for model: {model_name}")
            try:
                del model
                del sampling_params
            except NameError:
                print("Model or sampling params not found in scope; already cleared or not created.")
            torch.cuda.empty_cache()
        
        print(f"Completed all datasets for model: {model_name}")

if __name__ == "__main__":
    main()

import json
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from pathlib import Path
import argparse
import torch
# ----------------------- Argument Parsing ----------------------- #
parser = argparse.ArgumentParser()
parser.add_argument('--evidence_dir', type=str, required=True, help="Directory containing evidence files (e.g., ./results/outputs)")
parser.add_argument('--output', type=str, required=True, help="Base output directory for justifications (e.g., ./results)")
args = parser.parse_args()

# Define models and their short names
MODELS = {
    #"meta-llama/Llama-3.1-8B-Instruct": "llama",
    #"Qwen/Qwen2.5-7B-Instruct-1M": "qwen",
    "google/gemma-7b-it": "gemma",
    "mistralai/Mistral-7B-Instruct-v0.3": "mistral",
    "tiiuae/Falcon3-7B-Instruct": "falcon"
}

DATASETS = ["train", "test", "val"]

# Load JSON file
def load_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# Setup model and tokenizer using a given model_id
def setup_model(model_id):
    # Initialize tokenizer for the specific model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    llm = LLM(
        model=model_id,
        tensor_parallel_size=1,
        max_model_len=6800,
        enable_chunked_prefill=True,
        max_num_batched_tokens=131072,
        enforce_eager=True,
        gpu_memory_utilization=0.7
    )
    sampling_params = SamplingParams(
        temperature=0.001,
        top_p=0.8,
        repetition_penalty=1.05,
        max_tokens=512
    )
    return llm, sampling_params, tokenizer

# Generate justification using chat template or raw prompt for Gemma
def generate_justification(model, sampling_params, tokenizer, prompt, model_id):
    if model_id == "google/gemma-7b-it":
        # For Gemma, use the raw prompt directly without chat template
        text = prompt
    else:
        # For other models, use chat template
        system_message = {
            "role": "system",
            "content": "You have been specially designed to perform abductive reasoning for the fake news detection task and justification generation."
        }
        user_message = {
            "role": "user",
            "content": prompt
        }
        messages = [system_message, user_message]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    outputs = model.generate([text], sampling_params)
    if not outputs or not outputs[0].outputs or not outputs[0].outputs[0].text.strip():
        print(f"ERROR: Empty justification for claim: {prompt}")
        return "ERROR: No Output Generated"
    return outputs[0].outputs[0].text.strip()

# Process claims
def process_claims(data, model, sampling_params, tokenizer, model_id):
    results = []
    for entry in tqdm(data, desc="Processing Claims for Justification"):
        claim = entry["claim"]
        label = entry["label"]
        # True Justification
        if entry.get("supporting_evidence"):
            true_prompt = (
                f"Given a claim: {claim}\n a veracity label True, analyze the statement and provide a concise, well-reasoned explanation (150 words or fewer) based on the provided evidence. "
                f"Do not mention the label in your response. Focus only on key supporting factors and logical reasoning. Evidence: {entry['supporting_evidence']}"
            )
        else:
            true_prompt = (
                f"Given a claim: {claim}\n a veracity label True, analyze the statement and provide a concise, well-reasoned explanation (150 words or fewer) using logical reasoning and relevant knowledge. "
                f"Do not mention the label '{label}' in your response. Focus on key aspects that strengthen the claim."
            )
        support_justification = generate_justification(model, sampling_params, tokenizer, true_prompt, model_id)
        # Refuting Justification
        if entry.get("refuting_evidence"):
            false_prompt = (
                f"Given a claim: {claim}\n a veracity label False, examine the statement and present a logically sound explanation (150 words or fewer) using the provided evidence. "
                f"Do not mention the label '{label}' in your response. Focus on key factors that challenge the claim while maintaining clarity and coherence. Evidence: {entry['refuting_evidence']}"
            )
        else:
            false_prompt = (
                f"Given a claim: {claim}\n a veracity label False, examine the statement and provide a well-reasoned counter-explanation (150 words or fewer) using logical inference and general knowledge. "
                f"Do not mention the label '{label}' in your response. Ensure the reasoning directly addresses weaknesses in the claim."
            )
        refute_justification = generate_justification(model, sampling_params, tokenizer, false_prompt, model_id)
        if "ERROR" in support_justification:
            print(f"Empty True Justification for claim: {claim}")
        if "ERROR" in refute_justification:
            print(f"Empty False Justification for claim: {claim}")
        results.append({
            "claim": claim,
            "label": label,
            "support_justification": support_justification,
            "refute_justification": refute_justification
        })
    return results

# Main execution
def main():
    evidence_base = Path(args.evidence_dir)
    output_base = Path(args.output) / "generated_justification_rawfc_data"
    output_base.mkdir(parents=True, exist_ok=True)

    for model_id, model_short_name in MODELS.items():
        print(f"Processing model: {model_short_name}")
        model_output_dir = output_base / model_short_name
        model_output_dir.mkdir(exist_ok=True)

        # Setup the model and tokenizer
        model, sampling_params, tokenizer = setup_model(model_id)

        for split in DATASETS:
            # Dynamically locate evidence file
            evidence_file = evidence_base / model_short_name / f"entailment_evidence_rawfc_{split}_{model_short_name}.json"
            if not evidence_file.exists():
                print(f"Evidence file not found: {evidence_file}")
                continue

            print(f"Processing split: {split} from file: {evidence_file}")
            data = load_data(evidence_file)
            justifications = process_claims(data, model, sampling_params, tokenizer, model_id)

            # Save justifications
            output_filename = model_output_dir / f"generated_justification_{split}_{model_short_name}_rawfc.json"
            with open(output_filename, "w", encoding='utf-8') as f:
                json.dump(justifications, f, indent=4, ensure_ascii=False)
            print(f"Saved results to: {output_filename}")

        # Cleanup
        del model, sampling_params, tokenizer
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

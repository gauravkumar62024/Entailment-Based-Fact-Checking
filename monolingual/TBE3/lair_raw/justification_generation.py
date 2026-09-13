import json
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch
from pathlib import Path

LIAR_RAW_LABELS = {
    "pants-fire": 0,
    "false": 0,
    "barely-true": 0,
    "half-true": 1,
    "mostly-true": 2,
    "true": 2
}

# Dictionary mapping models to their IDs and data file names
models_config = {
    #"gemma": {
    #    "model_id": "google/gemma-7b-it",
     #   "tokenizer_id": "google/gemma-7b-it",
    #    "data_files": {
     #       "train": "outputs/google_gemma_7b_it/train_lair_raw_it.json",
     #       "val": "outputs/google_gemma_7b_it/val_lair_raw_it.json",
    #        "test": "outputs/google_gemma_7b_it/test_lair_raw_it.json"
   #     }
   # },
   "mistral": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "tokenizer_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "data_files": {
            "train": "outputs/mistralai_Mistral_7B_Instruct_v0.3/train_lair_raw_v0.3.json",
            "val": "outputs/mistralai_Mistral_7B_Instruct_v0.3/val_lair_raw_v0.3.json",
            "test": "outputs/mistralai_Mistral_7B_Instruct_v0.3/test_lair_raw_v0.3.json"
        }
    },
    "llama": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "tokenizer_id": "meta-llama/Llama-3.1-8B-Instruct",
        "data_files": {
            "train": "outputs/meta_llama_Llama_3.1_8B_Instruct/train_lair_raw_Instruct.json",
            "val": "outputs/meta_llama_Llama_3.1_8B_Instruct/val_lair_raw_Instruct.json",
            "test": "outputs/meta_llama_Llama_3.1_8B_Instruct/test_lair_raw_Instruct.json"
        }
    },
    "falcon": {
        "model_id": "tiiuae/Falcon3-7B-Instruct",
        "tokenizer_id": "tiiuae/falcon-7b-instruct",
        "data_files": {
            "train": "outputs/tiiuae_Falcon3_7B_Instructtrain_lair_raw_Instruct.json",
            "val": "outputs/tiiuae_Falcon3_7B_Instruct/val_lair_raw_Instruct.json",
            "test": "outputs/tiiuae_Falcon3_7B_Instruct/test_lair_raw_Instruct.json"
        }
    },
    "qwen": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "tokenizer_id": "Qwen/Qwen2.5-7B-Instruct",
        "data_files": {
            "train": "outputs/Qwen_Qwen2_7B_Instruct/train_lair_raw_Instruct.json",
            "val": "outputs/Qwen_Qwen2_7B_Instruct/val_lair_raw_Instruct.json",
            "test": "outputs/Qwen_Qwen2_7B_Instruct/test_lair_raw_Instruct.json"
        }
    }
}

# Load JSON file
def load_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# Setup model using a given model_id
def setup_model(model_id: str):
    llm = LLM(
        model=model_id,
        tensor_parallel_size=1,
        max_model_len=6600,
        enable_chunked_prefill=True,
        max_num_batched_tokens=131072,
        enforce_eager=True,
        gpu_memory_utilization=0.8
    )
    
    sampling_params = SamplingParams(
        temperature=0.01,
        top_p=0.8,
        repetition_penalty=1.05,
        max_tokens=512
    )
    
    return llm, sampling_params

# Generate justification using chat template for non-Gemma models, direct prompt for Gemma
def generate_justification(model, sampling_params, prompt, tokenizer, model_key):
    system_message_content = "You have been specially designed to perform abductive reasoning for the fake news detection task."

    if model_key == "gemma":
        # For Gemma, use the direct prompt without chat template
        text = f"{system_message_content}\n\n{prompt}"
    else:
        # For other models, use the chat template
        system_message = {
            "role": "system",
            "content": system_message_content
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

# Process claims while keeping your original prompt structure
def process_claims(data, model, sampling_params, tokenizer, model_key):
    results = []
    
    for entry in tqdm(data, desc="Processing claims"):
        claim = entry["claim"]
        label = entry["label"]
        mapped_label = LIAR_RAW_LABELS.get(label, 1)
        # Support Justification
        if entry["supporting_evidence"]:
            support_prompt = f"Given a claim: {claim}\n a veracity label True, analyze the statement and provide a concise, well-reasoned explanation (150 words or fewer) based on the provided evidence. Do not mention the label '{label}' in your response. Focus only on key supporting factors and logical reasoning. Evidence: {entry['supporting_evidence']}"
        else:
            support_prompt = f"Given a claim: {claim}\n a veracity label True, analyze the statement and provide a concise, well-reasoned explanation (150 words or fewer) using logical reasoning and relevant knowledge. Do not mention the label '{label}' in your response. Focus on key aspects that strengthen the claim."

        support_justification = generate_justification(model, sampling_params, support_prompt, tokenizer, model_key)

        # Refute Justification
        if entry["refuting_evidence"]:
            refute_prompt = f"Given a claim: {claim}\n a veracity label False, examine the statement and present a logically sound explanation (150 words or fewer) using the provided evidence. Do not mention the label '{label}' in your response. Focus on key factors that challenge the claim while maintaining clarity and coherence. Evidence: {entry['refuting_evidence']}"
        else:
            refute_prompt = f"Given a claim: {claim}\n a veracity label False, examine the statement and provide a well-reasoned counter-explanation (150 words or fewer) using logical inference and general knowledge. Do not mention the label '{label}' in your response. Ensure the reasoning directly addresses weaknesses in the claim."

        refute_justification = generate_justification(model, sampling_params, refute_prompt, tokenizer, model_key)

        # Log empty justifications
        if "ERROR" in support_justification:
            print(f"Empty Support Justification for claim: {claim}")
        if "ERROR" in refute_justification:
            print(f"Empty Refute Justification for claim: {claim}")

        results.append({
            "claim": claim,
            "label": label,
            "support_justification": support_justification,
            "refute_justification": refute_justification
        })
    
    return results

def process_model(model_key, model_info):
    print(f"Loading tokenizer for model: {model_key}")
    tokenizer_id = model_info["tokenizer_id"]
    
    # Load the model-specific tokenizer
    print(f"Loading tokenizer: {tokenizer_id}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    
    print(f"Processing model: {model_key}")
    model_id = model_info["model_id"]
    data_files = model_info["data_files"]
    
    # Setup the model
    model, sampling_params = setup_model(model_id)
    
    try:
        for split, file_path in data_files.items():
            print(f"  Processing split: {split} from file: {file_path}")
            if not Path(file_path).exists():
                print(f"    File not found: {file_path}. Skipping {split} for {model_key}.")
                continue
            
            data = load_data(file_path)
            justifications = process_claims(data, model, sampling_params, tokenizer, model_key)
            
            # Save the justifications in the same folder as the input file
            output_dir = Path(file_path).parent
            output_filename = output_dir / f"generated_justifications_{split}_{model_key}.json"
            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump(justifications, f, indent=4, ensure_ascii=False)
            print(f"  Saved results to: {output_filename}")
    finally:
        # Ensure GPU cleanup even if an error occurs
        print(f"Clearing GPU cache for model: {model_key}")
        try:
            del model
            del sampling_params
        except NameError:
            print("Model or sampling params not found in scope; already cleared or not created.")
        torch.cuda.empty_cache()

def main():
    for model_key, model_info in models_config.items():
        process_model(model_key, model_info)

if __name__ == "__main__":
    main()

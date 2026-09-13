import json
import os
import gc
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from sklearn.metrics import classification_report

##############################################################################
# 1. LIAR Label Configuration (6 classes)
##############################################################################
LIAR_RAW_LABELS = {
    "pants-fire": 0,
    "false": 1,
    "barely-true": 2,
    "half-true": 3,
    "mostly-true": 4,
    "true": 5
}
INV_LABEL_MAP = {v: k for k, v in LIAR_RAW_LABELS.items()}

##############################################################################
# 2. Data Loading Function
##############################################################################
def load_data(json_path):
    """Load dataset from a JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

##############################################################################
# 3. Prompt Definitions
##############################################################################
SYSTEM_PROMPT = (
    "You are a fact-checking assistant. You are given a claim along with an 'Understanding' of the evidence sentences for the claim. "
    "Do not consider any other evidence. Your task is to determine the veracity of the claim based on the understanding.\n\n"
    "First, explain your reasoning step-by-step using only the understanding provided.\n"
    "Then, determine the claim's veracity using one of the following six categories:\n\n"
    "1) pants-fire  : The claim is ridiculously or outrageously untrue.\n"
    "2) false       : The claim is entirely incorrect and unsupported by evidence.\n"
    "3) barely-true : The claim has a small element of truth but is mostly misleading.\n"
    "4) half-true   : The claim mixes true and false information in significant amounts.\n"
    "5) mostly-true : The claim is generally correct but contains minor inaccuracies.\n"
    "6) true        : The claim is completely accurate and fully supported by evidence.\n\n"
    "Finally, output your answer in exactly the following format:\n"
    "Claim Veracity: [label]\n\n"
    "where [label] must be one of (pants-fire, false, barely-true, half-true, mostly-true, or true)."
)

def build_user_prompt(claim, understanding):
    """
    Build the user prompt for Chain-of-Thought reasoning, given a claim and understanding.
    """
    return (
        f"Claim: {claim}\n"
        f"Understanding: {understanding}\n\n"
        "First, explain your reasoning based only on the understanding provided above.\n"
        "Then, determine the overall veracity of the claim using one of the following labels:\n"
        "- pants-fire\n"
        "- false\n"
        "- barely-true\n"
        "- half-true\n"
        "- mostly-true\n"
        "- true\n\n"
        "Finally, answer in the following format:\n"
        "Claim Veracity: [label]"
    )

##############################################################################
# 4. vLLM Setup and Inference Functions
##############################################################################
def setup_vllm_model(model_id: str):
    """
    Initialize the vLLM model and sampling parameters.
    Adjust GPU-related parameters (like gpu_memory_utilization) as needed.
    """
    model = LLM(
        model=model_id,
        tensor_parallel_size=1,
        max_model_len=5000,
        enable_chunked_prefill=True,
        max_num_batched_tokens=131072,
        enforce_eager=True,
        gpu_memory_utilization=0.9
    )
    sampling_params = SamplingParams(
        temperature=0.001,
        top_p=0.8,
        repetition_penalty=1.05,
        max_tokens=1024
    )
    return model, sampling_params

def generate_response_vllm(model, tokenizer, sampling_params, prompt):
    """
    Generate a text response using vLLM.
    Attempt to use a chat template if the tokenizer supports it; otherwise, use the raw prompt.
    """
    try:
        system_message = {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
        user_message = {
            "role": "user",
            "content": prompt
        }
        messages = [system_message, user_message]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception as e:
        print("Warning: Chat template not supported, using raw prompt.")
        text = prompt

    outputs = model.generate([text], sampling_params)
    if outputs and outputs[0].outputs and outputs[0].outputs[0].text.strip():
        return outputs[0].outputs[0].text.strip()
    else:
        return ""

def extract_liar_label(generated_text):
    """
    Enhanced function to extract labels with special handling for compound labels.
    """
    generated_text = generated_text.lower().strip()
    
    COMPOUND_LABELS = [
        ('mostly-true', ['mostly true', 'mostly-true', 'mostly_true']),
        ('barely-true', ['barely true', 'barely-true', 'barely_true']),
        ('half-true', ['half true', 'half-true', 'half_true']),
        ('pants-fire', ['pants fire', 'pants-fire', 'pants on fire', 'pantsfire'])
    ]
    
    SINGLE_LABELS = [
        ('true', ['true']),
        ('false', ['false'])
    ]
    
    for label, variants in COMPOUND_LABELS:
        for variant in variants:
            if variant in generated_text:
                return label
    
    for label, variants in SINGLE_LABELS:
        for variant in variants:
            if variant in generated_text and not any(compound in generated_text for compound, _ in COMPOUND_LABELS):
                return label
    
    all_labels = COMPOUND_LABELS + SINGLE_LABELS
    for label, variants in all_labels:
        for variant in variants:
            words = variant.split()
            if len(words) > 1:
                if all(word in generated_text for word in words):
                    return label
    
    text_words = generated_text.split()
    for word in text_words:
        if 'mostly' in word or ('most' in word and 'true' in generated_text):
            return 'mostly-true'
        elif 'barely' in word or ('bare' in word and 'true' in generated_text):
            return 'barely-true'
        elif 'half' in word and 'true' in generated_text:
            return 'half-true'
        elif ('pants' in word and 'fire' in generated_text) or ('pant' in word and 'fire' in generated_text):
            return 'pants-fire'
        elif word == 'true' and not any(compound in generated_text for compound, _ in COMPOUND_LABELS):
            return 'true'
        elif word == 'false':
            return 'false'
    
    if 'mostly' in generated_text and 'true' in generated_text:
        return 'mostly-true'
    elif 'barely' in generated_text and 'true' in generated_text:
        return 'barely-true'
    elif 'half' in generated_text and 'true' in generated_text:
        return 'half-true'
    elif ('pants' in generated_text and 'fire' in generated_text) or 'pantsfire' in generated_text:
        return 'pants-fire'
    elif 'true' in generated_text:
        return 'true'
    elif 'false' in generated_text:
        return 'false'
    
    return 'none'

def run_inference_for_dataset(model, tokenizer, sampling_params, data):
    """
    For each sample:
      - Build a prompt (system prompt + dynamic user prompt)
      - Generate the model's response using vLLM
      - Extract the predicted LIAR label
    Returns a list of results.
    """
    results = []
    for sample in tqdm(data, desc="Processing Samples"):
        claim = sample.get("claim", "").strip()
        understanding = sample.get("understanding", "").strip()
        actual_label = sample.get("label", "").strip().lower()
        
        user_prompt = build_user_prompt(claim, understanding)
        full_prompt = SYSTEM_PROMPT + "\n\n" + user_prompt
        
        generated_response = generate_response_vllm(model, tokenizer, sampling_params, full_prompt)
        predicted_label = extract_liar_label(generated_response)
        
        results.append({
            "claim": claim,
            "actual_label": actual_label,
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": user_prompt,
            "full_prompt": full_prompt,
            "model_response": generated_response,
            "predicted_label": predicted_label
        })
    return results

def evaluate_liar_results(results):
    """Compute and return a classification report for the 6 LIAR classes."""
    y_true, y_pred = [], []
    for r in results:
        gold = r.get("actual_label", "half-true")
        pred = r.get("predicted_label", "half-true")
        if gold not in LIAR_RAW_LABELS:
            gold = "half-true"
        if pred not in LIAR_RAW_LABELS:
            pred = "half-true"
        y_true.append(LIAR_RAW_LABELS[gold])
        y_pred.append(LIAR_RAW_LABELS[pred])
    
    target_names = [INV_LABEL_MAP[i] for i in range(len(INV_LABEL_MAP))]
    report_text = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    return report_text

def save_json(data, filename):
    """Save the given data as a JSON file."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def save_text(text, filename):
    """Save text to a file."""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)

def clear_resources(model=None):
    """
    Clear GPU memory and Python garbage to free resources.
    """
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

##############################################################################
# 5. Main Function Using Experiments Dictionary
##############################################################################
def main():
    # Define experiments: mapping dataset keys to their respective vLLM model and dataset file
    experiments = {
        "mistral": {
            "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
            "dataset_file": "understanding_test_mistral.json"
        },
        "meta_llama": {
            "model_id": "meta-llama/Llama-3.1-8B-Instruct",
            "dataset_file": "understanding_test_llama.json"
        },
        "gemma": {
            "model_id": "google/gemma-7b-it",
            "dataset_file": "understanding_test_gemma.json"
        },
        "qwen": {
            "model_id": "Qwen/Qwen2.5-7B-Instruct-1M",
            "dataset_file": "understanding_test_qwen.json"
        },
        "falcon": {
            "model_id": "tiiuae/Falcon3-7B-Instruct",
            "dataset_file": "understanding_test_falcon.json"
        }
    }
    
    # Create output folders
    output_results_folder = "lair_generated_results2"
    output_reports_folder = "lair_classification_reports2"
    os.makedirs(output_results_folder, exist_ok=True)
    os.makedirs(output_reports_folder, exist_ok=True)
    
    # Process each experiment sequentially
    for key, config in experiments.items():
        print(f"\n--- Running inference for {key} ---")
        model_id = config["model_id"]
        dataset_file = config["dataset_file"]
        
        # Load dataset
        data = load_data(dataset_file)
        print(f"Loaded {len(data)} samples from {dataset_file}")
        
        # Set up the vLLM model and sampling parameters
        print(f"Loading vLLM model: {model_id}")
        model, sampling_params = setup_vllm_model(model_id)
        
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Run inference
        results = run_inference_for_dataset(model, tokenizer, sampling_params, data)
        
        # Save generated results
        results_filename = os.path.join(output_results_folder, f"{key}_generated_results.json")
        save_json(results, results_filename)
        print(f"Saved generated results to {results_filename}")
        
        # Evaluate results and save the classification report
        report_text = evaluate_liar_results(results)
        print(f"Classification Report for {key}:\n{report_text}")
        report_filename = os.path.join(output_reports_folder, f"{key}_classification_report.txt")
        save_text(report_text, report_filename)
        print(f"Saved classification report to {report_filename}")
        
        # Clear resources before loading the next model
        print(f"Clearing resources for {key}...")
        clear_resources(model)
        del tokenizer
        gc.collect()

if __name__ == "__main__":
    main()

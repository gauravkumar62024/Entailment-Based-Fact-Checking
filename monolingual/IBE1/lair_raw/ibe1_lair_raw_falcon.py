import json
import os
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from sklearn.metrics import classification_report
import torch
import re
# Mapping for the new 6 labels
LABEL_MAP = {"false": 0, "half-true": 1, "true": 2, "mostly-true": 3, "barely-true": 4, "pants-fire": 5}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

def load_data(json_path):
    """Load dataset from a JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Flatten tokenized_sentences into a single evidence string for each sample
    for sample in data:
        sample["evidence"] = " ".join(sample["tokenized_sentences"])
    return data

# System prompt (updated for 6 veracity labels)
SYSTEM_PROMPT = (
    "You are a fact-checking assistant. You are given a claim along with evidence sentences. "
    "Your task is to determine the veracity of the claim based on the provided evidence sentences.\n\n"
    "Output your answer in exactly the following format:\n"
    "Claim Veracity: [label]\n\n"
    "Answer with one word: true, false, half-true, mostly-true, barely-true, or pants-fire.\n\n"
    "Label Definitions:\n"
    " true: The claim is completely supported by the evidence and is verified as accurate.\n"
    " false: The claim is clearly refuted by the evidence and is demonstrably incorrect.\n"
    " half-true: The claim contains both accurate and inaccurate details, making it partially true.\n"
    " mostly-true: The claim is mostly supported by the evidence, but contains some inaccuracies.\n"
    " barely-true: The claim contains some truth but is mostly misleading or inaccurate.\n"
    " pants-fire: The claim is completely false and outrageously ridiculous."
)

def build_user_prompt(claim, evidence):
    """Build the dynamic user prompt given a claim and its evidence."""
    return (
        f"Claim: {claim}\n"
        f"Evidence: {evidence}\n\n"
        "Based on the above evidence sentnces, determine the veracity of the claim. "
        "Answer using one word (true, false, half-true, mostly-true, barely-true, pants-fire) using the specified format."
    )

def setup_vllm_model(model_id: str):
    """
    Initialize the LLM model and sampling parameters with reduced GPU usage.
    Adjust max_tokens and other parameters as needed.
    """
    model = LLM(
        model=model_id,
        tensor_parallel_size=1,
        max_model_len=6700,
        enable_chunked_prefill=True,
        max_num_batched_tokens=13107,
        enforce_eager=True,
        gpu_memory_utilization=0.7
    )
    sampling_params = SamplingParams(
        temperature=0.001,
        top_p=0.8,
        repetition_penalty=1.05,
        max_tokens=50
    )
    return model, sampling_params

def generate_response_vllm(model, tokenizer, sampling_params, prompt):
    """
    Generate text using vLLM.
    If the tokenizer supports a chat template, use it; otherwise, use the raw prompt.
    """
    try:
        system_message = {"role": "system", "content": SYSTEM_PROMPT}
        user_message = {"role": "user", "content": prompt}
        messages = [system_message, user_message]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception as e:
        print("Warning: Chat template not supported, using raw prompt.")
        text = prompt

    outputs = model.generate([text], sampling_params)
    if outputs and outputs[0].outputs and outputs[0].outputs[0].text.strip():
        return outputs[0].outputs[0].text.strip()
    else:
        return ""

def extract_veracity(response):
    """
    Enhanced function to extract labels with special handling for compound labels
    """
    # Clean the response
    response = response.lower().strip()
    response = re.sub(r'<\|.*?\|>', '', response)  # Remove special tokens
    
    # Define compound labels first (ordered by length to catch longer matches first)
    COMPOUND_LABELS = [
        ('mostly-true', ['mostly true', 'mostly-true', 'mostly_true']),
        ('barely-true', ['barely true', 'barely-true', 'barely_true']),
        ('half-true', ['half true', 'half-true', 'half_true']),
        ('pants-fire', ['pants fire', 'pants-fire', 'pants on fire', 'pantsfire'])
    ]
    
    # Define single-word labels
    SINGLE_LABELS = [
        ('true', ['true']),
        ('false', ['false'])
    ]
    
    # First check for compound labels (exact matches)
    for label, variants in COMPOUND_LABELS:
        for variant in variants:
            if variant in response:
                return label
    
    # Then check for single-word labels
    for label, variants in SINGLE_LABELS:
        for variant in variants:
            if variant in response and not any(compound in response for compound, _ in COMPOUND_LABELS):
                return label
    
    # If no exact match found, try to find partial matches in order of preference
    all_labels = COMPOUND_LABELS + SINGLE_LABELS
    for label, variants in all_labels:
        for variant in variants:
            words = variant.split()
            if len(words) > 1:  # For compound labels
                if all(word in response for word in words):
                    return label
    
    # Default case - look for any matching word and map to appropriate label
    response_words = response.split()
    for word in response_words:
        if 'mostly' in word or ('most' in word and 'true' in response):
            return 'mostly-true'
        elif 'barely' in word or ('bare' in word and 'true' in response):
            return 'barely-true'
        elif 'half' in word and 'true' in response:
            return 'half-true'
        elif ('pants' in word and 'fire' in response) or ('pant' in word and 'fire' in response):
            return 'pants-fire'
        elif word == 'true' and not any(compound in response for compound, _ in COMPOUND_LABELS):
            return 'true'
        elif word == 'false':
            return 'false'
    
    # If still no match found, look for closest match in raw response
    if 'mostly' in response and 'true' in response:
        return 'mostly-true'
    elif 'barely' in response and 'true' in response:
        return 'barely-true'
    elif 'half' in response and 'true' in response:
        return 'half-true'
    elif ('pants' in response and 'fire' in response) or 'pantsfire' in response:
        return 'pants-fire'
    elif 'true' in response:
        return 'true'
    elif 'false' in response:
        return 'false'
    
    # Default if no match found
    return 'unknown'
def run_inference_for_dataset(model, tokenizer, sampling_params, data):
    """
    For each sample in the dataset:
      - Build the full prompt (system prompt + user prompt)
      - Generate response using vLLM
      - Extract the predicted veracity label
    Returns a list of result dictionaries.
    """
    results = []
    for sample in tqdm(data, desc="Processing Samples"):
        claim = sample.get("claim", "").strip()
        evidence = sample.get("tokenized_sentences", "")
        actual_label = sample.get("label", "").strip().lower()
        
        user_prompt = build_user_prompt(claim, evidence)
        full_prompt = SYSTEM_PROMPT + "\n\n" + user_prompt
        
        generated_response = generate_response_vllm(model, tokenizer, sampling_params, full_prompt)
        predicted_label = extract_veracity(generated_response)
        
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

def evaluate_results(results):
    """Compute a classification report comparing gold versus predicted labels."""
    y_true, y_pred = [], []
    for res in results:
        gold = res.get("actual_label", "")
        pred = res.get("predicted_label", "")
        if gold not in LABEL_MAP:
            gold = "half-true"
        if pred not in LABEL_MAP:
            pred = "half-true"
        y_true.append(LABEL_MAP[gold])
        y_pred.append(LABEL_MAP[pred])
    report_text = classification_report(
        y_true, y_pred, target_names=["false", "half-true", "true", "mostly-true", "barely-true", "pants-fire"], digits=4
    )
    return report_text

def save_json(data, filename):
    """Save data as JSON."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def save_text(text, filename):
    """Save text to a file."""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)

def main():
    # Define experiments: mapping dataset keys to their respective vLLM model and dataset file.
    experiments = {
        "falcon": {
            "model_id": "tiiuae/Falcon3-7B-Instruct",
            "dataset_file": "lair_raw_transformed_data_test.json"
        }
    }
    output_results_folder = "lair_raw_generated_results_falcon"
    output_reports_folder = "lair_raw_classification_reports_falcon"
    os.makedirs(output_results_folder, exist_ok=True)
    os.makedirs(output_reports_folder, exist_ok=True)
    
    for key, config in experiments.items():
        print(f"\n--- Running inference for {key} ---")
        model_id = config["model_id"]
        dataset_file = config["dataset_file"]
        
        data = load_data(dataset_file)
        print(f"Loaded {len(data)} samples from {dataset_file}")
        
        print(f"Loading vLLM model: {model_id}")
        model, sampling_params = setup_vllm_model(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        results = run_inference_for_dataset(model, tokenizer, sampling_params, data)
        
        results_filename = os.path.join(output_results_folder, f"{key}_generated_results.json")
        save_json(results, results_filename)
        print(f"Saved generated results to {results_filename}")
        
        report_text = evaluate_results(results)
        print(f"Classification Report for {key}:\n{report_text}")
        report_filename = os.path.join(output_reports_folder, f"{key}_classification_report.txt")
        save_text(report_text, report_filename)
        print(f"Saved classification report to {report_filename}")
        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()


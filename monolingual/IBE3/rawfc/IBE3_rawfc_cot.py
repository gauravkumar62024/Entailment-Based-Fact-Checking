import json
import os
import gc
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from sklearn.metrics import classification_report

##############################################################################
# 1. Label Configuration (3 classes)
##############################################################################
LABEL_MAP = {"false": 0, "half": 1, "true": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

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
    "You are a fact-checking assistant. You are given a claim along with an 'Understanding' of the claim. "
    "Do not consider any other evidence. Your task is to determine the overall veracity of the claim based on the understanding.\n\n"
    "First, explain your reasoning in detail based only on the understanding.\n"
    "Then output the claim veracity label in **exactly** the following format:\n"
    "Claim Veracity: [label]\n\n"
    "Answer with one word: true, false, or half."
)

def build_user_prompt(claim, understanding):
    """Build a chain-of-thought user prompt given a claim and its understanding."""
    return (
        f"Claim: {claim}\n"
        f"Understanding: {understanding}\n\n"
        "Label Definitions:\n"
        " - true: The claim is completely supported by the evidence and accurate.\n"
        " - false: The claim is clearly refuted by the evidence and incorrect.\n"
        " - half: The claim contains both accurate and inaccurate details, making it partially true.\n\n"
        "First, explain your reasoning step-by-step using only the understanding above.\n"
        "Then, based on your reasoning, provide the final claim veracity in the format:\n"
        "Claim Veracity: [true/false/half]"
    )

##############################################################################
# 4. vLLM Setup and Inference Functions
##############################################################################
def setup_vllm_model(model_id: str):
    """
    Initialize the LLM model and sampling parameters with reduced GPU usage.
    """
    model = LLM(
        model=model_id,
        tensor_parallel_size=1,
        max_model_len=2048,
        enable_chunked_prefill=True,
        max_num_batched_tokens=13107,
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
    Generate text using vLLM.
    If the tokenizer supports a chat template, use it; otherwise, use the raw prompt.
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

def extract_veracity(generated_text):
    """
    Extract the veracity label from the generated output.
    Expected output is a line like "Claim Veracity: [label]".
    """
    generated_text = generated_text.lower().strip()
    if "true" in generated_text:
        return "true"
    elif "false" in generated_text:
        return "false"
    elif "half" in generated_text:
        return "half"
    return "half"  # Default fallback

def run_inference_for_dataset(model, tokenizer, sampling_params, data):
    """
    For each sample:
      - Build the full prompt (system prompt + user prompt)
      - Generate response using vLLM
      - Extract the predicted veracity label
    Returns a list of result dictionaries.
    """
    results = []
    for sample in tqdm(data, desc="Processing Samples"):
        claim = sample.get("claim", "").strip()
        understanding = sample.get("understanding", "").strip()
        actual_label = sample.get("label", "").strip().lower()
        
        user_prompt = build_user_prompt(claim, understanding)
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
        gold = res.get("actual_label", "half")
        pred = res.get("predicted_label", "half")
        if gold not in LABEL_MAP:
            gold = "half"
        if pred not in LABEL_MAP:
            pred = "half"
        y_true.append(LABEL_MAP[gold])
        y_pred.append(LABEL_MAP[pred])
    report_text = classification_report(
        y_true, y_pred, target_names=["false", "half", "true"], digits=4
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
        "falcon": {
            "model_id": "tiiuae/Falcon3-7B-Instruct",
            "dataset_file": "understanding_test_falcon.json"
        },
        "qwen": {
            "model_id": "Qwen/Qwen2.5-7B-Instruct-1M",
            "dataset_file": "understanding_test_qwen.json"
        },
        "gemma": {
            "model_id": "google/gemma-7b-it",
            "dataset_file": "understanding_test_gemma.json"
        },
        "meta_llama": {
            "model_id": "meta-llama/Llama-3.1-8B-Instruct",
            "dataset_file": "understanding_test_llama.json"
        }
    }
    
    # Create output folders
    output_results_folder = "ibe2_generated_results"
    output_reports_folder = "ibe2_classification_reports"
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
        report_text = evaluate_results(results)
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

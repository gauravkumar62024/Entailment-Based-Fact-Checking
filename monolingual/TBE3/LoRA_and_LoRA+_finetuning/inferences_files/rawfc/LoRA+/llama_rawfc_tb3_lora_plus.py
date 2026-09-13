import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import classification_report
from tqdm import tqdm

# Constant system prompt (unchanged)
SYSTEM_PROMPT = (
    "You are a fact-checking assistant. You are given a news claim along with two generated justifications: a 'True Justification' and a 'False Justification'. "
    "Do not consider any other evidence. Your task is to compare these justifications and determine the overall veracity of the claim based solely on them.\n\n"
    "Output your answer in exactly the following format:\n"
    "Claim Veracity: [label]\n\n"
    "Answer with one word: true, false, or half."
)

def construct_prompt(instruction, input_text):
    """
    Construct the final prompt by combining the system prompt and the user prompt.
    The user prompt is built from the dataset's instruction and input text.
    """
    user_prompt = f"{instruction}\n\n{input_text}"
    return f"{SYSTEM_PROMPT}\n\n{user_prompt}"

def clean_response(response):
    """
    Extract and standardize the veracity label from the model response.
    Expects the model to output a response in the format:
      "Claim Veracity: [label]"
    If not found, it falls back to simply checking for the keywords.
    """
    response = response.lower().strip()
    if response.startswith("claim veracity:"):
        # Get label from the expected format
        label = response.split("claim veracity:")[1].strip().split()[0]
        if label in ["true", "false", "half"]:
            return label
    # Fallback: check for the keywords anywhere in the response
    if "true" in response:
        return "true"
    elif "false" in response:
        return "false"
    elif "half" in response:
        return "half"
    return "half"

def evaluate_model(base_model_path, adapter_path, test_data_path, output_dir):
    """
    Evaluate the model using test data in the following format:
      - "instruction": Classification guidelines.
      - "input": Contains the claim and its two justifications.
      - "output": The ground truth veracity label.
    The function runs the model on each example, obtains a prediction, 
    computes a classification report, and saves all outputs.
    """
    # Load the tokenizer and set the pad token
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    print("Loading test data...")
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    predictions = []
    true_labels = []
    all_outputs = []

    print("Running predictions...")
    for item in tqdm(test_data):
        try:
            # Extract fields from test data
            instruction = item['instruction']
            input_text = item['input']
            true_label = item['output'].lower()

            # Construct final prompt: system prompt + user prompt built from dataset fields
            prompt = construct_prompt(instruction, input_text)

            # Tokenize prompt and generate response
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                num_return_sequences=1,
                temperature=0.1,
                do_sample=False
            )

            # Decode only the newly generated tokens (after the prompt)
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            predicted_label = clean_response(response)

            output_item = {
                "instruction": instruction,
                "input": input_text,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "model_response": response,
                "prompt": prompt
            }
            all_outputs.append(output_item)
            predictions.append(predicted_label)
            true_labels.append(true_label)
        except Exception as e:
            print(f"Error processing item: {e}")
            # In case of error, add default prediction
            predictions.append("half")
            true_labels.append(true_label)

    # Calculate and print classification metrics
    print("\nClassification Report:")
    report = classification_report(true_labels, predictions, digits=4)
    print(report)

    # Save detailed predictions and metrics
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "predictions_with_responses.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_outputs, f, indent=2, ensure_ascii=False)

    metrics_file = os.path.join(output_dir, "metrics.txt")
    with open(metrics_file, 'w') as f:
        f.write(report)

    print(f"\nOutputs saved to: {output_file}")
    print(f"Metrics saved to: {metrics_file}")

def main():
    # Configure paths for base model, adapter, test data, and output directory.
    base_model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    adapter_path = "saves/justification/lora_plus/rawfc/llama"
    test_data_path = "llama_rawfc_test_llama_fact_transformed_dataset.json"
    output_dir = "inference_results/llama_rawfc/lora_plus"

    evaluate_model(base_model_path, adapter_path, test_data_path, output_dir)

if __name__ == "__main__":
    main()


"""
Fixed inference script with proper response extraction and raw response logging
"""
import argparse
import json
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
import re

# Configuration
MODEL_PATH = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
parser = argparse.ArgumentParser()

parser.add_argument(
    "--adapter_path",
    type=str,
    required=True,
    help="Path to LoRA adapter"
)

parser.add_argument(
    "--test_file",
    type=str,
    required=True,
    help="Path to test file (JSON or JSONL)"
)

parser.add_argument(
    "--output_file",
    type=str,
    required=True,
    help="Path to save prediction results"
)

args = parser.parse_args()

ADAPTER_PATH = args.adapter_path
TEST_FILE = args.test_file
OUTPUT_FILE = args.output_file

# Create output folder if it does not exist
output_dir = os.path.dirname(OUTPUT_FILE)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")


# Load test data
print(f"Loading test data from {TEST_FILE}...")
with open(TEST_FILE, 'r', encoding='utf-8') as f:
    content = f.read().strip()
    if content.startswith('['):
        test_data = json.loads(content)
    else:
        test_data = [json.loads(line) for line in content.split('\n') if line.strip()]

print(f"Loaded {len(test_data)} test samples")

# Load model and tokenizer
print("\nLoading model and tokenizer...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Load LoRA adapter
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

print("Model loaded successfully!\n")

# Llama3 chat template
def format_prompt(instruction, input_text=""):
    """Format prompt using Llama3 template"""
    if input_text:
        prompt = f"{instruction}\n\n{input_text}"
    else:
        prompt = instruction
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return formatted

def extract_label(response):
    """
    Extract only the label from the response.
    The valid labels are: Support_Text, Support_Multimodal, Insufficient_Text, 
    Insufficient_Multimodal, Refute
    """
    # Define valid labels
    valid_labels = [
        "Support_Text",
        "Support_Multimodal", 
        "Insufficient_Text",
        "Insufficient_Multimodal",
        "Refute"
    ]
    
    # Method 1: Look for exact label match (case-insensitive)
    response_lower = response.lower()
    for label in valid_labels:
        if label.lower() in response_lower:
            return label
    
    # Method 2: Try to find label pattern with word boundaries
    for label in valid_labels:
        pattern = r'\b' + re.escape(label) + r'\b'
        if re.search(pattern, response, re.IGNORECASE):
            return label
    
    # Method 3: Check if response itself is a valid label (after stripping)
    response_stripped = response.strip()
    for label in valid_labels:
        if response_stripped.lower() == label.lower():
            return label
    
    # If no valid label found, return the response as-is for debugging
    return response.strip()

# Inference function
def generate_prediction(instruction, input_text="", max_new_tokens=100):
    """Generate prediction for a single sample"""
    prompt = format_prompt(instruction, input_text)
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Get full response (with special tokens removed)
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Get raw response (with special tokens)
    raw_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract assistant's response
    # For Llama-3 format, find the assistant's response after the header
    if "<|start_header_id|>assistant<|end_header_id|>" in full_response:
        # Split by assistant header and take the last part
        response = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        response = response.strip()
    else:
        # Fallback: try to remove the prompt
        # Find where "assistant" appears and take everything after
        if "assistant" in full_response.lower():
            parts = full_response.lower().split("assistant")
            if len(parts) > 1:
                # Find the position in the original string
                pos = full_response.lower().rfind("assistant") + len("assistant")
                response = full_response[pos:].strip()
            else:
                response = full_response
        else:
            response = full_response
    
    # Clean up any remaining special tokens
    response = response.replace("<|eot_id|>", "").strip()
    
    # Extract only the label
    label = extract_label(response)
    
    return {
        'extracted_response': label,
        'full_assistant_response': response,
        'full_response': full_response,
        'raw_response': raw_response
    }

# Run inference
print(f"Running inference on {len(test_data)} samples...\n")
predictions = []
true_labels = []
predicted_labels = []

for i, sample in enumerate(tqdm(test_data)):
    instruction = sample.get('instruction', '')
    input_text = sample.get('input', '')
    true_output = sample.get('output', sample.get('true_output', ''))
    
    # Generate prediction
    result = generate_prediction(instruction, input_text, max_new_tokens=100)
    
    predictions.append({
        'sample_id': i,
        'instruction': instruction,
        'input': input_text,
        'true_output': true_output,
        'predicted_output': result['extracted_response'],
        'full_assistant_response': result['full_assistant_response'],
        'raw_response': result['raw_response']
    })
    
    true_labels.append(true_output)
    predicted_labels.append(result['extracted_response'])
    
    # Print first few predictions for verification
    if i < 5:
        print(f"\nSample {i+1}:")
        print(f"  True: {true_output}")
        print(f"  Pred: {result['extracted_response']}")
        print(f"  Full: {result['full_assistant_response'][:100]}...")
    
    # Save intermediate results every 500 samples
    if (i + 1) % 500 == 0:
        checkpoint_file = f"predictions_checkpoint_{i+1}.json"
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        print(f"\nCheckpoint saved at {i+1} samples to {checkpoint_file}")

# Save final predictions
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(predictions, f, ensure_ascii=False, indent=2)

print(f"\n\nPredictions saved to {OUTPUT_FILE}")

# Analysis
from collections import Counter

print("\n" + "="*80)
print("QUICK ANALYSIS")
print("="*80)

# Count empty predictions
empty_preds = sum(1 for p in predicted_labels if not p.strip())
print(f"Empty predictions: {empty_preds}/{len(predicted_labels)}")

# Show prediction distribution
print("\nPredicted label distribution:")
pred_counter = Counter(predicted_labels)
for label, count in pred_counter.most_common():
    print(f"  {label}: {count}")

# Show true label distribution
print("\nTrue label distribution:")
true_counter = Counter(true_labels)
for label, count in true_counter.most_common():
    print(f"  {label}: {count}")

# Exact match accuracy
exact_matches = sum(1 for t, p in zip(true_labels, predicted_labels) if t.strip() == p.strip())
print(f"\nExact Match Accuracy: {exact_matches}/{len(true_labels)} = {exact_matches/len(true_labels):.4f}")

# Check for invalid labels
valid_labels = ["Support_Text", "Support_Multimodal", "Insufficient_Text", "Insufficient_Multimodal", "Refute"]
invalid_preds = [p for p in predicted_labels if p not in valid_labels]
if invalid_preds:
    print(f"\nInvalid predictions found: {len(invalid_preds)}")
    print("Sample invalid predictions:")
    for pred in invalid_preds[:5]:
        print(f"  {pred[:100]}")

print("\n" + "="*80)
print("Inference complete!")
print(f"Results saved to: {OUTPUT_FILE}")
print("="*80)

"""
Inference script for Llama 3.1 8B Instruct trained model on Verite dataset
→ Three-class classification: true, miscaptioned, out-of-context
→ Robust label extraction - extracts from raw_response only
→ Raw response logging
→ Classification report saved automatically
"""

import argparse
import json
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
import re
from sklearn.metrics import classification_report

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

# Llama3 prompt format
def format_prompt(instruction, input_text=""):
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


# -------------------------------------------------------------------
#  ROBUST LABEL EXTRACTION (VERITE)
# -------------------------------------------------------------------
def extract_label(response):
    """
    Extract a normalized Verite label:
        true, miscaptioned, out-of-context
    Handles uppercase/lowercase and synonyms.
    """

    resp = response.strip().lower()

    LABEL_TRUE = "true"
    LABEL_MISCAPTIONED = "miscaptioned"
    LABEL_OUT_OF_CONTEXT = "out-of-context"

    true_aliases = [
        "true", "accurate", "correct", "verified",
        "factual", "real", "authentic", "valid"
    ]

    miscaptioned_aliases = [
        "miscaptioned", "mis-captioned", "false caption",
        "wrong caption", "incorrect caption", "misleading",
        "false", "fake", "fabricated", "misinformation"
    ]

    out_of_context_aliases = [
        "out-of-context", "out of context", "outofcontext",
        "wrong context", "incorrect context", "miscontextualized",
        "context error", "recontextualized"
    ]

    # Exact match
    if resp == "true":
        return LABEL_TRUE
    if resp == "miscaptioned":
        return LABEL_MISCAPTIONED
    if resp == "out-of-context":
        return LABEL_OUT_OF_CONTEXT

    # Synonym matching (order matters - check most specific first)
    for alias in out_of_context_aliases:
        if alias in resp:
            return LABEL_OUT_OF_CONTEXT

    for alias in miscaptioned_aliases:
        if alias in resp:
            return LABEL_MISCAPTIONED

    for alias in true_aliases:
        if alias in resp:
            return LABEL_TRUE

    # Default fallback → out-of-context
    return LABEL_OUT_OF_CONTEXT


def extract_answer_from_raw(raw_response):
    """
    Extract ONLY the model's actual answer from raw_response
    For Llama: Look for pattern <|start_header_id|>assistant<|end_header_id|> answer<|eot_id|>
    """
    # Pattern: <|start_header_id|>assistant<|end_header_id|> {actual_answer}<|eot_id|>
    if "<|start_header_id|>assistant<|end_header_id|>" in raw_response:
        # Split by assistant header and take everything after
        after_assistant = raw_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        # Remove <|eot_id|> token
        answer = after_assistant.replace("<|eot_id|>", "").strip()
        return answer
    
    return raw_response.strip()


# Inference function
def generate_prediction(instruction, input_text="", max_new_tokens=100):
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
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    raw_response = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # CRITICAL: Extract actual answer from raw_response (which has the assistant marker)
    actual_answer = extract_answer_from_raw(raw_response)
    
    # Extract label from the actual answer only
    label = extract_label(actual_answer)
    
    return {
        'extracted_response': label,
        'actual_answer': actual_answer,  # The model's actual response
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
    true_output = sample.get('output', sample.get('true_output', '')).strip()

    result = generate_prediction(instruction, input_text, max_new_tokens=100)

    predictions.append({
        'sample_id': i,
        'instruction': instruction,
        'input': input_text,
        'true_output': true_output,
        'predicted_output': result['extracted_response'],
        'actual_answer': result['actual_answer'],
        'raw_response': result['raw_response']
    })

    true_labels.append(true_output)
    predicted_labels.append(result['extracted_response'])

    if i < 5:
        print(f"\nSample {i+1}:")
        print(f"  True: {true_output}")
        print(f"  Pred: {result['extracted_response']}")
        print(f"  Actual answer: '{result['actual_answer']}'")

    if (i + 1) % 500 == 0:
        checkpoint_file = f"predictions_checkpoint_{i+1}.json"
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        print(f"\nCheckpoint saved at {i+1} samples → {checkpoint_file}")

# Save final predictions
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(predictions, f, ensure_ascii=False, indent=2)

print(f"\nPredictions saved to: {OUTPUT_FILE}")

# ==========================================================
# ANALYSIS + CLASSIFICATION REPORT
# ==========================================================
from collections import Counter

print("\n" + "="*80)
print("QUICK ANALYSIS")
print("="*80)

empty_preds = sum(1 for p in predicted_labels if not p.strip())
print(f"Empty predictions: {empty_preds}/{len(predicted_labels)}")

print("\nPredicted label distribution:")
for label, count in Counter(predicted_labels).most_common():
    print(f"  {label}: {count}")

print("\nTrue label distribution:")
for label, count in Counter(true_labels).most_common():
    print(f"  {label}: {count}")

# Normalize labels
y_true = [lbl.lower() for lbl in true_labels]
y_pred = [lbl.lower() for lbl in predicted_labels]

valid_set = ["true", "miscaptioned", "out-of-context"]

y_true_clean = [lbl if lbl in valid_set else "out-of-context" for lbl in y_true]
y_pred_clean = [lbl if lbl in valid_set else "out-of-context" for lbl in y_pred]

# Generate classification report
report = classification_report(
    y_true_clean, 
    y_pred_clean,
    labels=["true", "miscaptioned", "out-of-context"],
    target_names=["true", "miscaptioned", "out-of-context"]
)

print("\n" + report)

# Save report
report_path = os.path.join(output_dir, "classification_report.txt")

with open(report_path, "w", encoding="utf-8") as f:
    f.write("Classification Report - Verite Dataset\n")
    f.write("======================================\n\n")
    f.write(report)

print(f"\nClassification report saved to: {report_path}")

print("\n" + "="*80)
print("Inference complete!")
print("="*80)

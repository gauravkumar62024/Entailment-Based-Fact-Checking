"""
Inference script for Mistral Instruct v0.3 trained model on Verite dataset
→ Three-class classification: true, miscaptioned, out-of-context
→ Robust label extraction - extracts from raw_response only
→ Raw response logging
→ Classification report saved automatically
"""

import json
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
import re
import argparse
from sklearn.metrics import classification_report

# Configuration
MODEL_PATH = "unsloth/Mistral-7B-Instruct-v0.3-bnb-4bit"
parser = argparse.ArgumentParser()

parser.add_argument("--adapter_path", type=str, required=True)
parser.add_argument("--test_file", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)

args = parser.parse_args()
ADAPTER_PATH = args.adapter_path
TEST_FILE = args.test_file
OUTPUT_FILE = args.output_file

# Create output folder
output_dir = os.path.dirname(OUTPUT_FILE)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")

# Load test dataset
print(f"Loading test data from {TEST_FILE}...")
with open(TEST_FILE, 'r', encoding='utf-8') as f:
    content = f.read().strip()
    test_data = json.loads(content) if content.startswith('[') else [json.loads(l) for l in content.split('\n') if l.strip()]
print(f"Loaded {len(test_data)} samples")

# Load model & tokenizer
print("\nLoading model and tokenizer...")
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()
print("Model loaded successfully!\n")

# -------------------------------------------------------------------
#   MISTRAL CHAT FORMAT
# -------------------------------------------------------------------
def format_prompt(instruction, input_text=""):
    prompt = f"{instruction}\n\n{input_text}" if input_text else instruction
    return f"[INST] {prompt} [/INST]"

# -------------------------------------------------------------------
#   ROBUST LABEL EXTRACTION (VERITE)
# -------------------------------------------------------------------
def extract_label(response):
    """
    Extract: true, miscaptioned, out-of-context
    Supports uppercase/lowercase + synonyms.
    """
    text = response.strip().lower()

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

    # Exact matches
    if text == "true": return LABEL_TRUE
    if text == "miscaptioned": return LABEL_MISCAPTIONED
    if text == "out-of-context": return LABEL_OUT_OF_CONTEXT

    # Synonym scanning (order matters - check most specific first)
    for a in out_of_context_aliases:
        if a in text:
            return LABEL_OUT_OF_CONTEXT
    
    for a in miscaptioned_aliases:
        if a in text:
            return LABEL_MISCAPTIONED
    
    for a in true_aliases:
        if a in text:
            return LABEL_TRUE

    return LABEL_OUT_OF_CONTEXT  # fallback


def extract_answer_from_raw(raw_response):
    """
    Extract ONLY the model's actual answer from raw_response
    For Mistral: Look for pattern [/INST] answer</s>
    """
    # Pattern: [/INST] {actual_answer}</s>
    if "[/INST]" in raw_response:
        # Split by [/INST] and take everything after
        after_inst = raw_response.split("[/INST]")[-1]
        # Remove </s> token
        answer = after_inst.replace("</s>", "").strip()
        return answer
    
    return raw_response.strip()


# -------------------------------------------------------------------
#   INFERENCE
# -------------------------------------------------------------------
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
        )

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    raw_response = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # CRITICAL: Extract actual answer from raw_response (which has the [/INST] marker)
    actual_answer = extract_answer_from_raw(raw_response)
    
    # Extract label from the actual answer only
    label = extract_label(actual_answer)

    return {
        "extracted_response": label,
        "actual_answer": actual_answer,  # The model's actual response
        "full_response": full_response,
        "raw_response": raw_response
    }

# -------------------------------------------------------------------
#   RUN INFERENCE
# -------------------------------------------------------------------
print(f"Running inference on {len(test_data)} samples...\n")

predictions = []
true_labels = []
predicted_labels = []

for i, sample in enumerate(tqdm(test_data)):
    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    true_output = sample.get("output", sample.get("true_output", "")).strip()

    result = generate_prediction(instruction, input_text)

    predictions.append({
        "sample_id": i,
        "instruction": instruction,
        "input": input_text,
        "true_output": true_output,
        "predicted_output": result["extracted_response"],
        "actual_answer": result["actual_answer"],
        "raw_response": result["raw_response"]
    })

    true_labels.append(true_output)
    predicted_labels.append(result["extracted_response"])

    if i < 5:
        print(f"\nExample {i+1}")
        print(f"  True: {true_output}")
        print(f"  Pred: {result['extracted_response']}")
        print(f"  Actual answer: '{result['actual_answer']}'")

# Save predictions
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(predictions, f, ensure_ascii=False, indent=2)

print(f"\nPredictions saved to: {OUTPUT_FILE}")

# -------------------------------------------------------------------
#   ANALYSIS + CLASSIFICATION REPORT
# -------------------------------------------------------------------
from collections import Counter

print("\n" + "="*80)
print("QUICK ANALYSIS")
print("="*80)

print("\nPrediction distribution:")
print(Counter(predicted_labels))

print("\nTrue label distribution:")
print(Counter(true_labels))

# Normalize
y_true = [lbl.lower() for lbl in true_labels]
y_pred = [lbl.lower() for lbl in predicted_labels]

valid = ["true", "miscaptioned", "out-of-context"]

y_true_clean = [lbl if lbl in valid else "out-of-context" for lbl in y_true]
y_pred_clean = [lbl if lbl in valid else "out-of-context" for lbl in y_pred]

report = classification_report(
    y_true_clean,
    y_pred_clean,
    labels=["true", "miscaptioned", "out-of-context"],
    target_names=["true", "miscaptioned", "out-of-context"]
)

print("\n" + report)

# Save classification report
report_file = os.path.join(output_dir, "classification_report.txt")
with open(report_file, "w", encoding="utf-8") as f:
    f.write("Classification Report - Verite Dataset\n")
    f.write("======================================\n\n")
    f.write(report)

print(f"\nClassification report saved to: {report_file}")

print("\n" + "="*80)
print("Inference complete!")
print("="*80)

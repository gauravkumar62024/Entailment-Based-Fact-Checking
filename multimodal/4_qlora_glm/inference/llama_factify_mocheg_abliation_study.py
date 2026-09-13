"""
Fixed inference script with proper response extraction, raw response logging,
and classification report saving.

Extended with minimal changes:
- --input_mode: support | refute | claim
- mode-specific instruction prompt
- mode-specific input construction (claim + support, claim + refute, claim only)
- mode-specific classification report filename
"""

import argparse
import json
import torch
import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
from sklearn.metrics import classification_report
from collections import Counter

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

# NEW: input mode
parser.add_argument(
    "--input_mode",
    type=str,
    default="support",
    choices=["support", "refute", "claim"],
    help="Input construction mode: support=claim+support justification, refute=claim+refute justification, claim=claim only"
)

args = parser.parse_args()

ADAPTER_PATH = args.adapter_path
TEST_FILE = args.test_file
OUTPUT_FILE = args.output_file
INPUT_MODE = args.input_mode

# Create output folder if it does not exist
output_dir = os.path.dirname(OUTPUT_FILE)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")

# -----------------------------
# MODE-SPECIFIC INSTRUCTIONS
# -----------------------------
def get_instruction_for_mode(mode: str) -> str:
    """
    The dataset instruction assumes both support+refute are provided.
    We change it depending on the task mode.
    Output must be exactly one label: supported, refuted, NEI.
    """
    if mode == "support":
        return (
            "You are a fact-checking assistant. Given a claim and a supporting justification, "
            "classify the claim's veracity based only on the supporting justification provided. "
            "Your output must be exactly one of the following labels: supported, refuted, and NEI (Not enough information). "
            "Return only the label."
        )
    if mode == "refute":
        return (
            "You are a fact-checking assistant. Given a claim and a refuting justification, "
            "classify the claim's veracity based only on the refuting justification provided. "
            "Your output must be exactly one of the following labels: supported, refuted, and NEI (Not enough information). "
            "Return only the label."
        )
    # mode == "claim"
    return (
        "You are a fact-checking assistant. Given only a claim (no evidence), "
        "classify its veracity. If there is not enough information to decide, output NEI. "
        "Your output must be exactly one of the following labels: supported, refuted, and NEI (Not enough information). "
        "Return only the label."
    )

# -----------------------------
# INPUT PARSING / BUILDING
# -----------------------------
def _extract_section(text: str, header: str) -> str:
    """
    Extract a section like:
      Claim:
      ...
    stopping at the next known header or end.
    """
    if not isinstance(text, str):
        return ""

    # Match "Header:" then capture until next header or end
    pattern = rf"(?:^|\n)\s*{re.escape(header)}\s*:\s*(.+?)(?=\n\s*(Claim|Support Justification|Refute Justification)\s*:|\Z)"
    m = re.search(pattern, text, flags=re.I | re.S)
    return m.group(1).strip() if m else ""

def parse_claim_support_refute(sample_input: str):
    claim = _extract_section(sample_input, "Claim")
    support = _extract_section(sample_input, "Support Justification")
    refute = _extract_section(sample_input, "Refute Justification")
    return claim, support, refute

def build_input_text(sample: dict, mode: str) -> str:
    """
    Dataset typically stores everything in sample["input"].
    We parse it and keep only what we need per mode.
    """
    raw_input = sample.get("input", "")
    if not isinstance(raw_input, str):
        raw_input = ""

    claim, support, refute = parse_claim_support_refute(raw_input)

    # If claim couldn't be parsed, fallback to raw input (best effort)
    if not claim:
        if mode == "claim":
            return raw_input.strip()
        return raw_input.strip()

    if mode == "claim":
        return f"Claim:\n{claim}".strip()

    if mode == "support":
        return f"Claim:\n{claim}\n\nSupport Justification:\n{support}".strip()

    # mode == "refute"
    return f"Claim:\n{claim}\n\nRefute Justification:\n{refute}".strip()

# -----------------------------
# Load test data
# -----------------------------
print(f"Loading test data from {TEST_FILE}...")
with open(TEST_FILE, 'r', encoding='utf-8') as f:
    content = f.read().strip()
    if content.startswith('['):
        test_data = json.loads(content)
    else:
        test_data = [json.loads(line) for line in content.split('\n') if line.strip()]

print(f"Loaded {len(test_data)} test samples")

# -----------------------------
# Load model and tokenizer
# -----------------------------
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

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

print("Model loaded successfully!\n")

# -----------------------------
# Llama3 prompt format
# -----------------------------
def format_prompt(instruction, input_text=""):
    if input_text:
        prompt = f"{instruction}\n\n{input_text}"
    else:
        prompt = instruction

    messages = [{"role": "user", "content": prompt}]

    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return formatted

# -------------------------------------------------------------------
#  ROBUST LABEL EXTRACTION (MOCHENG VERSION)
# -------------------------------------------------------------------
def extract_label(response):
    """
    Extract a normalized MoCheng label:
        supported, NEI, refuted
    Handles uppercase/lowercase and synonyms.
    """
    resp = response.strip().lower()

    LABEL_SUPPORTED = "supported"
    LABEL_NEI = "NEI"
    LABEL_REFUTED = "refuted"

    supported_aliases = [
        "supported", "support", "supporting",
        "is supported", "true", "correct", "verified"
    ]

    nei_aliases = [
        "nei", "n.e.i", "not enough information",
        "not enough info", "insufficient information",
        "insufficient info", "not enough evidence",
        "unknown", "cannot determine", "can't determine",
        "undetermined", "uncertain", "no evidence",
        "not sure"
    ]

    refuted_aliases = [
        "refuted", "refute", "refuting", "false",
        "incorrect", "not true", "is false", "disproved"
    ]

    # Exact match
    if resp == "supported":
        return LABEL_SUPPORTED
    if resp == "nei":
        return LABEL_NEI
    if resp == "refuted":
        return LABEL_REFUTED

    # Synonym matching
    for alias in supported_aliases:
        if alias in resp:
            return LABEL_SUPPORTED

    for alias in nei_aliases:
        if alias in resp:
            return LABEL_NEI

    for alias in refuted_aliases:
        if alias in resp:
            return LABEL_REFUTED

    # Default fallback → NEI (common in fact-checking)
    return LABEL_NEI

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

    # Extract assistant text
    if "<|start_header_id|>assistant<|end_header_id|>" in full_response:
        response = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
    else:
        if "assistant" in full_response.lower():
            pos = full_response.lower().rfind("assistant") + len("assistant")
            response = full_response[pos:].strip()
        else:
            response = full_response

    response = response.replace("<|eot_id|>", "").strip()

    label = extract_label(response)

    return {
        "extracted_response": label,
        "full_assistant_response": response,
        "full_response": full_response,
        "raw_response": raw_response
    }

# -----------------------------
# Run inference
# -----------------------------
print(f"Running inference on {len(test_data)} samples... (mode={INPUT_MODE})\n")

predictions = []
true_labels = []
predicted_labels = []

# Use one fixed instruction per mode (instead of sample["instruction"])
mode_instruction = get_instruction_for_mode(INPUT_MODE)

for i, sample in enumerate(tqdm(test_data)):
    input_text = build_input_text(sample, INPUT_MODE)
    true_output = sample.get("output", sample.get("true_output", "")).strip()

    result = generate_prediction(mode_instruction, input_text, max_new_tokens=100)

    predictions.append({
        "sample_id": i,
        "input_mode": INPUT_MODE,
        "instruction": mode_instruction,
        "input": input_text,
        "true_output": true_output,
        "predicted_output": result["extracted_response"],
        "full_assistant_response": result["full_assistant_response"],
        "raw_response": result["raw_response"]
    })

    true_labels.append(true_output)
    predicted_labels.append(result["extracted_response"])

    if i < 5:
        print(f"\nSample {i+1}:")
        print(f"  True: {true_output}")
        print(f"  Pred: {result['extracted_response']}")
        print(f"  Full: {result['full_assistant_response'][:120]}...")

    if (i + 1) % 500 == 0:
        checkpoint_file = f"predictions_checkpoint_{INPUT_MODE}_{i+1}.json"
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        print(f"\nCheckpoint saved at {i+1} samples → {checkpoint_file}")

# Save final predictions
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(predictions, f, ensure_ascii=False, indent=2)

print(f"\nPredictions saved to: {OUTPUT_FILE}")

# ==========================================================
# ANALYSIS + CLASSIFICATION REPORT
# ==========================================================
print("\n" + "=" * 80)
print("QUICK ANALYSIS")
print("=" * 80)

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

valid_set = ["supported", "nei", "refuted"]

y_true_clean = [lbl if lbl in valid_set else "nei" for lbl in y_true]
y_pred_clean = [lbl if lbl in valid_set else "nei" for lbl in y_pred]

# Generate classification report
report = classification_report(
    y_true_clean,
    y_pred_clean,
    labels=["supported", "nei", "refuted"],
    target_names=["supported", "NEI", "refuted"]
)

print("\n" + report)

# Save report (mode-specific filename)
report_path = os.path.join(output_dir, f"classification_report_{INPUT_MODE}.txt")

with open(report_path, "w", encoding="utf-8") as f:
    f.write("Classification Report\n")
    f.write("=====================\n\n")
    f.write(f"Input mode: {INPUT_MODE}\n\n")
    f.write(report)

print(f"\nClassification report saved to: {report_path}")

print("\n" + "=" * 80)
print("Inference complete!")
print("=" * 80)


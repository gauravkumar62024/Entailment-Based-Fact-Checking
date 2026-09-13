import json
import os

# ✅ Your allowed Factify labels
VALID_LABELS = {
    "supported",
    "refuted",
    "NEI",
}

# ✅ LLaMA-Factory template
INSTRUCTION_TEMPLATE = (
    "You are a fact-checking assistant. Given a claim, classify its veracity on the basis of "
    "the support and refute justification provided. Your output must be exactly one of the "
    "following labels: supported, refuted, and NEI (Not enough information). Return only the label."
)

INPUT_TEMPLATE = (
    "Claim:\n{claim}\n\n"
    "Support Justification:\n{support}\n\n"
    "Refute Justification:\n{refute}\n\n"
    "Choose the correct label from:\n"
    "supported\n"
    "refuted\n"
    "NEI\n"
)

def convert_factify_to_jsonl(input_path, output_path):
    """
    Converts a Factify JSON dataset into a LLaMA-Factory-compatible JSONL file.
    """
    print(f"Loading: {input_path}")
    
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_file = open(output_path, "w", encoding="utf-8")

    count = 0
    skipped = 0

    for item in data:
        claim = item.get("claim_text") or item.get("claim") or ""
        support = item.get("supporting_text_evidence") or item.get("support_justification") or ""
        refute = item.get("refuting_text_evidence") or item.get("refute_justification") or ""
        label = str(item.get("label", "")).strip()

        # ✅ Skip invalid label entries
        if label not in VALID_LABELS:
            skipped += 1
            continue

        # ✅ Build the input text
        input_text = INPUT_TEMPLATE.format(
            claim=claim.strip(),
            support=support.strip(),
            refute=refute.strip()
        )

        # ✅ Build the final record
        record = {
            "instruction": INSTRUCTION_TEMPLATE,
            "input": input_text,
            "output": label
        }

        # ✅ Write JSONL
        out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        count += 1

    out_file.close()

    print(f"✅ Saved: {output_path}")
    print(f"✅ Converted entries: {count}")
    print(f"⚠️ Skipped invalid-label entries: {skipped}\n")


# ✅ Example usage — update your dataset filenames here
convert_factify_to_jsonl(
    "mocheg_justification/palegemma/train_justifications_final.json",
    "train_llamafactory_palegemma_mocheng.json"
)

convert_factify_to_jsonl(
    "mocheg_justification/palegemma/val_justifications_final.json",
    "val_llamafactory_palegemma_mocheng.json"
)

convert_factify_to_jsonl(
    "mocheg_justification/palegemma/test_justifications_final.json",
    "test_llamafactory_palegemma_mocheng.json"
)


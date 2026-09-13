#!/usr/bin/env python3
"""
Convert step-2 multimodal justification files into LLaMA-Factory (alpaca) format
for QLoRA fine-tuning of the GLMs (Llama-3.1-8B / Mistral-7B) -- Table 8, "QLoRA" block.

Covers all three multimodal datasets with one CLI. It replaces the earlier
per-dataset scripts (to_llamafactory_factify.py / to_llamafactory_mocheg.py),
which are kept alongside it for reference; this one additionally handles VERITE.

Input  : the *_final_justification.json / *_justifications.json produced by
         ../../2_justification_generation/, i.e. records carrying a claim, a
         support justification, a refute justification and a gold label.
Output : JSON-lines with {"instruction", "input", "output"} -- register the file
         in LLaMA-Factory's data/dataset_info.json (see env/dataset_info.project.json).

Examples
--------
    # Factify-2, PaliGemma justifications, all three splits
    python to_llamafactory.py --dataset factify2 \
        --input  ../../data/justifications/factify2/paligemma/train_justifications_paligemma2_final_justification.json \
        --output train_llamafactory_paligemma_factify.json

    # MOCHEG, Qwen2.5-VL
    python to_llamafactory.py --dataset mocheg \
        --input ../../data/justifications/mocheg/qwen2vl/train_justifications.json \
        --output train_llamafactory_qwen2vl_mocheng.json

    # VERITE
    python to_llamafactory.py --dataset verite \
        --input ../../data/justifications/verite/verite_full_justifications_qwen2vl.json \
        --output qwen2vl_verite_full_train_with_labels_llama_factory.json
"""

import argparse
import json
import os
import sys

# ---------------------------------------------------------------------------
# Label spaces, exactly as reported in Table 10 of the paper.
# ---------------------------------------------------------------------------
LABEL_SPACE = {
    "factify2": [
        "Support_Text",
        "Support_Multimodal",
        "Insufficient_Text",
        "Insufficient_Multimodal",
        "Refute",
    ],
    "mocheg": ["supported", "refuted", "NEI"],
    "verite": ["true", "miscaptioned", "out-of-context"],
}

# Some source files use different surface forms for the same class.
LABEL_ALIASES = {
    "mocheg": {
        "supported": "supported", "support": "supported", "Supported": "supported",
        "refuted": "refuted", "refute": "refuted", "Refuted": "refuted",
        "nei": "NEI", "NEI": "NEI", "not enough info": "NEI",
        "not_enough_info": "NEI", "Not Enough Info": "NEI",
    },
    "verite": {
        "true": "true", "True": "true", "TRUE": "true",
        "miscaptioned": "miscaptioned", "MisCaptioned": "miscaptioned", "MC": "miscaptioned",
        "out-of-context": "out-of-context", "out_of_context": "out-of-context",
        "Out-of-Context": "out-of-context", "OOC": "out-of-context",
    },
    "factify2": {},
}

INSTRUCTION = (
    "You are a fact-checking assistant. Given a claim, classify its veracity on the basis of "
    "the support and refute justification provided. Your output must be exactly one of the "
    "following labels: {labels}. Return ONLY the label."
)

INPUT_TEMPLATE = (
    "Claim:\n{claim}\n\n"
    "Support Justification:\n{support}\n\n"
    "Refute Justification:\n{refute}\n\n"
    "Choose the correct label from:\n{label_list}\n\n"
    "Answer with only the label."
)

# Field names vary a little across the step-2 writers; try each in order.
CLAIM_KEYS = ("claim_text", "claim", "caption")
SUPPORT_KEYS = ("support_justification", "supporting_text_evidence", "true_justification",
                "support", "supporting_evidence")
REFUTE_KEYS = ("refute_justification", "refuting_text_evidence", "false_justification",
               "refute", "refuting_evidence")


def first_present(item, keys, default=""):
    for k in keys:
        v = item.get(k)
        if v:
            return v if isinstance(v, str) else " ".join(map(str, v))
    return default


def normalise_label(raw, dataset):
    raw = str(raw).strip()
    aliases = LABEL_ALIASES[dataset]
    if raw in aliases:
        return aliases[raw]
    if raw.lower() in aliases:
        return aliases[raw.lower()]
    return raw


def convert(input_path, output_path, dataset, keep_unlabelled=False):
    labels = LABEL_SPACE[dataset]
    instruction = INSTRUCTION.format(labels=", ".join(labels))
    label_list = "\n".join(labels)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    kept = skipped = 0
    skipped_labels = {}
    with open(output_path, "w", encoding="utf-8") as out:
        for item in data:
            label = normalise_label(item.get("label", ""), dataset)
            if label not in labels:
                if not keep_unlabelled:
                    skipped += 1
                    skipped_labels[label] = skipped_labels.get(label, 0) + 1
                    continue
            record = {
                "instruction": instruction,
                "input": INPUT_TEMPLATE.format(
                    claim=first_present(item, CLAIM_KEYS).strip(),
                    support=first_present(item, SUPPORT_KEYS).strip(),
                    refute=first_present(item, REFUTE_KEYS).strip(),
                    label_list=label_list,
                ),
                "output": label,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    print(f"{input_path}\n  -> {output_path}")
    print(f"  converted: {kept}   skipped (label outside {dataset} label space): {skipped}")
    if skipped_labels:
        top = sorted(skipped_labels.items(), key=lambda kv: -kv[1])[:5]
        print(f"  most common skipped labels: {top}")
    if kept == 0:
        print("  WARNING: nothing was written -- check --dataset and the input's field names.",
              file=sys.stderr)
    return kept


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(LABEL_SPACE),
                   help="Which label space to use")
    p.add_argument("--input", required=True, help="Step-2 justification JSON")
    p.add_argument("--output", required=True, help="LLaMA-Factory JSONL to write")
    p.add_argument("--keep_unlabelled", action="store_true",
                   help="Keep records whose label is outside the label space (default: drop)")
    args = p.parse_args()
    convert(args.input, args.output, args.dataset, args.keep_unlabelled)


if __name__ == "__main__":
    main()

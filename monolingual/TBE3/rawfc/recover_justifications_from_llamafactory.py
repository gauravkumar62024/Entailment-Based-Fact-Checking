#!/usr/bin/env python3
"""
Recover ELM-format justifications from the LLaMA-Factory training files.

TBE-3 step 3 wants records shaped

    {"claim", "label", "support_justification", "refute_justification"}

but only the *test* split is committed in that form. The *train* split survives
only inside the LLaMA-Factory files under
`TBE3/LoRA_and_LoRA+_finetuning/data/`, where the same two justifications are
embedded in a single `input` string:

    Claim: <claim>

    True Justification:
    <support>

    False Justification:
    <refute>

    half_justification:          <- LIAR-RAW only, ignored here
    <...>

This script parses those back out so the ELM trainers can consume them, rather
than regenerating ~2k GLM calls that already exist.

Example
-------
    python recover_justifications_from_llamafactory.py \
        --input  ../LoRA_and_LoRA+_finetuning/data/rawfc_train_with_justi_for_lafct_formate_llama.json \
        --output cleaned_data/llama/train_llama_cleaned.json
"""

import argparse
import json
import os
import re
import sys

CLAIM_RX = re.compile(r"^Claim:\s*(.*?)(?=\n\s*\n|\nTrue Justification:)", re.S)
SUPPORT_RX = re.compile(r"True Justification:\s*\n?(.*?)(?=\n\s*False Justification:|\Z)", re.S)
REFUTE_RX = re.compile(r"False Justification:\s*\n?(.*?)(?=\n\s*half_justification:|\Z)", re.S)


def parse_record(rec):
    text = rec.get("input", "")
    label = str(rec.get("output", "")).strip()

    claim = CLAIM_RX.search(text)
    support = SUPPORT_RX.search(text)
    refute = REFUTE_RX.search(text)

    return {
        "claim": claim.group(1).strip() if claim else "",
        "label": label,
        "support_justification": support.group(1).strip() if support else "",
        "refute_justification": refute.group(1).strip() if refute else "",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="LLaMA-Factory json (instruction/input/output)")
    p.add_argument("--output", required=True, help="ELM-format json to write")
    p.add_argument("--drop_empty", action="store_true",
                   help="Drop records where the claim or either justification is empty")
    args = p.parse_args()

    data = json.load(open(args.input, encoding="utf-8"))
    out, dropped = [], 0
    stats = {"no_claim": 0, "no_support": 0, "no_refute": 0}

    for rec in data:
        r = parse_record(rec)
        if not r["claim"]:
            stats["no_claim"] += 1
        if not r["support_justification"]:
            stats["no_support"] += 1
        if not r["refute_justification"]:
            stats["no_refute"] += 1
        if args.drop_empty and not (r["claim"] and r["support_justification"] and r["refute_justification"]):
            dropped += 1
            continue
        out.append(r)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    json.dump(out, open(args.output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print(f"{args.input}\n  -> {args.output}")
    print(f"  in: {len(data)}   out: {len(out)}   dropped: {dropped}")
    print(f"  parse misses -> claim: {stats['no_claim']}  support: {stats['no_support']}  refute: {stats['no_refute']}")
    labels = {}
    for r in out:
        labels[r["label"]] = labels.get(r["label"], 0) + 1
    print(f"  label distribution: {dict(sorted(labels.items()))}")
    if not out:
        print("  WARNING: nothing recovered -- check the input format.", file=sys.stderr)


if __name__ == "__main__":
    main()

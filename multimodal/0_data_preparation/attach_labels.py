#!/usr/bin/env python3
"""
Attach gold veracity labels to step-2 justification files.

The justification writers in ../2_justification_generation/ carry the claim and
the two justifications but not always the gold label -- MOCHEG and VERITE keep
their labels in the dataset's own corpus file. This joins them back on the id
column so the result can be fed to ../3_train_elm/ or to
../4_qlora_glm/prepare_data/to_llamafactory.py.

Examples
--------
    # MOCHEG: labels live in the Corpus CSV shipped with the dataset
    python attach_labels.py \
        --justifications ../data/justifications/mocheg/qwen2vl/test_justifications.json \
        --labels /path/to/mocheg/test/Corpus2.csv \
        --just_id_field Id --label_id_field claim_id --label_field cleaned_truthfulness \
        --output test_justifications_with_labels.json

    # VERITE: labels live in VERITE.csv, keyed by the unnamed index column
    python attach_labels.py \
        --justifications ../data/justifications/verite/verite_full_justifications_qwen2vl.json \
        --labels /path/to/VERITE/VERITE.csv \
        --just_id_field claim_id --label_id_field "Unnamed: 0" --label_field label \
        --output qwen2vl_verite_full_with_labels.json
"""

import argparse
import csv
import json
import os
import sys


def load_records(path):
    if path.lower().endswith((".csv", ".tsv")):
        delim = "\t" if path.lower().endswith(".tsv") else ","
        with open(path, newline="", encoding="utf-8") as f:
            csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
            return list(csv.DictReader(f, delimiter=delim))
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--justifications", required=True, help="Step-2 justification JSON")
    p.add_argument("--labels", required=True, help="CSV/JSON carrying the gold labels")
    p.add_argument("--output", required=True)
    p.add_argument("--just_id_field", default="Id", help="Id field inside --justifications")
    p.add_argument("--label_id_field", default="claim_id", help="Id field inside --labels")
    p.add_argument("--label_field", default="label", help="Label column inside --labels")
    p.add_argument("--drop_unmatched", action="store_true",
                   help="Drop records with no matching label (default: keep with empty label)")
    args = p.parse_args()

    just = load_records(args.justifications)
    labels = load_records(args.labels)

    lut = {}
    for row in labels:
        key = str(row.get(args.label_id_field, "")).strip()
        if key:
            lut[key] = str(row.get(args.label_field, "")).strip()

    out, matched, missing = [], 0, 0
    for item in just:
        key = str(item.get(args.just_id_field, "")).strip()
        if key in lut and lut[key]:
            item["label"] = lut[key]
            matched += 1
            out.append(item)
        else:
            missing += 1
            if not args.drop_unmatched:
                item.setdefault("label", "")
                out.append(item)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"justifications : {len(just)}")
    print(f"labels loaded  : {len(lut)}  (from {args.labels})")
    print(f"matched        : {matched}")
    print(f"unmatched      : {missing}" + ("  [dropped]" if args.drop_unmatched else "  [kept with empty label]"))
    print(f"written        : {len(out)} -> {args.output}")
    if matched == 0:
        print("WARNING: nothing matched -- check --just_id_field / --label_id_field.", file=sys.stderr)


if __name__ == "__main__":
    main()

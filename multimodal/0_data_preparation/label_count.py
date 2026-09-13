# import json
# from collections import Counter

# # Paths to your dataset files
# files = [
#     "/data4/adityaks/multimodal_fact_checking/other_datasets/results/verite_justifications/qwen2vl_verite_full_train_with_labels.json",
#     "/data4/adityaks/multimodal_fact_checking/other_datasets/results/verite_justifications/qwen2vl_verite_full_val_with_labels.json",
#     "/data4/adityaks/multimodal_fact_checking/other_datasets/results/verite_justifications/qwen2vl_verite_full_test_with_labels.json"
# ]

# total_samples = 0
# label_counter = Counter()

# for file_path in files:
#     with open(file_path, "r", encoding="utf-8") as f:
#         data = json.load(f)  # Load JSON list
        
#         total_samples += len(data)
#         for item in data:
#             label_counter[item["original_label"]] += 1

# print("Total number of samples:", total_samples)
# print("Label counts:", dict(label_counter))

import json
import csv
import os
import argparse
from collections import Counter, defaultdict


# ----------------------------------------------------------------------
# Label normalization (same style as your training scripts)
# ----------------------------------------------------------------------
LABEL_MAP = {
    "true": 0,
    "miscaptioned": 1,
    "out-of-context": 2,
}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def normalize_label_text(raw: str) -> str:
    """
    Map various label strings into canonical keys:
        'true', 'miscaptioned', 'out-of-context'
    """
    if raw is None:
        return "miscaptioned"

    t = str(raw).strip().lower()

    # ---- TRUE-LIKE ----
    if t in {
        "true", "true.", "correct", "yes", "verified",
        "supported", "support", "support.", "entails", "entailment"
    }:
        return "true"

    # ---- MISCAPTIONED-LIKE ----
    if t in {
        "miscaptioned", "mis-captioned", "mis_captioned",
        "false", "false.", "fake", "incorrect",
        "refuted", "refute", "refute.", "contradiction"
    }:
        return "miscaptioned"

    # ---- OUT-OF-CONTEXT-LIKE ----
    if t in {
        "out-of-context", "out_of_context", "out of context",
        "context-shifted", "context shifted"
    }:
        return "out-of-context"

    # ---- NEI / unknown / neutral – here mapped to "true" (you can change if you want)
    if t in {
        "nei", "n.e.i", "not enough info", "not-enough-info",
        "unknown", "uncertain", "neutral", "unrelated"
    }:
        return "true"

    # Default fall-back
    return "miscaptioned"


# ----------------------------------------------------------------------
# Load VERITE labels from CSV
# ----------------------------------------------------------------------
def load_verite_labels(csv_path):
    """
    Load VERITE gold labels from the original CSV.

    Assumption (same as your training pipeline):
      - claim_id in JSON == CSV row index as string ("0", "1", "2", ...)

    We detect the label column name (either 'label' or 'veracity_label').
    Returns:
        mapping: dict[str, str]  # claim_id -> raw label string from CSV
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"VERITE CSV not found at: {csv_path}")

    mapping = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in labels CSV: {csv_path}")

        # Find label column
        lower_names = [c.lower() for c in reader.fieldnames]
        lab_col = None
        for cand in ["label", "veracity_label"]:
            if cand in lower_names:
                lab_col = reader.fieldnames[lower_names.index(cand)]
                break

        if lab_col is None:
            raise ValueError(
                "Could not find a label column in VERITE labels CSV "
                "(tried: 'label', 'veracity_label')"
            )

        # Enumerate rows: index i becomes the claim_id
        for i, row in enumerate(reader):
            lab = str(row.get(lab_col, "")).strip()
            if not lab:
                continue
            cid = str(i)  # this is the claim_id used in JSON
            mapping[cid] = lab

    print(f"✅ Loaded {len(mapping)} label entries from CSV: {csv_path}")
    return mapping


# ----------------------------------------------------------------------
# Analyze JSON files using claim_id -> label mapping
# ----------------------------------------------------------------------
def analyze_json_files(json_paths, id2label_raw):
    """
    For each JSON file:
      - For each sample, look up claim_id in id2label_raw
      - Normalize the label
      - Count per-label statistics.
    Also compute global stats across all files.
    """
    global_counts = Counter()
    per_file_counts = {}
    per_file_missing_ids = {}
    per_file_total = {}

    for path in json_paths:
        if not os.path.exists(path):
            print(f"⚠️ JSON file not found: {path}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        file_counts = Counter()
        missing_ids = []
        total = 0

        for item in data:
            total += 1
            cid = str(item.get("claim_id", "")).strip()

            if cid == "":
                missing_ids.append("(empty-claim-id)")
                continue

            raw_label = id2label_raw.get(cid, None)
            if raw_label is None:
                missing_ids.append(cid)
                continue

            canonical = normalize_label_text(raw_label)  # "true" / "miscaptioned" / "out-of-context"
            file_counts[canonical] += 1
            global_counts[canonical] += 1

        per_file_counts[path] = file_counts
        per_file_missing_ids[path] = missing_ids
        per_file_total[path] = total

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PER-FILE LABEL STATISTICS (normalized labels)")
    print("=" * 80)

    for path in json_paths:
        if path not in per_file_counts:
            continue

        print(f"\nFile: {path}")
        total = per_file_total[path]
        counts = per_file_counts[path]
        missing = per_file_missing_ids[path]

        print(f"  Total samples in JSON: {total}")
        print(f"  Samples with matched label (via claim_id): {sum(counts.values())}")
        print(f"  Samples with missing / unmatched claim_id: {len(missing)}")

        for label in ["true", "miscaptioned", "out-of-context"]:
            c = counts.get(label, 0)
            pct = (c / total * 100.0) if total > 0 else 0.0
            print(f"    {label:15s}: {c:5d}  ({pct:6.2f}%)")

        # Optionally show some missing IDs (if any)
        if missing:
            show_n = min(5, len(missing))
            print(f"  Example missing claim_ids (showing up to {show_n}): {missing[:show_n]}")

    print("\n" + "=" * 80)
    print("GLOBAL LABEL STATISTICS (across all JSON files, normalized)")
    print("=" * 80)

    total_global = sum(global_counts.values())
    print(f"Total matched samples across all JSON files: {total_global}")
    for label in ["true", "miscaptioned", "out-of-context"]:
        c = global_counts.get(label, 0)
        pct = (c / total_global * 100.0) if total_global > 0 else 0.0
        print(f"  {label:15s}: {c:5d}  ({pct:6.2f}%)")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Trace VERITE labels by claim_id for one or more JSON files."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to VERITE.csv (original veracity annotations)."
    )
    parser.add_argument(
        "--json_files",
        type=str,
        nargs="+",
        required=True,
        help="One or more JSON files that contain 'claim_id' fields "
             "(e.g., train/val/test JSONs)."
    )

    args = parser.parse_args()

    # 1) Load CSV labels
    id2label_raw = load_verite_labels(args.csv_path)

    # 2) Analyze JSON files
    analyze_json_files(args.json_files, id2label_raw)


if __name__ == "__main__":
    main()

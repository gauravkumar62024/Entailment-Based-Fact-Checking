#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Attach VERITE labels from VERITE.csv to VERITE JSON files.

Assumptions:
- VERITE.csv has columns like: 'Unnamed: 0', 'caption', 'image_path', 'label'
- Your JSON has entries like:
    {
      "claim_id": "666",
      "claim_text": "...",
      "support_justification": "...",
      "refute_justification": "..."
    }

We will:
- Treat the CSV ID column (typically 'Unnamed: 0') as the claim_id
- Map: str(id_value) -> label
- Add that label to each JSON entry as "label".
"""

import json
import os
import sys
import pandas as pd
import random


# -------------------------
# 1. Build label mapping
# -------------------------

def build_verite_label_mapping(csv_path: str):
    """
    Build a dict: claim_id (string) -> label (string)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded VERITE CSV from {csv_path}")
    print("Columns:", list(df.columns))

    # --- Find ID column ---
    # Prefer 'Unnamed: 0', but fall back to other reasonable options.
    id_col = None
    cols_lower = {c.lower(): c for c in df.columns}

    for cand in ["unnamed: 0", "claim_id", "id", "index"]:
        if cand in cols_lower:
            id_col = cols_lower[cand]
            break

    if id_col is None:
        # As a last resort, we will use the DataFrame index
        print("⚠️  No explicit ID column found; using CSV row index as claim_id.")
        df["_tmp_id"] = df.index
        id_col = "_tmp_id"

    # --- Find label column ---
    label_col = None
    for cand in ["label", "veracity_label"]:
        if cand in cols_lower:
            label_col = cols_lower[cand]
            break

    if label_col is None:
        raise ValueError(
            "Could not find a label column in VERITE CSV "
            "(tried: 'label', 'veracity_label'). "
            f"Found columns: {list(df.columns)}"
        )

    mapping = {}
    for _, row in df[[id_col, label_col]].iterrows():
        cid = str(row[id_col]).strip()
        lab = str(row[label_col]).strip()
        if cid and lab:
            mapping[cid] = lab

    print(f"✅ Built label mapping for {len(mapping)} rows")
    return mapping


# -------------------------
# 2. Add labels to JSON
# -------------------------

def add_labels_to_json(json_in_path: str,
                       json_out_path: str,
                       label_map: dict,
                       default_label: str | None = None):
    """
    Read JSON, add 'label' field using label_map[claim_id],
    and write a new JSON file.
    """
    if not os.path.exists(json_in_path):
        raise FileNotFoundError(f"JSON file not found: {json_in_path}")

    with open(json_in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"\nProcessing JSON: {json_in_path}  ({len(data)} items)")

    missing = 0
    updated = 0

    for item in data:
        cid = str(item.get("claim_id", "")).strip()
        if cid in label_map:
            item["label"] = label_map[cid]
            updated += 1
        else:
            missing += 1
            if default_label is not None:
                item.setdefault("label", default_label)

    os.makedirs(os.path.dirname(json_out_path) or ".", exist_ok=True)
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ Wrote: {json_out_path}")
    print(f"   Updated with label: {updated}")
    print(f"   Missing label:      {missing}")


# -------------------------
# 3. Main (edit paths here)
# -------------------------

if __name__ == "__main__":
    # 👉 EDIT THESE PATHS TO MATCH YOUR SETUP

    CSV_PATH = "/data4/adityaks/multimodal_fact_checking/other_datasets/image-text-verification/VERITE/VERITE.csv"

    # Example: one file
    # JSON_FILES_IN  = ["results/verite_justifications/qwen2vl_verite_full_train.json"]
    # JSON_FILES_OUT = ["results/verite_justifications/qwen2vl_verite_full_train_with_labels.json"]

    # Example: train / val / test splits
    JSON_FILES_IN = [
        "/data4/adityaks/multimodal_fact_checking/other_datasets/results/verite_justifications/idefics3_verite_full_train.json",
        "/data4/adityaks/multimodal_fact_checking/other_datasets/results/verite_justifications/idefics3_verite_full_val.json",
        "/data4/adityaks/multimodal_fact_checking/other_datasets/results/verite_justifications/idefics3_verite_full_test.json",
    ]
    JSON_FILES_OUT = [
        "/data4/adityaks/multimodal_fact_checking/other_datasets/results/verite_justifications/idefics3_verite_full_train_with_labels.json",
        "/data4/adityaks/multimodal_fact_checking/other_datasets/results/verite_justifications/idefics3_verite_full_val_with_labels.json",
        "/data4/adityaks/multimodal_fact_checking/other_datasets/results/verite_justifications/idefics3_verite_full_test_with_labels.json",
    ]

    # Build mapping from CSV
    label_map = build_verite_label_mapping(CSV_PATH)

    # Attach labels to each JSON file
    for jin, jout in zip(JSON_FILES_IN, JSON_FILES_OUT):
        add_labels_to_json(jin, jout, label_map, default_label=None)

    print("\n🎉 Done. Your new JSON files now contain a 'label' field alongside claim_id/claim_text/etc.")

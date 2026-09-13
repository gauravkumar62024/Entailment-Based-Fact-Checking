import os
import pandas as pd
import json

# --- Step 1: Define dataset mapping ---
datasets = {
    "train": {
        "csv": "train.csv",
        "jsons": [
            "train_evidence_qwen2vl_justification.json",
            "train_justifications_idefics2.json",
            "train_justifications_paligemma2.json"
        ]
    },
    "val": {
        "csv": "val.csv",
        "jsons": [
            "val_evidence_qwen2vl_justification.json",
            "val_justifications_idefics2.json",
	    "val_justifications_paligemma2.json"
        ]
    },
    "test": {
        "csv": "test.csv",
        "jsons": [
            "test_evidence_qwen2vl_justification.json",
            "test_justifications_idefics2.json",
           "test_justifications_paligemma2.json"
        ]
    }
}

# --- Step 2: Create output folder ---
output_dir = "justification"
os.makedirs(output_dir, exist_ok=True)

# --- Step 3: Merge helper function ---
def merge_json_csv(json_path, csv_path, output_path):
    # Load data
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df_json = pd.DataFrame(data)
    df_csv = pd.read_csv(csv_path)

    # Normalize columns
    df_json.columns = df_json.columns.str.lower()
    df_csv.columns = df_csv.columns.str.lower()

    # Convert id to string
    if "id" not in df_json.columns or "id" not in df_csv.columns:
        print(f"⚠️ Skipping {json_path}: no 'id' column found.")
        return None
    df_json["id"] = df_json["id"].astype(str)
    df_csv["id"] = df_csv["id"].astype(str)

    # Merge (only matching ids)
    merged = pd.merge(df_json, df_csv, on="id", how="inner")

    # Combine claim columns if both exist
    if "claim_x" in merged.columns and "claim_y" in merged.columns:
        merged["claim_text"] = merged["claim_x"].fillna(merged["claim_y"])
    elif "claim" in merged.columns:
        merged.rename(columns={"claim": "claim_text"}, inplace=True)
    else:
        merged["claim_text"] = None

    # Rename other fields
    rename_map = {
        "support_justification": "supporting_text_evidence",
        "refute_justification": "refuting_text_evidence",
        "claim_image": "claim_image_url",
        "document_image": "document_image_url",
        "category": "label"
    }
    merged.rename(columns=rename_map, inplace=True)

    # Keep only desired fields
    final_cols = [
        "id",
        "claim_text",
        "supporting_text_evidence",
        "refuting_text_evidence",
        "label",
        "claim_ocr",
        "claim_image_url",
        "document_image_url"
    ]
    merged = merged[[c for c in final_cols if c in merged.columns]]

    # Save
    merged.to_json(output_path, orient="records", indent=4, force_ascii=False)

    return {
        "json_file": os.path.basename(json_path),
        "csv_file": os.path.basename(csv_path),
        "json_records": len(df_json),
        "csv_records": len(df_csv),
        "merged_records": len(merged),
        "output_file": output_path
    }

# --- Step 4: Run merges and track summary ---
summary = []

for split, info in datasets.items():
    csv_file = info["csv"]
    for json_file in info["jsons"]:
        name = os.path.splitext(os.path.basename(json_file))[0]
        output_path = os.path.join(output_dir, f"{name}_final_justification.json")
        stats = merge_json_csv(json_file, csv_file, output_path)
        if stats:
            summary.append(stats)
            print(f"✅ {name} → {stats['merged_records']} merged (of {stats['json_records']} JSON, {stats['csv_records']} CSV)")

# --- Step 5: Print summary table ---
print("\n📊 Merge Summary")
print("=" * 85)
print(f"{'JSON File':40s} {'CSV File':12s} {'JSON':>8s} {'CSV':>8s} {'Merged':>8s}")
print("-" * 85)
for s in summary:
    print(f"{s['json_file'][:40]:40s} {s['csv_file'][:12]:12s} {s['json_records']:8d} {s['csv_records']:8d} {s['merged_records']:8d}")
print("=" * 85)
print(f"✅ All merged files saved to folder: '{output_dir}/'")


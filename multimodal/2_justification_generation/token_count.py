import os
import json
import tiktoken
import pandas as pd
from statistics import mean

# --- Step 1: Initialize tokenizer ---
enc = tiktoken.get_encoding("cl100k_base")

# --- Step 2: Define input/output folders ---
input_dir = "justification"
output_dir = "justification"
os.makedirs(output_dir, exist_ok=True)

# --- Step 3: Token counting helper ---
def count_tokens(text):
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(enc.encode(text))

# --- Step 4: Process each final justification file ---
summary = []

for filename in os.listdir(input_dir):
    if filename.endswith("_final_justification.json"):
        path = os.path.join(input_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Add token counts per field
        for item in data:
            item["tokens_claim_text"] = count_tokens(item.get("claim_text", ""))
            item["tokens_claim_ocr"] = count_tokens(item.get("claim_ocr", ""))
            item["tokens_supporting"] = count_tokens(item.get("supporting_text_evidence", ""))
            item["tokens_refuting"] = count_tokens(item.get("refuting_text_evidence", ""))
            item["total_tokens"] = (
                item["tokens_claim_text"]
                + item["tokens_claim_ocr"]
                + item["tokens_supporting"]
                + item["tokens_refuting"]
            )

        # Save all records with tokens
        output_path = os.path.join(output_dir, filename.replace("_final_justification", "_final_justification_with_tokens"))
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        # --- Filter >512 tokens ---
        over_512 = [d for d in data if d["total_tokens"] > 512]
        over_path = os.path.join(output_dir, filename.replace("_final_justification", "_over512"))
        with open(over_path, "w", encoding="utf-8") as f:
            json.dump(over_512, f, indent=4, ensure_ascii=False)

        # --- Collect stats ---
        def field_stats(field):
            vals = [d[field] for d in data if d[field] > 0]
            return {
                "min": min(vals) if vals else 0,
                "max": max(vals) if vals else 0,
                "avg": round(mean(vals), 2) if vals else 0
            }

        stats = {
            "file": filename,
            "records": len(data),
            "over_512": len(over_512),
            "claim_text": field_stats("tokens_claim_text"),
            "claim_ocr": field_stats("tokens_claim_ocr"),
            "support": field_stats("tokens_supporting"),
            "refute": field_stats("tokens_refuting"),
            "total": field_stats("total_tokens"),
        }
        summary.append(stats)

        print(f"✅ {filename}: {len(data)} samples, {len(over_512)} >512 tokens")

# --- Step 5: Print summary ---
print("\n📊 Token Count Summary (Min / Avg / Max)")
print("=" * 110)
print(f"{'File':45s} {'#Rec':>6s} {'>512':>6s} {'Claim':>15s} {'Claim OCR':>15s} {'Support':>15s} {'Refute':>15s} {'Total':>15s}")
print("-" * 110)
for s in summary:
    print(f"{s['file'][:45]:45s} "
          f"{s['records']:6d} {s['over_512']:6d} "
          f"{s['claim_text']['min']:>3d}/{s['claim_text']['avg']:>5.1f}/{s['claim_text']['max']:>4d} "
          f"{s['claim_ocr']['min']:>3d}/{s['claim_ocr']['avg']:>5.1f}/{s['claim_ocr']['max']:>4d} "
          f"{s['support']['min']:>3d}/{s['support']['avg']:>5.1f}/{s['support']['max']:>4d} "
          f"{s['refute']['min']:>3d}/{s['refute']['avg']:>5.1f}/{s['refute']['max']:>4d} "
          f"{s['total']['min']:>3d}/{s['total']['avg']:>5.1f}/{s['total']['max']:>4d}")
print("=" * 110)

# --- Step 6: Optional - Save summary CSV ---
summary_records = []
for s in summary:
    summary_records.append({
        "file": s["file"],
        "records": s["records"],
        "over_512": s["over_512"],
        "claim_text_min": s["claim_text"]["min"],
        "claim_text_avg": s["claim_text"]["avg"],
        "claim_text_max": s["claim_text"]["max"],
        "claim_ocr_min": s["claim_ocr"]["min"],
        "claim_ocr_avg": s["claim_ocr"]["avg"],
        "claim_ocr_max": s["claim_ocr"]["max"],
        "support_min": s["support"]["min"],
        "support_avg": s["support"]["avg"],
        "support_max": s["support"]["max"],
        "refute_min": s["refute"]["min"],
        "refute_avg": s["refute"]["avg"],
        "refute_max": s["refute"]["max"],
        "total_min": s["total"]["min"],
        "total_avg": s["total"]["avg"],
        "total_max": s["total"]["max"],
    })

pd.DataFrame(summary_records).to_csv(os.path.join(output_dir, "token_summary.csv"), index=False)
print(f"\n📄 Detailed summary saved to {os.path.join(output_dir, 'token_summary.csv')}")


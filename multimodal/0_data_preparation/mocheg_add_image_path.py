import json

# === Input and output files ===
pred_file = "outputs/paligemma/test_predictions.json"            # Step 1 output (no paths yet)
paths_file = "outputs/test_image_paths_only.json"      # File with image paths
output_file = "outputs/paligemma/test_predictions_with_paths.json"

# === Load JSON files ===
with open(pred_file, "r", encoding="utf-8") as f:
    predictions = json.load(f)

with open(paths_file, "r", encoding="utf-8") as f:
    img_data = json.load(f)

# === Build lookup from image data ===
img_map = {str(item["Id"]).strip(): item.get("document_image_path", "")
           for item in img_data}

# === Merge and track stats ===
matched, missing = 0, 0
merged = []

for item in predictions:
    cid = str(item.get("Id", "")).strip()
    if cid in img_map and img_map[cid]:
        item["document_image_path"] = img_map[cid]
        matched += 1
    else:
        item["document_image_path"] = ""
        missing += 1
    merged.append(item)

# === Save merged output ===
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)

# === Summary report ===
print("\n=== Merge Summary ===")
print(f"🔗 Total predictions: {len(predictions)}")
print(f"✅ Matched image paths: {matched}")
print(f"⚠️ Missing image paths: {missing}")
print(f"💾 Saved merged file: {output_file}")
print("======================\n")

# Optional: list a few missing IDs for quick debug
if missing > 0:
    missing_ids = [str(item.get('Id', '')) for item in predictions if not item.get('document_image_path')]
    print("Example missing IDs:", missing_ids[:10])

